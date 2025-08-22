# ============================================
# Live Trading — cTrader + saved ML pipelines
# ============================================

import os, time, logging, sqlite3, warnings, signal, threading, json
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv
from zoneinfo import ZoneInfo

warnings.filterwarnings("ignore")

# --- our libs (this file lives in src/, so these import siblings)
from feature_engineering import add_core_features
from ctrader_client import (
    ensure_client_ready, get_ohlc_df, place_order, get_open_positions,
    symbol_name_to_id, wait_for_deferred, close_position,
    client as CTR_CLIENT, ACCOUNT_ID as CTR_ACCOUNT_ID
)
from notion_journal import NotionJournal, TradeEvent  # make sure src/notion_journal.py exists

# ---------- graceful shutdown ----------
STOP = threading.Event()

def _graceful(signum, _frame):
    log(f"Signal {signum} received → graceful shutdown…")
    STOP.set()

signal.signal(signal.SIGINT, _graceful)   # Ctrl+C
signal.signal(signal.SIGTERM, _graceful)  # docker stop

LOCAL_TZ = os.getenv("LOCAL_TZ", "Europe/Zurich")

def utc_to_local(ts: pd.Timestamp) -> pd.Timestamp:
    if ts is None:
        return ts
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(ZoneInfo(LOCAL_TZ))


# -------------------
# Config (env-driven with sane defaults)
# -------------------
load_dotenv()  # local runs; in Docker, compose injects env

# ---- Logging (init first so log() is available)
LOG_FILE = Path("logs/live_trader.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    filename=str(LOG_FILE),
    level=getattr(logging, _LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def log(msg: str) -> None:
    print(msg, flush=True)
    logging.info(msg)

# ---- Env helpers
def _env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}

def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, "").strip())
    except Exception:
        return default

def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, "").strip())
    except Exception:
        return default

def _env_list(key: str, default: str) -> list[str]:
    raw = os.getenv(key, default)
    parts: list[str] = []
    for tok in str(raw).replace(";", ",").split(","):
        parts.extend(tok.split())
    return [p.strip().upper() for p in parts if p.strip()]

def _parse_lots_map(key: str, default_map: dict[str, float]) -> dict[str, float]:
    """LOTS=EURUSD:0.10,GBPUSD:0.05"""
    raw = os.getenv(key, "")
    if not raw:
        return default_map
    out: dict[str, float] = {}
    for item in raw.split(","):
        if ":" not in item:
            continue
        sym, val = item.split(":", 1)
        sym = sym.strip().upper()
        try:
            out[sym] = float(val)
        except Exception:
            continue
    return out or default_map

def _parse_lots_json(key: str):
    """
    LOTS_JSON={"EURUSD":0.10,"GBPUSD":0.05,"DEFAULT":0.10}
    Returns (lots_map, default_or_None) or None if missing/invalid.
    """
    raw = os.getenv(key, "")
    if not raw:
        return None
    try:
        data = json.loads(raw)
        out = {}
        default = data.get("DEFAULT", None)
        if default is not None:
            default = float(default)
        for k, v in data.items():
            if k.upper() == "DEFAULT":
                continue
            out[k.upper()] = float(v)
        return out, default
    except Exception:
        return None

# ---- Core config
TF        = os.getenv("TF", "H1").upper()
SYMBOLS   = _env_list("SYMBOLS", "EURUSD,GBPUSD,AUDUSD")
N_BARS    = _env_int("N_BARS", 2500)
N_FORWARD = _env_int("N_FORWARD", 3)
SLEEP_SEC = _env_int("SLEEP_SEC", 60)

CLOSE_ON_FLAT    = _env_bool("CLOSE_ON_FLAT", True)
ALLOW_PYRAMIDING = _env_bool("ALLOW_PYRAMIDING", False)
MAX_POS_PER_SIDE = _env_int("MAX_POS_PER_SIDE", 1)

# Sizing
DEFAULT_LOTS = _env_float("DEFAULT_LOTS", 0.10)
LOTS = {s: DEFAULT_LOTS for s in SYMBOLS}

parsed_json = _parse_lots_json("LOTS_JSON")
if parsed_json:
    lots_map, default_override = parsed_json
    if default_override is not None:
        DEFAULT_LOTS = default_override
        LOTS = {s: DEFAULT_LOTS for s in SYMBOLS}
    LOTS.update(lots_map)
else:
    LOTS.update(_parse_lots_map("LOTS", LOTS))

# SL/TP pips (optional; blank → None)
SL_PIPS = os.getenv("SL_PIPS")
TP_PIPS = os.getenv("TP_PIPS")
SL_PIPS = None if SL_PIPS in (None, "", "None") else int(float(SL_PIPS))
TP_PIPS = None if TP_PIPS in (None, "", "None") else int(float(TP_PIPS))

# Paths
MODEL_FOLDER = Path(f"models/{TF.lower()}_models")

# Timeframe delta map
TF_DELTA = {
    "M1":  pd.Timedelta(minutes=1),
    "M5":  pd.Timedelta(minutes=5),
    "M15": pd.Timedelta(minutes=15),
    "M30": pd.Timedelta(minutes=30),
    "H1":  pd.Timedelta(hours=1),
    "H4":  pd.Timedelta(hours=4),
    "D1":  pd.Timedelta(days=1),
}.get(TF, pd.Timedelta(hours=1))

def lots_to_volume_units(lots: float) -> int:
    """1 lot = 10,000,000 native units in cTrader."""
    return int(round(lots * 10_000_000))

# ---- Startup banner
log(
    "Config → "
    f"TF={TF}, SYMBOLS={SYMBOLS}, N_BARS={N_BARS}, N_FORWARD={N_FORWARD}, "
    f"SLEEP_SEC={SLEEP_SEC}, CLOSE_ON_FLAT={CLOSE_ON_FLAT}, "
    f"ALLOW_PYRAMIDING={ALLOW_PYRAMIDING}, MAX_POS_PER_SIDE={MAX_POS_PER_SIDE}, "
    f"DEFAULT_LOTS={DEFAULT_LOTS}, LOTS={LOTS}, SL_PIPS={SL_PIPS}, TP_PIPS={TP_PIPS}"
)

# -------------------
# Helpers
# -------------------
def load_model_bundle(path: Path):
    """Supports {'pipeline','features'} or legacy {'model','scaler','features'}."""
    b = joblib.load(path)
    if isinstance(b, dict) and "pipeline" in b:
        return dict(kind="pipeline", pipeline=b["pipeline"], features=b.get("features"))
    if isinstance(b, dict) and "model" in b and "scaler" in b:
        return dict(kind="legacy", model=b["model"], scaler=b["scaler"], features=b.get("features"))
    # Also allow direct Pipeline (back-compat)
    if hasattr(b, "predict"):
        return dict(kind="pipeline", pipeline=b, features=None)
    raise ValueError(f"Unrecognized bundle at {path}")

def to_signals(preds: np.ndarray) -> np.ndarray:
    """Map {0,1,2} → {-1,0,+1}; pass-through if already in -1/0/+1."""
    preds = np.asarray(preds)
    uniq = set(np.unique(preds))
    return preds - 1 if uniq.issubset({0,1,2}) else preds.astype(int)

def save_signal_to_db(symbol: str, prediction: int, timestamp: str):
    conn = sqlite3.connect("live_signals.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT, prediction INTEGER, timestamp TEXT,
            UNIQUE(symbol, prediction, timestamp)
        )
    """)
    try:
        c.execute("INSERT OR IGNORE INTO signals (symbol,prediction,timestamp) VALUES (?,?,?)",
                  (symbol, int(prediction), timestamp))
        conn.commit()
    except Exception as e:
        log(f"[DB] insert failed: {e}")
    finally:
        conn.close()

def most_recent_bar_time(df: pd.DataFrame) -> pd.Timestamp | None:
    return None if df.empty else pd.to_datetime(df.index[-1])


# -------------------
# Trader
# -------------------
class LiveTrader:
    def __init__(self, symbol: str, lots: float, model_path: Path, journal: NotionJournal | None = None):
        self.symbol = symbol.upper()
        self.lots = float(lots)
        self.model_path = model_path
        self.bundle = None
        self.feature_cols = None
        self.last_bar_ts: pd.Timestamp | None = None
        self.journal = journal

    def load(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"No model file for {self.symbol}: {self.model_path}")
        self.bundle = load_model_bundle(self.model_path)
        self.feature_cols = self.bundle.get("features")
        log(f"✅ {self.symbol}: loaded model → {self.model_path.name}")

    def predict_multi(self, n_forward=N_FORWARD) -> tuple[np.ndarray, pd.Series]:
        """Return (preds[-N_FORWARD:], close_series aligned)."""
        df = get_ohlc_df(self.symbol, tf=TF, n=N_BARS)
        if df.empty:
            raise RuntimeError(f"{self.symbol}: no bars")
        df_feat = add_core_features(df.copy())
        # choose columns
        if self.feature_cols:
            use_cols = [c for c in self.feature_cols if c in df_feat.columns]
        else:
            core_cols = [
                "sma_20","ema_20","kama_10","rsi_14","macd_diff",
                "atr_14","obv","rolling_std_20","spread","fill","amplitude",
                "autocorr_1","autocorr_5","autocorr_10","market_regime","stationary_flag"
            ]
            use_cols = [c for c in core_cols if c in df_feat.columns]

        X = df_feat[use_cols].dropna()
        close = df_feat.loc[X.index, "close"]
        if X.empty:
            raise RuntimeError(f"{self.symbol}: no valid feature rows after FE")

        # predict
        if self.bundle["kind"] == "pipeline":
            preds_012 = self.bundle["pipeline"].predict(X)
        else:
            Xs = self.bundle["scaler"].transform(X)
            preds_012 = self.bundle["model"].predict(Xs)
        preds = to_signals(preds_012)

        # return last N_FORWARD signals and the aligned close series
        return preds[-n_forward:], close

    # ---- position administration (simple, netting-friendly) ----
    def _open_positions_for_symbol(self):
        pos = get_open_positions()
        return [p for p in pos if p.get("symbol_name","").upper() == self.symbol]

    def _has_dir(self, positions, side: str) -> bool:
        side = side.lower()
        return any((p.get("direction","").lower() == side) for p in positions)

    def _close_all(self) -> bool:
        """
        Close all open positions for this symbol using ClosePosition
        (works on hedging & netting).
        """
        positions = self._open_positions_for_symbol()
        if not positions:
            return True

        ok = True
        for p in positions:
            try:
                d = close_position(
                    client=CTR_CLIENT,
                    account_id=CTR_ACCOUNT_ID,
                    position_id=p["position_id"],
                    volume_units=p.get("volume_units"),  # full close in native units
                )
                res = wait_for_deferred(d, timeout=30)
                if isinstance(res, dict) and res.get("status") == "failed":
                    ok = False
                    log(f"[CLOSE] Fail pos {p['position_id']} {self.symbol}: {res}")
                else:
                    log(f"[CLOSE] OK pos {p['position_id']} {self.symbol}")
                    if self.journal:
                        self.journal.log_trade(TradeEvent(
                            event="CLOSE", symbol=self.symbol,
                            position_id=p["position_id"],
                            volume_units=p.get("volume_units"),
                            account_id=CTR_ACCOUNT_ID,
                            status=(res.get("status") if isinstance(res, dict) else "ok"),
                            # pnl=p.get("realized_pnl"), price=res.get("close_price")  # if available
                        ))
            except Exception as e:
                ok = False
                log(f"[CLOSE] Exception closing {self.symbol} pos {p.get('position_id')}: {e}")
        return ok

    def _ensure_direction(self, want: int):
        """
        Idempotent position policy (no pyramiding by default).
        want ∈ {-1, 0, +1}: +1=LONG, -1=SHORT, 0=FLAT
        """
        n_buy, n_sell = self._side_counts()
        log(f"{self.symbol}: state n_buy={n_buy}, n_sell={n_sell}, want={want}")

        # 1) FLAT → close everything (if configured)
        if want == 0:
            if (n_buy + n_sell) == 0:
                log(f"{self.symbol}: flat signal but already flat → no action")
                return
            if CLOSE_ON_FLAT:
                log(f"{self.symbol}: flat signal → closing open positions")
                self._close_all()
                self._wait_until_flat(timeout=20)
            else:
                log(f"{self.symbol}: flat signal, CLOSE_ON_FLAT=False → holding")
            return

        # 2) LONG
        if want > 0:
            # If any SELL exists, flip: close all then wait to be flat
            if n_sell > 0:
                log(f"{self.symbol}: flipping SELL→BUY, closing {n_sell} position(s)")
                self._close_all()
                if not self._wait_until_flat(timeout=20):
                    log(f"{self.symbol}: still not flat after timeout; skip open this bar")
                    return
                n_buy = n_sell = 0  # known flat now

            # Idempotency / pyramiding guard
            if not ALLOW_PYRAMIDING and n_buy > 0:
                log(f"{self.symbol}: already LONG → no action")
                return
            if ALLOW_PYRAMIDING and n_buy >= MAX_POS_PER_SIDE:
                log(f"{self.symbol}: LONG cap reached ({n_buy}/{MAX_POS_PER_SIDE}) → no action")
                return

            self._place_market("BUY")
            return

        # 3) SHORT (want < 0)
        if n_buy > 0:
            log(f"{self.symbol}: flipping BUY→SELL, closing {n_buy} position(s)")
            self._close_all()
            if not self._wait_until_flat(timeout=20):
                log(f"{self.symbol}: still not flat after timeout; skip open this bar")
                return
            n_buy = n_sell = 0

        if not ALLOW_PYRAMIDING and n_sell > 0:
            log(f"{self.symbol}: already SHORT → no action")
            return
        if ALLOW_PYRAMIDING and n_sell >= MAX_POS_PER_SIDE:
            log(f"{self.symbol}: SHORT cap reached ({n_sell}/{MAX_POS_PER_SIDE}) → no action")
            return

        self._place_market("SELL")

    def step(self):
        """One loop step: only act once per new bar."""
        preds, close = self.predict_multi(n_forward=N_FORWARD)
        last_ts = most_recent_bar_time(close.to_frame())  # UTC Timestamp
        if last_ts is None:
            return

        # gate executions to new bar
        if (self.last_bar_ts is not None) and (last_ts <= self.last_bar_ts):
            return
        self.last_bar_ts = last_ts

        # persist N_FORWARD predictions with future UTC timestamps (ISO8601)
        for i, p in enumerate(preds):
            future_utc = (last_ts + TF_DELTA * (i + 1))
            save_signal_to_db(self.symbol, int(p), timestamp=future_utc.isoformat())

        # trade first prediction
        sig = int(preds[0])  # {-1,0,+1}
        self._ensure_direction(sig)

        # log both UTC and local (Swiss) for readability
        bar_local = utc_to_local(last_ts)
        log(
            f"{self.symbol}: bar_utc={last_ts.isoformat()} | "
            f"bar_{LOCAL_TZ}={bar_local.strftime('%Y-%m-%d %H:%M:%S %Z')} | "
            f"signal={sig} | preds_next={preds.tolist()}"
        )

    def _current_side(self) -> int:
        """
        Return +1 if there's any BUY, -1 if any SELL, else 0.
        (If both exist on a hedging account, we consider it 0 → 'mixed'.)
        """
        positions = self._open_positions_for_symbol()
        has_buy  = self._has_dir(positions, "buy")
        has_sell = self._has_dir(positions, "sell")
        if has_buy and not has_sell:
            return +1
        if has_sell and not has_buy:
            return -1
        return 0

    def _side_counts(self) -> tuple[int, int]:
        """Return (#buy, #sell) open positions for this symbol."""
        pos = self._open_positions_for_symbol()
        n_buy  = sum(1 for p in pos if p.get("direction","").lower() == "buy")
        n_sell = sum(1 for p in pos if p.get("direction","").lower() == "sell")
        return n_buy, n_sell

    def _wait_until_flat(self, timeout: float = 20.0, poll: float = 0.5) -> bool:
        """Poll reconcile until there are no open positions for this symbol."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            n_buy, n_sell = self._side_counts()
            if (n_buy + n_sell) == 0:
                return True
            time.sleep(poll)
        return False

    def _place_market(self, side: str) -> bool:
        d = place_order(
            client=CTR_CLIENT, account_id=CTR_ACCOUNT_ID,
            symbol_id=symbol_name_to_id[self.symbol],
            order_type="MARKET", side=side.upper(),
            volume=lots_to_volume_units(self.lots),
            stop_loss=SL_PIPS, take_profit=TP_PIPS,
        )
        res = wait_for_deferred(d, timeout=30)
        failed = isinstance(res, dict) and res.get("status") == "failed"
        if failed:
            log(f"[ORDER] {self.symbol} {side.upper()} failed → {res}")
            if self.journal:
                self.journal.log_trade(TradeEvent(
                    event="OPEN", symbol=self.symbol, direction=side.upper(),
                    order_type="MARKET", lots=self.lots,
                    volume_units=lots_to_volume_units(self.lots),
                    account_id=CTR_ACCOUNT_ID, status="failed", note=str(res)
                ))
            return False

        log(("🟢 " if side.lower()=="buy" else "🔴 ") + f"{self.symbol}: {side.upper()} sent")
        if self.journal:
            self.journal.log_trade(TradeEvent(
                event="OPEN", symbol=self.symbol, direction=side.upper(),
                order_type="MARKET", lots=self.lots,
                volume_units=lots_to_volume_units(self.lots),
                stop_loss=SL_PIPS, take_profit=TP_PIPS,
                account_id=CTR_ACCOUNT_ID,
                status=(res.get("status") if isinstance(res, dict) else None),
                # price=res.get("price"), order_id=res.get("order_id")  # if available
            ))
        return True


# -------------------
# Main
# -------------------
if __name__ == "__main__":
    ensure_client_ready(timeout=20)
    log(f"cTrader ready. Symbols loaded: {len(symbol_name_to_id)}")

    traders: dict[str, LiveTrader] = {}
    journal = NotionJournal.from_env()

    for sym in SYMBOLS:
        lots = LOTS.get(sym.upper(), DEFAULT_LOTS)
        model_path = MODEL_FOLDER / f"{sym.upper()}_{TF}_best_model.pkl"
        t = LiveTrader(sym, lots, model_path, journal=journal)
        t.load()
        traders[sym] = t

    log("▶️  Live loop started.")
    try:
        while not STOP.is_set():
            for sym, t in traders.items():
                try:
                    t.step()
                except Exception as e:
                    log(f"⚠️ {sym}: step error → {e}")
            # sleep in small chunks so we can exit fast
            for _ in range(SLEEP_SEC):
                if STOP.is_set():
                    break
                time.sleep(1)
    finally:
        log("✅ Clean shutdown complete.")
