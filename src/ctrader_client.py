# ctrader_client.py

from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
from ctrader_open_api.messages.OpenApiMessages_pb2 import (
    ProtoOAApplicationAuthReq,
    ProtoOAAccountAuthReq,
    ProtoOASymbolsListReq,
    ProtoOAReconcileReq,
    ProtoOAGetTrendbarsReq,
    ProtoOANewOrderReq,
    ProtoOAAmendOrderReq,
    ProtoOAAmendPositionSLTPReq,
    ProtoOAErrorRes,
    ProtoOAGetAccountListByAccessTokenReq,
    ProtoOAClosePositionReq,    # <- ADD THIS
)
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import (
    ProtoOAOrderType,
    ProtoOATradeSide,
    ProtoOATrendbarPeriod,
)

from twisted.internet import reactor
from datetime import datetime, timezone, timedelta
import calendar, time, threading, os, random, logging
from dotenv import load_dotenv
import numpy as np
import pandas as pd  # for the df helpers

# ───────────────────────────────────────────────────────────────────────────
# Credentials & Client Bootstrap
# ───────────────────────────────────────────────────────────────────────────

load_dotenv()

def _env_clean(s: str | None) -> str | None:
    if s is None:
        return None
    return s.strip().strip('"').strip("'")

CLIENT_ID     = _env_clean(os.getenv("CTRADER_CLIENT_ID"))
CLIENT_SECRET = _env_clean(os.getenv("CTRADER_CLIENT_SECRET"))
ACCESS_TOKEN  = _env_clean(os.getenv("CTRADER_ACCESS_TOKEN"))
HOST_TYPE_RAW = _env_clean(os.getenv("CTRADER_HOST_TYPE")) or "demo"
HOST_TYPE     = HOST_TYPE_RAW.lower()

acct_raw = _env_clean(os.getenv("CTRADER_ACCOUNT_ID"))
try:
    ACCOUNT_ID = int(acct_raw) if acct_raw else 0
except ValueError:
    ACCOUNT_ID = 0

print("[BOOT] host_type =", HOST_TYPE)
print("[BOOT] account_id =", ACCOUNT_ID)
print("[BOOT] token_len =", len(ACCESS_TOKEN) if ACCESS_TOKEN else 0)
print("[BOOT] token_head =", (ACCESS_TOKEN or "")[:12])
print("[BOOT] client_id_head =", (CLIENT_ID or "")[:8])

host = EndPoints.PROTOBUF_LIVE_HOST if HOST_TYPE == "live" else EndPoints.PROTOBUF_DEMO_HOST
client = Client(host, EndPoints.PROTOBUF_PORT, TcpProtocol)

# Reconnect / retry config
RECONNECT_ENABLED   = (os.getenv("RECONNECT_ENABLED", "1").strip().lower() in {"1","true","yes","on"})
RECONNECT_BASE_SEC  = int(os.getenv("RECONNECT_BASE_SEC", "5"))
RECONNECT_MAX_SEC   = int(os.getenv("RECONNECT_MAX_SEC", "60"))
RECONNECT_JITTER    = float(os.getenv("RECONNECT_JITTER_SEC", "1.5"))

REQ_RETRY_ATTEMPTS  = int(os.getenv("REQ_RETRY_ATTEMPTS", "3"))
REQ_RETRY_BACKOFF   = float(os.getenv("REQ_RETRY_BACKOFF", "0.75"))

LOG = logging.getLogger("ctrader_client")

# ───────────────────────────────────────────────────────────────────────────
# Symbol Maps & Ready Flags
# ───────────────────────────────────────────────────────────────────────────

symbol_map        : dict[int, str] = {}
symbol_name_to_id : dict[str, int] = {}
symbol_digits_map : dict[int, int] = {}
symbols_ready = threading.Event()

# ───────────────────────────────────────────────────────────────────────────
# Helpers (pips, prices, error handling)
# ───────────────────────────────────────────────────────────────────────────

def encode_price(price_float: float) -> int:
    """Float price -> int units (1/100000) used by some API fields."""
    return int(round(price_float * 100_000))

def decode_price(price_int: int) -> float:
    """Int API units (1/100000) -> float price."""
    return price_int / 100_000.0

def pip_size(symbol_id: int) -> float:
    """Approximate pip size based on digits mapping (default 1e-5)."""
    digits = symbol_digits_map.get(symbol_id, 5)
    return 10 ** (-digits)

def pips_to_relative_by_symbol(symbol_id: int, pips: int | float) -> int:
    """
    Convert pips -> 'relative' units (1e-5) for MARKET SL/TP fields.
    cTrader OPEN-API relative fields are in 1/100000 increments.
    """
    ps = pip_size(symbol_id)               # e.g., 1e-5 for most FX
    return int(round((pips * ps) / 1e-5))  # normalize to 1e-5 units

def _pretty_error(msg):
    try:
        return f"{msg.errorCode} – {msg.description}"
    except Exception:
        return f"Unknown error payload: {type(msg)}"

def on_error(failure):
    """Uniform Twisted error handler that decodes ProtoOAErrorRes if present."""
    try:
        maybe = getattr(failure, "value", None)
        args0 = getattr(maybe, "args", [None])[0]
        if args0 is not None:
            msg = Protobuf.extract(args0)
            if isinstance(msg, ProtoOAErrorRes):
                print("[ERROR]", _pretty_error(msg))
                return
    except Exception:
        pass
    # Fallback: stringified failure
    try:
        print("[ERROR] Twisted Failure:", failure.getErrorMessage())
    except Exception:
        print("[ERROR] Twisted Failure:", str(failure))

# ───────────────────────────────────────────────────────────────────────────
# Auth & Symbol Bootstrap
# ───────────────────────────────────────────────────────────────────────────

def symbols_response_cb(res):
    """Handle either a symbols list or an error response gracefully."""
    global symbol_map, symbol_name_to_id, symbol_digits_map

    payload = Protobuf.extract(res)

    # Error payload (varies by lib version / broker)
    if isinstance(payload, ProtoOAErrorRes) or (
        getattr(payload, "DESCRIPTOR", None) and payload.DESCRIPTOR.name == "ProtoOAErrorRes"
    ):
        print("[ERROR] Symbol list request failed:", _pretty_error(payload))
        # Try one quick retry at startup; if it fails again we leave 'symbols_ready' unset
        if not symbols_ready.is_set():
            _request_symbols(retry=True)
        return

    # Accept multiple possible field names
    iterable = []
    for field_name in ("symbol", "symbols"):
        if hasattr(payload, field_name):
            try:
                iterable = list(getattr(payload, field_name))
            except Exception:
                iterable = []
            if iterable:
                break
    if not iterable and isinstance(payload, (list, tuple)):
        iterable = list(payload)

    if not iterable:
        try:
            fields = payload.ListFields()
        except Exception:
            fields = "N/A"
        print("[ERROR] Symbols payload not iterable; type:", type(payload), "| fields:", fields)
        return

    symbol_map.clear(); symbol_name_to_id.clear(); symbol_digits_map.clear()
    added = 0
    for s in iterable:
        sid    = getattr(s, "symbolId", getattr(s, "id", None))
        name   = getattr(s, "symbolName", getattr(s, "name", None))
        digits = getattr(s, "digits", getattr(s, "pipPosition", 5))
        if sid is None or name is None:
            print("[WARN] Skipping malformed symbol entry:", s)
            continue
        try:
            sid = int(sid); digits = int(digits)
        except Exception:
            digits = 5
        symbol_map[sid] = name
        symbol_name_to_id[name.upper()] = sid
        symbol_digits_map[sid] = digits
        added += 1

    print(f"[DEBUG] Loaded {added} symbols (e.g. {list(symbol_map.items())[:3]})")
    symbols_ready.set()

def _request_symbols(retry: bool = False):
    """Send the SymbolsList request; when retry=True wait 1s and try again."""
    if retry:
        print("[INFO] Retrying symbol list in 1s…")
        time.sleep(1)
    req = ProtoOASymbolsListReq(
        ctidTraderAccountId=ACCOUNT_ID,
        includeArchivedSymbols=True  # safer across brokers
    )
    ensure_client_ready(timeout=20)
    client.send(req).addCallbacks(symbols_response_cb, on_error)

def _list_accounts_cb(res):
    payload = Protobuf.extract(res)
    if getattr(payload, "DESCRIPTOR", None) and payload.DESCRIPTOR.name == "ProtoOAErrorRes":
        raise RuntimeError(f"GetAccountList failed: {_pretty_error(payload)}")

    # Find accounts array field name safely across versions
    acc_fields = []
    for fname in ("ctidTraderAccount", "traderAccount", "account"):
        if hasattr(payload, fname):
            acc_fields = list(getattr(payload, fname))
            break
    if not acc_fields:
        try:
            field_names = [f[0].name for f in payload.ListFields()]
        except Exception:
            field_names = ["<unknown>"]
        raise RuntimeError(f"Unexpected GetAccountList payload. Fields={field_names}")

    ids = []
    for a in acc_fields:
        aid = getattr(a, "ctidTraderAccountId", None)
        if aid is None:
            aid = getattr(a, "traderAccountId", None)
        if aid is not None:
            ids.append(int(aid))

    print("[AUTH] token has accounts:", ids)
    if ACCOUNT_ID not in ids:
        raise RuntimeError(
            f"ACCOUNT_ID {ACCOUNT_ID} is NOT authorized by this ACCESS_TOKEN on host='{HOST_TYPE}'. "
            f"Use an access token for the same host and select this account during OAuth."
        )
    print(f"[AUTH] OK: ACCOUNT_ID {ACCOUNT_ID} is authorized by token.")

def account_auth_cb(_):
    # After account auth → request symbols
    _request_symbols()

def app_auth_cb(_):
    # 1) Verify the token really grants this account
    client.send(ProtoOAGetAccountListByAccessTokenReq(accessToken=ACCESS_TOKEN))\
          .addCallbacks(_list_accounts_cb, on_error)
    # 2) Then do account auth
    req = ProtoOAAccountAuthReq(ctidTraderAccountId=ACCOUNT_ID, accessToken=ACCESS_TOKEN)
    client.send(req).addCallbacks(account_auth_cb, on_error)

def connected(_):
    req = ProtoOAApplicationAuthReq(clientId=CLIENT_ID, clientSecret=CLIENT_SECRET)
    client.send(req).addCallbacks(app_auth_cb, on_error)
    # reset reconnect attempts on successful connect
    globals().setdefault("_reconnect_attempt", 0)
    _reconnect_attempt = 0

def _client_ready() -> bool:
    try:
        return client is not None and reactor.running
    except Exception:
        return False

def init_client():
    symbols_ready.clear()
    client.setConnectedCallback(connected)
    def _on_disc(c, reason):
        print("[INFO] Disconnected:", reason)
        LOG.info("Disconnected: %s", reason)
        if not RECONNECT_ENABLED:
            return
        # exponential backoff with jitter
        globals().setdefault("_reconnect_attempt", 0)
        delay = min(RECONNECT_BASE_SEC * (2 ** _reconnect_attempt), RECONNECT_MAX_SEC)
        delay = delay + random.random() * RECONNECT_JITTER
        _reconnect_attempt += 1
        try:
            reactor.callLater(delay, lambda: (client.startService()))
        except Exception as ex:
            print("[WARN] schedule reconnect failed:", ex)
            LOG.warning("schedule reconnect failed: %s", ex)
    client.setDisconnectedCallback(_on_disc)
    client.setMessageReceivedCallback(lambda c, m: None)
    client.startService()
    reactor.run(installSignalHandlers=False)

# ───────────────────────────────────────────────────────────────────────────
# Convenience Bootstrap
# ───────────────────────────────────────────────────────────────────────────

def start_background():
    """Start the cTrader client in background if not running."""
    if not reactor.running:
        threading.Thread(target=init_client, daemon=True).start()

def wait_until_symbols_loaded(timeout: int = 20) -> None:
    """Wait for symbol list. Raises if not ready in timeout."""
    if symbols_ready.wait(timeout=timeout):
        return
    raise TimeoutError("Symbols not loaded. Check earlier [ERROR] lines for errorCode/description.")

def ensure_client_ready(timeout: int = 20):
    start_background()
    wait_until_symbols_loaded(timeout)

def _ensure_and_send(req, *, client_arg=None, timeout: int | None = None):
    """Ensure client is ready and send a request (one retry on AttributeError)."""
    ensure_client_ready(timeout=20)
    c = client_arg or client
    try:
        if timeout is None:
            return c.send(req)
        return c.send(req, timeout=timeout)
    except AttributeError:
        time.sleep(1.0)
        ensure_client_ready(timeout=20)
        c = client_arg or client
        if timeout is None:
            return c.send(req)
        return c.send(req, timeout=timeout)

# ───────────────────────────────────────────────────────────────────────────
# OHLC Fetchers
# ───────────────────────────────────────────────────────────────────────────

daily_bars, ready_event = [], threading.Event()

def _trendbars_cb(res):
    bars = Protobuf.extract(res).trendbar
    def _tb(tb):
        ts = datetime.fromtimestamp(tb.utcTimestampInMinutes * 60, timezone.utc)
        return dict(
            time   = ts.isoformat(),
            open   = (tb.low + tb.deltaOpen)   / 100_000,
            high   = (tb.low + tb.deltaHigh)   / 100_000,
            low    = tb.low                    / 100_000,
            close  = (tb.low + tb.deltaClose)  / 100_000,
            volume = tb.volume,
        )
    global daily_bars
    daily_bars = list(map(_tb, bars))[-50:]
    ready_event.set()

def get_ohlc_data(symbol: str, tf: str = "D1", n: int = 10):
    """Return dict with candles/context/trend. Safe for concurrent calls."""
    sid = symbol_name_to_id.get(symbol.upper())
    if sid is None:
        raise ValueError(f"Unknown symbol '{symbol}'")

    evt = threading.Event()
    rows = []

    def _cb(res):
        bars = Protobuf.extract(res).trendbar
        for tb in bars:
            ts = datetime.fromtimestamp(tb.utcTimestampInMinutes * 60, timezone.utc)
            rows.append(dict(
                time   = ts.isoformat(),
                open   = (tb.low + tb.deltaOpen)   / 100_000,
                high   = (tb.low + tb.deltaHigh)   / 100_000,
                low    = tb.low                    / 100_000,
                close  = (tb.low + tb.deltaClose)  / 100_000,
                volume = tb.volume,
            ))
        evt.set()

    now = datetime.utcnow()
    req = ProtoOAGetTrendbarsReq(
        symbolId            = sid,
        ctidTraderAccountId = ACCOUNT_ID,
        period              = getattr(ProtoOATrendbarPeriod, tf.upper()),
        fromTimestamp       = int(calendar.timegm((now - timedelta(weeks=52)).utctimetuple())) * 1000,
        toTimestamp         = int(calendar.timegm(now.utctimetuple())) * 1000,
    )
    # Simple retry loop on timeout
    attempts = 0
    while True:
        evt.clear(); rows.clear()
        ensure_client_ready(timeout=20)
        client.send(req).addCallbacks(_cb, on_error)
        if evt.wait(10):
            break
        attempts += 1
        if attempts >= max(1, REQ_RETRY_ATTEMPTS):
            raise TimeoutError("Trendbars request timed out")
        backoff = REQ_RETRY_BACKOFF * (2 ** (attempts - 1))
        time.sleep(backoff)

    candles = rows[-n:] if n else rows
    highs   = [c["high"]  for c in candles]
    lows    = [c["low"]   for c in candles]
    closes  = [c["close"] for c in candles]

    context_levels = {}
    if len(candles) >= 2:
        context_levels = {
            "today_high": candles[-1]["high"],
            "today_low": candles[-1]["low"],
            "prev_day_high": candles[-2]["high"],
            "prev_day_low": candles[-2]["low"],
            "range_high_5": max(highs[-5:]) if len(highs) >= 5 else max(highs),
            "range_low_5":  min(lows[-5:])  if len(lows)  >= 5 else min(lows),
        }

    trend_strength = {}
    if tf.upper() in ("D1", "H4") and len(closes) >= 5:
        x = np.arange(len(closes))
        slope, _ = np.polyfit(x, closes, 1)
        r = np.corrcoef(x, closes)[0, 1]
        trend_strength = {
            "slope": float(slope),
            "correlation": float(r),
            "confidence": (
                "Ultra Strong Bullish" if slope > 0.5 and r > 0.9 else
                "Strong Bearish" if slope < -0.5 and r > 0.9 else
                "Sideways/Neutral"
            )
        }

    return {"candles": candles, "context": context_levels, "trend": trend_strength}

def get_ohlc_df(symbol: str, tf: str = "H1", n: int = 2000) -> pd.DataFrame:
    """Return OHLCV as a UTC-indexed DataFrame (open, high, low, close, volume)."""
    out = get_ohlc_data(symbol, tf, n)
    rows = out.get("candles", [])
    if not rows:
        return pd.DataFrame(columns=["open","high","low","close","volume"]).astype(float)
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time")[["open","high","low","close","volume"]].astype(float)
    return df.sort_index()

# ───────────────────────────────────────────────────────────────────────────
# Positions & Reconcile
# ───────────────────────────────────────────────────────────────────────────

open_positions, pos_ready = [], threading.Event()

def _reconcile_cb(res):
    global open_positions
    open_positions = []
    rec = Protobuf.extract(res)
    for p in rec.position:
        td = p.tradeData
        open_positions.append(
            dict(
                symbol_name = symbol_map.get(td.symbolId, str(td.symbolId)),
                position_id = p.positionId,
                direction   = "buy" if td.tradeSide == ProtoOATradeSide.BUY else "sell",
                entry_price = getattr(p, "price", 0),
                volume_lots = td.volume / 10_000_000,   # 1 lot = 10,000,000
                volume_units = td.volume,               # <- ADD THIS (native units)
            )
        )
    pos_ready.set()

def get_open_positions():
    pos_ready.clear()
    req = ProtoOAReconcileReq(ctidTraderAccountId = ACCOUNT_ID)
    client.send(req).addCallbacks(_reconcile_cb, on_error)
    pos_ready.wait(5)
    return open_positions

def is_forex_symbol(symbol: str) -> bool:
    """Treat majors as Forex; expand as needed."""
    return symbol.upper() in {
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD", "USDCHF", "USDCAD",
        "EURJPY", "EURGBP", "GBPJPY"
    }

# ───────────────────────────────────────────────────────────────────────────
# Order Placement & Amend
# ───────────────────────────────────────────────────────────────────────────

def place_order(
    *, client, account_id, symbol_id,
    order_type, side, volume,
    price=None, stop_loss=None, take_profit=None,
    client_msg_id=None,
):
    """
    Place MARKET/LIMIT/STOP order.
    - volume in native units (1 lot = 10,000,000).
    - LIMIT/STOP prices & SL/TP absolute floats (e.g., 1.17700).
    - MARKET SL/TP sent as 'relative' (1e-5) distances; optional absolute SL/TP patch follows fill.
    """
    req = ProtoOANewOrderReq(
        ctidTraderAccountId=account_id,
        symbolId=symbol_id,
        orderType=ProtoOAOrderType.Value(order_type.upper()),
        tradeSide=ProtoOATradeSide.Value(side.upper()),
        volume=int(volume),
    )

    if order_type.upper() == "LIMIT":
        if price is None:
            raise ValueError("Limit order requires price.")
        req.limitPrice = float(price)
        if stop_loss   is not None: req.stopLoss   = float(stop_loss)
        if take_profit is not None: req.takeProfit = float(take_profit)

    elif order_type.upper() == "STOP":
        if price is None:
            raise ValueError("Stop order requires price.")
        req.stopPrice = float(price)
        if stop_loss   is not None: req.stopLoss   = float(stop_loss)
        if take_profit is not None: req.takeProfit = float(take_profit)

    else:  # MARKET
        if stop_loss   is not None: req.relativeStopLoss   = pips_to_relative_by_symbol(symbol_id, int(stop_loss))
        if take_profit is not None: req.relativeTakeProfit = pips_to_relative_by_symbol(symbol_id, int(take_profit))

    print(f"[DEBUG] Sending order: order_type='{order_type}' side='{side}' price={price} SL={stop_loss} TP={take_profit}")
    d = _ensure_and_send(req, client_arg=client, timeout=30)

    # Optional delayed SL/TP patch after MARKET fill (convert pips → absolute)
    if order_type.upper() == "MARKET" and (stop_loss is not None or take_profit is not None):
        def _delayed_sltp(_):
            time.sleep(8)
            open_pos = get_open_positions()
            for p in open_pos:
                if (
                    p["symbol_name"].upper() == symbol_map[symbol_id].upper()
                    and p["direction"].upper() == side.upper()
                ):
                    try:
                        entry = float(p.get("entry_price", 0.0))
                        ps = pip_size(symbol_id)
                        sl_abs = None
                        tp_abs = None
                        if stop_loss is not None:
                            sl_abs = entry - int(stop_loss)*ps if side.upper() == "BUY" else entry + int(stop_loss)*ps
                        if take_profit is not None:
                            tp_abs = entry + int(take_profit)*ps if side.upper() == "BUY" else entry - int(take_profit)*ps
                        return modify_position_sltp(
                            client=client,
                            account_id=account_id,
                            position_id=p["position_id"],
                            stop_loss=sl_abs,
                            take_profit=tp_abs,
                        )
                    except Exception as ex:
                        print("[WARN] SLTP patch failed:", ex)
                        return {"status": "sltp_patch_failed", "error": str(ex)}
            return {"status": "position_not_found"}
        d.addCallback(_delayed_sltp)

    return d

# ───────────────────────────────────────────────────────────────────────────
# Close helpers
# ───────────────────────────────────────────────────────────────────────────

def close_position(*, client, account_id, position_id, volume_units: int | None = None, client_msg_id=None):
    """
    Close a position by ID. If volume_units is None → full close.
    volume_units are native units (1 lot = 10,000,000).
    """
    req = ProtoOAClosePositionReq(
        ctidTraderAccountId = account_id,
        positionId          = position_id,
    )
    if volume_units is not None:
        req.volume = int(volume_units)

    return _ensure_and_send(req, client_arg=client, timeout=30)



def modify_position_sltp(client, account_id, position_id, stop_loss=None, take_profit=None):
    req = ProtoOAAmendPositionSLTPReq(ctidTraderAccountId = account_id, positionId = position_id)
    if stop_loss   is not None: req.stopLoss   = float(stop_loss)
    if take_profit is not None: req.takeProfit = float(take_profit)
    return _ensure_and_send(req, client_arg=client)

def modify_pending_order_sltp(client, account_id, order_id, version, stop_loss=None, take_profit=None):
    req = ProtoOAAmendOrderReq(
        ctidTraderAccountId = account_id,
        orderId             = order_id,
        version             = version,
    )
    if stop_loss   is not None: req.stopLoss   = float(stop_loss)
    if take_profit is not None: req.takeProfit = float(take_profit)
    return _ensure_and_send(req, client_arg=client)

# ───────────────────────────────────────────────────────────────────────────
# Deferred Wait Helper (blocking)
# ───────────────────────────────────────────────────────────────────────────

def wait_for_deferred(d, timeout=30):
    """
    Block for a Twisted Deferred; return result or a dict describing failure.
    """
    evt, box = threading.Event(), {}
    d.addCallbacks(lambda r: (box.setdefault("r", r), evt.set()),
                   lambda f: (box.setdefault("f", f), evt.set()))
    evt.wait(timeout)
    if "r" in box:
        return box["r"]
    f = box.get("f")
    return {
        "status": "failed",
        "error": str(f),
        "failure_type": getattr(getattr(f, "type", None), "__name__", None)
    }

# ───────────────────────────────────────────────────────────────────────────
# Pending Orders Snapshot
# ───────────────────────────────────────────────────────────────────────────

def get_pending_orders():
    """Return all pending LIMIT/STOP orders via reconcile."""
    result_ready = threading.Event()
    pending_orders = []

    def callback(response):
        res = Protobuf.extract(response)
        for o in res.order:
            order_type = "LIMIT" if o.orderType == ProtoOAOrderType.LIMIT else "STOP"
            direction = "buy" if o.tradeData.tradeSide == ProtoOATradeSide.BUY else "sell"

            entry_price = None
            if hasattr(o, "limitPrice"):
                entry_price = decode_price(o.limitPrice)
            elif hasattr(o, "stopPrice"):
                entry_price = decode_price(o.stopPrice)

            symbol_id = o.tradeData.symbolId
            timestamp_ms = getattr(o, "orderTimestamp", None) or getattr(o, "lastUpdateTimestamp", 0)
            pending_orders.append({
                "order_id": o.orderId,
                "symbol_id": symbol_id,
                "symbol_name": symbol_map.get(symbol_id, str(symbol_id)),
                "direction": direction,
                "order_type": order_type,
                "entry_price": entry_price,
                "stop_loss": getattr(o, "stopLoss", None),
                "take_profit": getattr(o, "takeProfit", None),
                "volume": o.tradeData.volume,
                "creation_time": datetime.utcfromtimestamp(timestamp_ms / 1000).isoformat()
            })
        result_ready.set()

    req = ProtoOAReconcileReq(ctidTraderAccountId=ACCOUNT_ID)
    d = client.send(req)
    d.addCallbacks(callback, on_error)
    result_ready.wait(timeout=12)
    return {"orders": pending_orders}
