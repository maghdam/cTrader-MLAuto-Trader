import os
import os
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from control_store import load_map as ctrl_load, set_items as ctrl_set


DB_PATH = Path("live_signals.db")
TABLE = "signals"
LOG_PATH = Path("logs/live_trader.log")

st.set_page_config(page_title="cTrader ML Signals", layout="wide")
st.title("cTrader ML Auto-Trader — Live Signals")


@st.cache_data(ttl=30)
def load_signals() -> Optional[pd.DataFrame]:
    if not DB_PATH.exists():
        return None
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql(f"SELECT * FROM {TABLE} ORDER BY timestamp DESC", conn)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df
    except Exception as e:
        st.error(f"DB error: {e}")
        return None


def render_latest(df: pd.DataFrame) -> None:
    st.subheader("Latest per symbol")
    latest = df.sort_values("timestamp").groupby("symbol").tail(1).sort_values("symbol")
    st.dataframe(latest[["symbol", "prediction", "timestamp"]], use_container_width=True)


def render_recent(df: pd.DataFrame, n: int = 3) -> None:
    st.subheader(f"Last {n} per symbol")
    recent = (
        df.sort_values(["symbol", "timestamp"]).groupby("symbol").tail(n).sort_values(["symbol", "timestamp"])
    )
    st.dataframe(recent[["symbol", "prediction", "timestamp"]], use_container_width=True)

    st.write("---")
    st.subheader("Mini series")
    for sym in sorted(df["symbol"].dropna().unique()):
        mini = df[df["symbol"] == sym].sort_values("timestamp").tail(n)
        if mini.empty:
            continue
        st.write(f"**{sym}**")
        st.line_chart(mini.set_index("timestamp")["prediction"], height=120, use_container_width=True)


def render_dist(df: pd.DataFrame) -> None:
    st.subheader("Signal distribution (all time)")
    try:
        counts = df.groupby(["symbol", "prediction"]).size().unstack(fill_value=0)
        st.bar_chart(counts)
    except Exception:
        st.info("Not enough data to render distribution yet.")


def main() -> None:
    st.caption("-1=SELL, 0=FLAT, +1=BUY")

    # Sidebar controls
    st.sidebar.header("Controls")
    refresh_sec = st.sidebar.slider("Auto-refresh (sec)", min_value=5, max_value=60, value=int(os.getenv("DASH_REFRESH", "15")))
    st_autorefresh(interval=refresh_sec * 1000, key="autoref")

    ctrl = ctrl_load()
    paused = str(ctrl.get("PAUSE_TRADING", "0")).strip().lower() in {"1","true","yes","on"}
    with st.sidebar.form("ctrl_form"):
        st.subheader("Trader")
        paused_new = st.checkbox("Pause trading", value=paused)

        st.subheader("Timeframe & Symbols")
        tf_opts = ["M1","M5","M15","M30","H1","H4","D1"]
        tf_default = (ctrl.get("TF") or os.getenv("TF", "H1")).strip().upper()
        if tf_default not in tf_opts:
            tf_default = "H1"
        tf_sel = st.selectbox("Timeframe", options=tf_opts, index=tf_opts.index(tf_default))

        # Candidates from DB or env
        df_for_syms = load_signals()
        candidates: list[str] = []
        if df_for_syms is not None and not df_for_syms.empty and "symbol" in df_for_syms.columns:
            candidates = sorted(set([str(s) for s in df_for_syms["symbol"].dropna().unique()]))
        if not candidates:
            env_syms = os.getenv("SYMBOLS", "EURUSD,GBPUSD").replace(";", ",")
            candidates = [s.strip().upper() for s in env_syms.split(",") if s.strip()]
        # Current selection
        cur_sel_raw = ctrl.get("SYMBOLS") or os.getenv("SYMBOLS", ",".join(candidates))
        cur_sel = [s.strip().upper() for s in cur_sel_raw.replace(";", ",").split(",") if s.strip()]
        sel = st.multiselect("Symbols", options=candidates, default=[s for s in cur_sel if s in candidates])
        extra = st.text_input("Extra symbols (comma)", value="")

        st.subheader("News gate")
        news_enabled = st.checkbox("Enable news gate", value=str(ctrl.get("NEWS_ENABLED", "1")).strip().lower() in {"1","true","yes","on"})
        lookahead = st.number_input("Lookahead (min)", min_value=0, max_value=1440, value=int(ctrl.get("NEWS_LOOKAHEAD_MIN", "120") or 120))
        block_impacts_raw = ctrl.get("NEWS_BLOCK_IMPACTS", "extreme,high")
        block_impacts = st.text_input("Block impacts (comma)", value=block_impacts_raw)
        skip_upcoming = st.checkbox("Skip if upcoming", value=str(ctrl.get("NEWS_SKIP_IF_UPCOMING", "0")).strip().lower() in {"1","true","yes","on"})

        st.subheader("Regime gate")
        gate_on = st.checkbox("Enable regime gate", value=str(ctrl.get("GATE_BY_REGIME", "0")).strip().lower() in {"1","true","yes","on"})
        gate_src = st.text_input("Gate source column", value=ctrl.get("GATE_SOURCE", "market_regime"))
        gate_strict = st.checkbox("Strict (>0/<0)", value=str(ctrl.get("GATE_STRICT", "1")).strip().lower() in {"1","true","yes","on"})

        submitted = st.form_submit_button("Save controls")
        if submitted:
            all_syms = sel[:]
            if extra.strip():
                all_syms.extend([s.strip().upper() for s in extra.replace(";", ",").split(",") if s.strip()])
            ctrl_set({
                "PAUSE_TRADING": "1" if paused_new else "0",
                "TF": tf_sel,
                "SYMBOLS": ",".join(sorted(set(all_syms))) if all_syms else ",".join(candidates),
                "NEWS_ENABLED": "1" if news_enabled else "0",
                "NEWS_LOOKAHEAD_MIN": str(int(lookahead)),
                "NEWS_BLOCK_IMPACTS": block_impacts,
                "NEWS_SKIP_IF_UPCOMING": "1" if skip_upcoming else "0",
                "GATE_BY_REGIME": "1" if gate_on else "0",
                "GATE_SOURCE": gate_src,
                "GATE_STRICT": "1" if gate_strict else "0",
            })
            st.success("Controls saved. Trader will pick them up on next cycle.")
    df = load_signals()
    if df is None or df.empty:
        st.warning("No signals yet — waiting for trader to write new rows.")
        # still render logs
    else:
        mode = st.radio("View", ["Latest per symbol", "Recent per symbol"], horizontal=True)
        if mode.startswith("Latest"):
            render_latest(df)
        else:
            n = st.slider("Rows per symbol", min_value=2, max_value=10, value=3)
            render_recent(df, n=n)

    if df is not None and not df.empty:
        with st.expander("Full table"):
            st.dataframe(df, use_container_width=True)
        render_dist(df)

    # Status + model availability
    st.write("---")
    st.subheader("Status")
    eff_tf = (ctrl.get("TF") or os.getenv("TF", "H1")).strip().upper()
    eff_syms = [s.strip().upper() for s in (ctrl.get("SYMBOLS") or os.getenv("SYMBOLS", "")).replace(";", ",").split(",") if s.strip()]
    st.write(f"Timeframe: {eff_tf}")
    st.write(f"Symbols: {', '.join(eff_syms) if eff_syms else '(none)'}")

    if eff_syms:
        st.subheader("Model availability")
        rows = []
        model_folder = Path(f"models/{eff_tf.lower()}_models")
        for s in eff_syms:
            p = model_folder / f"{s}_{eff_tf}_best_model.pkl"
            rows.append({"symbol": s, "model_path": str(p), "exists": p.exists()})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Logs viewer
    st.write("---")
    st.subheader("Trader logs (tail)")
    tail_lines = st.slider("Lines", min_value=50, max_value=2000, value=400, step=50)
    if LOG_PATH.exists():
        try:
            with LOG_PATH.open("r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()[-tail_lines:]
            st.code("".join(lines), language="bash")
        except Exception as e:
            st.error(f"Log read error: {e}")
    else:
        st.info("Log file not found yet. It will appear after the trader writes the first lines.")


if __name__ == "__main__":
    main()
