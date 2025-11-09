Usage Guide
===========

CLI (train/backtest/tune)
-------------------------

Reproducible, scriptable flows that call the same `src/` code used by notebooks.

Examples:

```
# Train default (double-barrier) on env SYMBOLS/TF/N_BARS
python -m src.cli train

# Train ATR-scaled double-barrier on specific symbols
python -m src.cli train --symbols EURUSD,GBPUSD --tf H1 --strategy double_barrier_atr --horizon 20 --up-k 1.5 --down-k 1.0 --atr-window 14

# Backtest saved models quickly
python -m src.cli backtest --symbols EURUSD,GBPUSD --tf H1

# Tune with Optuna
python -m src.cli tune --symbols EURUSD --tf H1 --n-trials 30
```

The compose `optimize` service runs `src/optimize_models.py`, which delegates to `src.cli` tune.


Dashboard (optional)
--------------------

A minimal Streamlit dashboard is included to visualize `live_signals.db`.

Start alongside the trader:

```
docker compose up -d dashboard
```

Open http://localhost:8501 to see latest/recent predictions per symbol and distributions.


Regime/Momentum Gate (optional)
-------------------------------

You can gate live entries by a simple regime signal from the feature set (e.g., `market_regime` or `kama_trend`).
See `docs/REGIME_GATE.md` for details.

```
GATE_BY_REGIME=1
GATE_SOURCE=market_regime   # or kama_trend
GATE_STRICT=1               # 1: >0/<0; 0: >=0/<=0
```


Resilience Settings (Reconnect & Retries)
-----------------------------------------

Tune reconnects and request retries for robust unattended runs:

```
# Reconnect behavior on TCP disconnects
RECONNECT_ENABLED=1
RECONNECT_BASE_SEC=5
RECONNECT_MAX_SEC=60
RECONNECT_JITTER_SEC=1.5

# Request retries (e.g., trendbars fetch)
REQ_RETRY_ATTEMPTS=3
REQ_RETRY_BACKOFF=0.75   # seconds; exponential per attempt

# Log rotation (live_trader)
LOG_MAX_BYTES=5242880    # 5 MB
LOG_BACKUP_COUNT=3
```

