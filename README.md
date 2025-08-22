# cTrader ML Auto-Trader (Double-Barrier, Docker, VectorBT, Optuna)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Dockerized](https://img.shields.io/badge/docker-ready-2496ED.svg)

> **TL;DR (Hiring Managers)**  
> End-to-end ML pipeline on **cTrader OpenAPI** with **double-barrier labels**.  
> Data → Features/Labels → Model Selection → **VectorBT** Backtest → **Optuna** Tuning → **Dockerized** Live Trading.  
> Production-minded: env-driven config, idempotent execution, healthchecks, and SQLite audit trail.

---

## ✨ Highlights

- **Complete pipeline**: Data → FE/Labels → Model Select → VectorBT Backtest → Optuna Tuning → Live.
- **Double-Barrier labeling** (0/1/2) → mapped to trade signals **−1/0/+1**.
- **Per-symbol pipelines** persisted as `.pkl` (model pipeline + feature list).
- **Idempotent live logic**: no duplicate entries; clean flips; optional close on flat.
- **Dockerized** with health check, logs, and **SQLite** for live signal auditing.
- Works on **netting & hedging** (uses ClosePosition RPC).

---

## 📂 Repository Layout

```

.
├─ src/
│  ├─ **init**.py
│  ├─ bt\_core.py                # Backtest + time-series CV helpers
│  ├─ ctrader\_client.py         # cTrader OpenAPI client (auth, symbols, orders, close)
│  ├─ feature\_engineering.py    # Core indicators & FE
│  ├─ labeling\_schemes.py       # Double-barrier & other labelers
│  └─ live\_trader.py            # Live runner (idempotent execution)
├─ models/
│  └─ h1\_models/                # Saved pipelines (e.g., EURUSD\_H1\_best\_model.pkl)
├─ notebooks/
│  ├─ backtest\_training\_tuning.ipynb
│  └─ live\_trader.ipynb
├─ reports/
│  ├─ backtest\_summary.csv
│  ├─ model\_summary.csv
│  └─ model\_summary.xlsx
├─ logs/
│  └─ live\_trader.log
├─ live\_signals.db              # SQLite (live predictions audit)
├─ docker-compose.yml
├─ Dockerfile
├─ requirements.txt
├─ .env                         # credentials & runtime config (not committed)
└─ README.md

````

---

## 🔐 Prerequisites

- A cTrader **OpenAPI** application (Client ID/Secret) and **Access Token** with your **Account ID** on the same host (`demo`/`live`).
- Docker & Docker Compose (or Python 3.11 locally).

---

## 🚀 Quick Start (Docker)

1) Create `.env` in project root:

```env
# --- cTrader credentials ---
CTRADER_CLIENT_ID=xxxxxxxx
CTRADER_CLIENT_SECRET=xxxxxxxx
CTRADER_ACCESS_TOKEN=xxxxxxxx
CTRADER_ACCOUNT_ID=12345678
CTRADER_HOST_TYPE=demo   # or live

# --- runtime ---
TF=H1
SYMBOLS=EURUSD,GBPUSD,AUDUSD
LOG_LEVEL=INFO
TZ=Europe/Zurich          # container timezone (optional but recommended)
````

2. Precreate the SQLite file so the bind mount is a file (not a directory):

```bash
touch live_signals.db
```

3. Build & run:

```bash
docker compose up --build -d trader
```

4. Follow logs:

```bash
docker logs -f dbarrier-trader
```

Stop:

```bash
docker compose down
```

---

## 🤖 Live Trading — How It Works

1. Bootstraps cTrader, validates token ↔ account, resolves symbols.
2. Loads `{SYMBOL}_{TF}_best_model.pkl` from `models/<tf>_models/`.
3. On each **new bar**:

   * Fetches OHLCV → recomputes features → predicts **N\_FORWARD** labels `{0,1,2}`.
   * Saves future steps to **SQLite** (`live_signals.db`) with uniqueness guard.
   * Trades the **first** signal only → `2→BUY (+1)`, `0→SELL (−1)`, `1→FLAT (0)`.

**Position policy (idempotent)**

* **+1 (LONG)**: close SELLs → wait flat → open one BUY.
* **−1 (SHORT)**: close BUYs → wait flat → open one SELL.
* **0 (FLAT)**: if `CLOSE_ON_FLAT=True`, close any open; otherwise hold.
* No re-entries on same side when `ALLOW_PYRAMIDING=False`.

**Netting vs Hedging**

* Uses a dedicated **ClosePosition** RPC → true closures on both account types.

---

## 🧪 Model Selection & Tuning (Optuna)

Run a one-shot optimization; best per-symbol pipelines are saved under `models/<tf>_models/`:

```bash
docker compose --profile ops run --rm optimize
```

Outputs include:

* `EURUSD_H1_best_model.pkl` (etc.)
* `summary_optuna_H1.csv` (baseline vs tuned)

Then reload the trader:

```bash
docker compose restart trader
```

---

## 📊 Backtesting (VectorBT)

Evaluate saved models on fresh data:

```bash
docker compose run --rm trader python src/backtest_vectorbt.py
```

This builds a `vbt.Portfolio` (long/short), prints summary stats, and can render charts locally.

---

## ⚙️ Configuration (Env Vars)

| Variable                | Description                                  |
| ----------------------- | -------------------------------------------- |
| `CTRADER_CLIENT_ID`     | cTrader app client ID                        |
| `CTRADER_CLIENT_SECRET` | cTrader app secret                           |
| `CTRADER_ACCESS_TOKEN`  | OAuth token                                  |
| `CTRADER_ACCOUNT_ID`    | Trading account                              |
| `CTRADER_HOST_TYPE`     | `demo` or `live`                             |
| `TF`                    | `M1`,`M5`,`M15`,`M30`,`H1`,`H4`,`D1`         |
| `SYMBOLS`               | Comma-separated list (e.g., `EURUSD,GBPUSD`) |
| `LOG_LEVEL`             | `INFO` / `DEBUG`                             |
| `TZ`                    | Container timezone (e.g., `Europe/Zurich`)   |

Sizing:

* `DEFAULT_LOTS` or `LOTS_JSON={"EURUSD":0.10,"GBPUSD":0.10,"AUDUSD":0.10,"DEFAULT":0.10}`
* Optional `SL_PIPS` / `TP_PIPS` (integers; blank disables).

---

## 🩺 Ops & Files

* **Logs** → `logs/live_trader.log`
* **Healthcheck** → container is `healthy` after “Live loop started” appears
* **SQLite audit** → predictions saved in `live_signals.db`:

```bash
docker exec -it dbarrier-trader python - <<'PY'
import sqlite3
con = sqlite3.connect('/app/live_signals.db')
cur = con.cursor()
cur.execute("SELECT COUNT(*) FROM signals")
print("rows:", cur.fetchone()[0])
cur.execute("SELECT symbol, prediction, timestamp FROM signals ORDER BY id DESC LIMIT 10")
for r in cur.fetchall(): print(r)
con.close()
PY
```

---

## 🛠️ Development Tips

* The code is bind-mounted (`./src:/app/src`).
  If you run with `entrypoint: ["python","-u","src/live_trader.py"]`, restart the container after code changes:

  ```bash
  docker compose restart trader
  ```
* Optional hot-reload: use a small wrapper (`watchfiles`) and set `DEV_RELOAD=1` in `docker-compose.yml`.

Local (no Docker):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. python src/live_trader.py
```

---

## 🧱 Labels, Models, Backtests

* **Labels**: Double-Barrier (first-hit): **0=down**, **1=flat**, **2=up**.
* **Signals**: map to **−1/0/+1** respectively; classifiers train/predict in `{0,1,2}`.
* To switch labelers, use alternatives in `labeling_schemes.py` consistently across training/backtest/tuning.

> Sharpe annualization guide: `D1≈252`, `H4≈6*252`, `H1≈24*252`, `M15≈96*252`.

---

## 🧯 Troubleshooting

* **Auth/403**: host type must match token; token must be authorized for `CTRADER_ACCOUNT_ID`.
* **Symbols missing**: check logs for cTrader error codes; ensure market data permissions.
* **No trades**: if signals are always `0`, revisit thresholds/features or confirm models exist in `models/<tf>_models/`.
* **SQLite mount**: create `live_signals.db` before starting Compose.

---

## 🔗 Related

Looking for a broader **MT5** research lab (ML/DL/time-series, multi-strategy)?
**AlphaFlow** → [https://github.com/maghdam/AlphaFlow-ML-DL-Trading-Bot](https://github.com/maghdam/AlphaFlow-ML-DL-Trading-Bot)

---

## ⚠️ Disclaimer

This project is for educational/research purposes. **Trading involves risk.**
Use at your own discretion and comply with broker terms and local regulations.

---

## 📝 License

MIT

