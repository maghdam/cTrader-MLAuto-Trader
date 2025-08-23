# cTrader ML Auto-Trader (Double-Barrier, Docker, VectorBT, Optuna)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Dockerized](https://img.shields.io/badge/docker-ready-2496ED.svg)

> End-to-end ML pipeline on **cTrader OpenAPI** with **double-barrier labels**.
> Data → Features/Labels → Model Selection → **VectorBT** Backtest → **Optuna** Tuning → **Dockerized** Live Trading.
> Production-minded: env-driven config, idempotent execution, healthchecks, and SQLite audit trail.

---

## ✨ Highlights

* **Complete pipeline**: Data → FE/Labels → Model Select → VectorBT Backtest → Optuna Tuning → Live.
* **Double-Barrier labeling** (0/1/2) → mapped to trade signals **−1/0/+1**.
* **Per-symbol pipelines** persisted as `.pkl` (model pipeline + feature list).
* **Idempotent live logic**: no duplicate entries; clean flips; optional close on flat.
* **Dockerized** with health check, logs, and **SQLite** for live signal auditing.
* Works on **netting & hedging** (uses ClosePosition RPC).
* **News Gate (per symbol):** each iteration checks the **economic calendar** relevant to that symbol and can **block trading** around **high/critical** events (configurable).
* **NEW – Notion journaling:** all trade actions (opens/closes/blocks) are **logged to a Notion database** automatically.

---

## 📂 Repository Layout

```
.
├─ src/
│  ├─ __init__.py
│  ├─ bt_core.py                 # Backtest + time-series CV helpers
│  ├─ ctrader_client.py          # cTrader OpenAPI client (auth, symbols, orders, close)
│  ├─ feature_engineering.py     # Core indicators & FE
│  ├─ labeling_schemes.py        # Double-barrier & other labelers
│  ├─ news_guard.py              # ← NEW: per-symbol calendar gate (no API keys)
│  ├─ notion_journal.py          # ← NEW: Notion logging (trades & news events)
│  └─ live_trader.py             # Live runner (idempotent execution + news gate + Notion)
├─ models/
│  └─ h1_models/                 # Saved pipelines (e.g., EURUSD_H1_best_model.pkl)
├─ notebooks/
│  ├─ backtest_training_tuning.ipynb
│  └─ live_trader.ipynb
├─ reports/
│  ├─ backtest_summary.csv
│  ├─ model_summary.csv
│  └─ model_summary.xlsx
├─ logs/
│  └─ live_trader.log
├─ live_signals.db               # SQLite (live predictions audit)
├─ docker-compose.yml
├─ Dockerfile
├─ requirements.txt
├─ .env                          # credentials & runtime config (not committed)
└─ README.md
```

---

## 🔐 Prerequisites

* A cTrader **OpenAPI** application (Client ID/Secret) and **Access Token** with your **Account ID** on the same host (`demo`/`live`).
* Docker & Docker Compose (or Python 3.11 locally).

---

## 🚀 Quick Start (Docker)

1. Create `.env` in project root:

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
TZ=Europe/Zurich

# --- sizing ---
# Either DEFAULT_LOTS or LOTS_JSON (preferred)
# DEFAULT_LOTS=0.10
LOTS_JSON={"EURUSD":0.10,"GBPUSD":0.10,"AUDUSD":0.10,"DEFAULT":0.10}

# Optional protective distances (blank disables)
SL_PIPS=
TP_PIPS=

# --- NEW: News Gate (per-symbol calendar) ---
NEWS_ENABLED=1
NEWS_LOOKAHEAD_MIN=180
NEWS_BLOCK_IMPACTS=extreme,high
NEWS_SKIP_IF_UPCOMING=1
# Advanced (optional):
# FF_CACHE_TTL_SEC=1800
# AFFECTS_JSON={"XAUUSD":["USD"],"DE40":["EUR"]}   # override symbol→currencies
# WINDOWS_JSON={"extreme":[-90,45],"high":[-60,30],"medium":[-30,15]}

# --- NEW: Notion journaling ---
NOTION_ENABLED=1
NOTION_SECRET=secret_xxx
NOTION_DB_ID=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

2. Pre-create the SQLite file so the bind mount is a file (not a directory):

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
   * **News Gate (NEW):** checks the **economic calendar** for the symbol’s currencies and:

     * **Blocks** trading if currently inside a high/critical news window.
     * Optionally **skips** new entries if important news is upcoming (within `NEWS_LOOKAHEAD_MIN`).
     * Logs concise `news=` summaries on each bar, e.g.
       `| news=in 45m [high] USD • US CPI (YoY)`
   * Trades the **first** signal only → `2→BUY (+1)`, `0→SELL (−1)`, `1→FLAT (0)`.
   * **Notion journaling (NEW):** logs OPEN/CLOSE/NEWS\_BLOCK/NEWS\_UPCOMING events to the configured Notion database.

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

| Variable                     | Description                                         |
| ---------------------------- | --------------------------------------------------- |
| `CTRADER_CLIENT_ID`          | cTrader app client ID                               |
| `CTRADER_CLIENT_SECRET`      | cTrader app secret                                  |
| `CTRADER_ACCESS_TOKEN`       | OAuth token                                         |
| `CTRADER_ACCOUNT_ID`         | Trading account                                     |
| `CTRADER_HOST_TYPE`          | `demo` or `live`                                    |
| `TF`                         | `M1`,`M5`,`M15`,`M30`,`H1`,`H4`,`D1`                |
| `SYMBOLS`                    | Comma-separated list (e.g., `EURUSD,GBPUSD`)        |
| `LOG_LEVEL`                  | `INFO` / `DEBUG`                                    |
| `TZ`                         | Container timezone (e.g., `Europe/Zurich`)          |
| `DEFAULT_LOTS` / `LOTS_JSON` | Sizing; prefer `LOTS_JSON` for per-symbol + default |
| `SL_PIPS` / `TP_PIPS`        | Integer pip distances (blank disables)              |

**News Gate**

| Variable                | Description                                                               |
| ----------------------- | ------------------------------------------------------------------------- |
| `NEWS_ENABLED`          | Enable the calendar gate (`1`/`0`)                                        |
| `NEWS_LOOKAHEAD_MIN`    | Minutes ahead to consider events (e.g., `180`)                            |
| `NEWS_BLOCK_IMPACTS`    | Impacts that **block now** (comma list; e.g., `extreme,high`)             |
| `NEWS_SKIP_IF_UPCOMING` | If `1`, **skip entries** when important news is **upcoming**              |
| `FF_CACHE_TTL_SEC`      | (Advanced) Calendar cache TTL in seconds (default `1800`)                 |
| `AFFECTS_JSON`          | (Advanced) Override symbol→currencies JSON                                |
| `WINDOWS_JSON`          | (Advanced) Impact windows JSON: e.g. `{"high":[-60,30]}` (mins pre/after) |

**NEW – Notion journaling**

| Variable         | Description                       |
| ---------------- | --------------------------------- |
| `NOTION_ENABLED` | Enable Notion logging (`1`/`0`)   |
| `NOTION_SECRET`  | Notion integration token          |
| `NOTION_DB_ID`   | Target database ID for journaling |

---

## 🩺 Ops & Files

* **Logs** → `logs/live_trader.log`

  * Per-bar line includes `| news=…`
  * Explicit `[NEWS BLOCK]` / `[NEWS UPCOMING]` log lines when relevant
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

* **Notion**: each OPEN/CLOSE and news block/upcoming event is appended to your DB.

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
* **Notion**:

  * `NOTION_ENABLED=1` but no entries? Verify `NOTION_SECRET` and `NOTION_DB_ID` and that the integration has access to the DB.
* **News Gate**:

  * Seeing no events? Increase `NEWS_LOOKAHEAD_MIN`, or lower `NEWS_BLOCK_IMPACTS` strictness, or check `logs` for `news=` lines.
  * Timezones: all event times are treated as **UTC** internally and rendered as `Europe/Zurich` in per-bar logs.

---

## 🔗 Related

Looking for a broader **MT5** research lab (ML/DL/time-series, multi-strategy)?
**AlphaFlow** → [https://github.com/maghdam/AlphaFlow-ML-DL-Trading-Bot](https://github.com/maghdam/AlphaFlow-ML-DL-Trading-Bot)

---

## ⚠️ Disclaimer

This project is for educational/research purposes. Trading involves risk. Use at your own discretion and comply with broker terms and local regulations.


---

## 📝 License

MIT

