import os
import sys
import json
from pathlib import Path
import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

from ctrader_client import ensure_client_ready, get_ohlc_df
from feature_engineering import add_core_features, get_core_feature_cols
from labeling_schemes import (
    create_labels_double_barrier,
    create_labels_double_barrier_atr,
    create_labels_multi_bar,
)
from bt_core import quick_backtest_from_model


LOG = logging.getLogger("cli")
logging.basicConfig(level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
                    format="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def _symbols_from_env_or_arg(arg_symbols: str | None) -> list[str]:
    if arg_symbols:
        return [s.strip().upper() for s in arg_symbols.replace(";", ",").split(",") if s.strip()]
    env = os.getenv("SYMBOLS", "EURUSD")
    return [s.strip().upper() for s in env.replace(";", ",").split(",") if s.strip()]


def _ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def _feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df_feat = add_core_features(df.copy())
    cols = [c for c in get_core_feature_cols() if c in df_feat.columns]
    X = df_feat[cols].dropna()
    return X


def _label_frame(df: pd.DataFrame, strategy: str, **kwargs) -> tuple[pd.DataFrame, str]:
    if strategy == "double_barrier":
        labeled = create_labels_double_barrier(df, up=kwargs.get("up", 0.005), down=kwargs.get("down", 0.005), horizon=int(kwargs.get("horizon", 20)))
        return labeled, "barrier_label"
    if strategy == "double_barrier_atr":
        labeled = create_labels_double_barrier_atr(
            df,
            up_k=float(kwargs.get("up_k", 1.0)),
            down_k=float(kwargs.get("down_k", 1.0)),
            atr_window=int(kwargs.get("atr_window", 14)),
            horizon=int(kwargs.get("horizon", 20)),
        )
        return labeled, "barrier_label"
    if strategy == "multi_bar":
        labeled = create_labels_multi_bar(df, horizon=int(kwargs.get("horizon", 5)), threshold=float(kwargs.get("threshold", 0.005)))
        return labeled, "multi_bar_label"
    raise ValueError(f"Unknown strategy '{strategy}'")


def cmd_train(args: argparse.Namespace) -> int:
    ensure_client_ready(timeout=30)

    symbols = _symbols_from_env_or_arg(args.symbols)
    tf = args.tf or os.getenv("TF", "H1").upper()
    n_bars = int(args.n_bars or os.getenv("N_BARS", "2500"))

    model_dir = Path(f"models/{tf.lower()}_models")
    reports_dir = Path("reports")
    _ensure_dirs(model_dir, reports_dir)

    for sym in symbols:
        LOG.info("[TRAIN] %s TF=%s bars=%s strategy=%s", sym, tf, n_bars, args.strategy)
        df = get_ohlc_df(sym, tf=tf, n=n_bars)
        if df.empty:
            LOG.warning("No data for %s; skipping", sym)
            continue

        # Build features
        X_all = _feature_matrix(df)

        # Build labels and align
        labeled, y_col = _label_frame(df, args.strategy, **{
            "up": args.up, "down": args.down, "horizon": args.horizon,
            "up_k": args.up_k, "down_k": args.down_k, "atr_window": args.atr_window,
            "threshold": args.threshold,
        })
        labeled = labeled.loc[X_all.index]
        labeled = labeled.dropna(subset=[y_col])
        X = X_all.loc[labeled.index]
        y = labeled[y_col].astype(int)

        if X.empty or y.empty:
            LOG.warning("%s: empty X/y after alignment; skipping", sym)
            continue

        # Simple chronological split: last 20% as validation
        split = int(len(X) * 0.8)
        X_tr, X_va = X.iloc[:split], X.iloc[split:]
        y_tr, y_va = y.iloc[:split], y.iloc[split:]

        clf = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_leaf=2, random_state=42, n_jobs=-1)
        clf.fit(X_tr, y_tr)
        acc = accuracy_score(y_va, clf.predict(X_va)) if len(X_va) else float("nan")
        LOG.info("[TRAIN] %s validation accuracy=%.4f", sym, acc)

        bundle = {"pipeline": clf, "features": list(X.columns)}
        out_path = model_dir / f"{sym}_{tf}_best_model.pkl"
        joblib.dump(bundle, out_path)
        LOG.info("[TRAIN] Saved %s", out_path)

    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    ensure_client_ready(timeout=30)
    symbols = _symbols_from_env_or_arg(args.symbols)
    tf = args.tf or os.getenv("TF", "H1").upper()
    n_bars = int(args.n_bars or os.getenv("N_BARS", "2500"))
    threshold = float(args.threshold or 0.0)

    model_dir = Path(f"models/{tf.lower()}_models")
    reports_dir = Path("reports")
    _ensure_dirs(reports_dir)
    rows = []

    for sym in symbols:
        model_path = model_dir / f"{sym}_{tf}_best_model.pkl"
        if not model_path.exists():
            LOG.warning("[BT] Missing model for %s at %s", sym, model_path)
            continue

        bundle = joblib.load(model_path)
        features = bundle.get("features")
        model = bundle.get("pipeline") or bundle.get("model")
        if model is None:
            LOG.warning("[BT] Invalid bundle for %s", sym)
            continue

        df = get_ohlc_df(sym, tf=tf, n=n_bars)
        if df.empty:
            LOG.warning("[BT] No data for %s", sym)
            continue
        X_all = _feature_matrix(df)
        use_cols = [c for c in (features or list(X_all.columns)) if c in X_all.columns]
        if not use_cols:
            LOG.warning("[BT] %s: model feature list not found in current FE output", sym)
            continue
        X = X_all[use_cols].dropna()
        close = df.loc[X.index, "close"]

        acc, sr, stats = quick_backtest_from_model(model, X, y_shifted=None, close=close, threshold=threshold, fees=float(os.getenv("FEES", "0.0002")))
        LOG.info("[BT] %s Sharpe=%.3f TotalRet=%.2f%%", sym, sr, stats.get("total_return", 0.0) * 100)
        rows.append(dict(Symbol=sym, TF=tf, Sharpe=sr, TotalReturnPct=stats.get("total_return", 0.0) * 100))

    if rows:
        out = pd.DataFrame(rows)
        out_path = Path("reports") / f"backtest_summary_{tf}.csv"
        if out_path.exists():
            prev = pd.read_csv(out_path)
            out = pd.concat([prev, out], ignore_index=True)
        out.to_csv(out_path, index=False)
        LOG.info("[BT] Wrote %s", out_path)
    return 0


def cmd_tune(args: argparse.Namespace) -> int:
    import optuna

    ensure_client_ready(timeout=30)
    symbols = _symbols_from_env_or_arg(args.symbols)
    tf = args.tf or os.getenv("TF", "H1").upper()
    n_bars = int(args.n_bars or os.getenv("N_BARS", "2500"))
    n_trials = int(args.n_trials or os.getenv("N_TRIALS", "20"))

    model_dir = Path(f"models/{tf.lower()}_models")
    _ensure_dirs(model_dir)

    for sym in symbols:
        LOG.info("[TUNE] %s TF=%s bars=%s strategy=%s trials=%s", sym, tf, n_bars, args.strategy, n_trials)
        df = get_ohlc_df(sym, tf=tf, n=n_bars)
        if df.empty:
            LOG.warning("[TUNE] No data for %s; skipping", sym)
            continue
        X_all = _feature_matrix(df)
        labeled, y_col = _label_frame(df, args.strategy, **{
            "up": args.up, "down": args.down, "horizon": args.horizon,
            "up_k": args.up_k, "down_k": args.down_k, "atr_window": args.atr_window,
            "threshold": args.threshold,
        })
        labeled = labeled.loc[X_all.index]
        labeled = labeled.dropna(subset=[y_col])
        X = X_all.loc[labeled.index]
        y = labeled[y_col].astype(int)
        if X.empty or y.empty:
            LOG.warning("[TUNE] %s: empty X/y after alignment; skipping", sym)
            continue

        def objective(trial: optuna.Trial) -> float:
            n_estimators = trial.suggest_int("n_estimators", 100, 600)
            max_depth = trial.suggest_int("max_depth", 3, 20)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 6)

            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
                n_jobs=-1,
            )

            tscv = TimeSeriesSplit(n_splits=3)
            accs: list[float] = []
            for tr, te in tscv.split(X):
                X_tr, X_te = X.iloc[tr], X.iloc[te]
                y_tr, y_te = y.iloc[tr], y.iloc[te]
                clf.fit(X_tr, y_tr)
                preds = clf.predict(X_te)
                accs.append(accuracy_score(y_te, preds))
            return float(np.mean(accs)) if accs else 0.0

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        LOG.info("[TUNE] %s best params: %s best value=%.4f", sym, study.best_params, study.best_value)

        # Retrain on all data with best params and save
        best = study.best_params
        best_clf = RandomForestClassifier(
            n_estimators=best.get("n_estimators", 300),
            max_depth=best.get("max_depth", None),
            min_samples_leaf=best.get("min_samples_leaf", 1),
            random_state=42,
            n_jobs=-1,
        )
        best_clf.fit(X, y)
        bundle = {"pipeline": best_clf, "features": list(X.columns)}
        out_path = model_dir / f"{sym}_{tf}_best_model.pkl"
        joblib.dump(bundle, out_path)
        LOG.info("[TUNE] Saved %s", out_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="cTrader ML bot CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Common args template
    def add_common(pp):
        pp.add_argument("--symbols", help="Comma/semicolon-separated list (defaults to env SYMBOLS)")
        pp.add_argument("--tf", help="Timeframe (defaults to env TF, e.g., H1)")
        pp.add_argument("--n-bars", dest="n_bars", type=int, help="Bars to fetch (defaults to env N_BARS)")
        pp.add_argument("--strategy", choices=["double_barrier", "double_barrier_atr", "multi_bar"], default="double_barrier")
        pp.add_argument("--horizon", type=int, help="Horizon for labeling")
        pp.add_argument("--up", type=float, help="Upper pct for double_barrier (e.g., 0.005)")
        pp.add_argument("--down", type=float, help="Lower pct for double_barrier (e.g., 0.005)")
        pp.add_argument("--up-k", dest="up_k", type=float, help="Upper ATR k for double_barrier_atr")
        pp.add_argument("--down-k", dest="down_k", type=float, help="Lower ATR k for double_barrier_atr")
        pp.add_argument("--atr-window", dest="atr_window", type=int, help="ATR window for double_barrier_atr")
        pp.add_argument("--threshold", type=float, help="Threshold for multi_bar classification")

    pt = sub.add_parser("train", help="Train and save model(s)")
    add_common(pt)
    pt.set_defaults(func=cmd_train)

    pb = sub.add_parser("backtest", help="Backtest saved model(s) quickly")
    add_common(pb)
    pb.add_argument("--threshold", type=float, help="Decision threshold for continuous scores (unused for 0/1/2)")
    pb.set_defaults(func=cmd_backtest)

    po = sub.add_parser("tune", help="Optuna tune model(s)")
    add_common(po)
    po.add_argument("--n-trials", dest="n_trials", type=int, help="Optuna trials per symbol (default 20)")
    po.set_defaults(func=cmd_tune)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

