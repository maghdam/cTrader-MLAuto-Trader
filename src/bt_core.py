from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Callable, Dict, Any, Iterable, Tuple, Union, Optional

try:
    import vectorbt as vbt  # optional import; only needed for deep backtest
except Exception:
    vbt = None

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.base import clone

# ============== helpers ======================================================

def preds_to_labels012(preds: np.ndarray) -> np.ndarray:
    """Normalize predictions to integer class ids {0,1,2}."""
    arr = np.asarray(preds)
    if arr.ndim == 1:
        return arr.astype(int)
    if arr.ndim == 2:  # probabilities or scores per class
        return arr.argmax(axis=1).astype(int)
    raise ValueError("Unexpected prediction shape for multiclass output.")


def labels012_to_signals(labels012: np.ndarray) -> np.ndarray:
    """Map ML labels {0,1,2} -> trading signals {-1,0,+1}."""
    return np.asarray(labels012, dtype=int) - 1


def to_signals_from_scores(scores: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Convert model outputs to {-1,0,+1} signals.
    - If scores are class labels in {0,1,2}, we map → {-1,0,+1}.
    - If scores are continuous, we threshold around 0.0 (or user threshold).
    """
    scores = np.asarray(scores)

    # case 1: integer {0,1,2} labels
    if np.issubdtype(scores.dtype, np.integer) and set(np.unique(scores)).issubset({0, 1, 2}):
        return scores - 1

    # case 2: probabilities for class 2 (buy) vs class 0 (sell) optional
    # user can precompute "buy_prob - sell_prob" and pass it here

    # case 3: continuous scores
    sig = np.zeros_like(scores, dtype=int)
    sig[scores > threshold] = 1
    sig[scores < -threshold] = -1
    return sig


def simple_returns_from_signals(signals: Iterable[int], close: pd.Series, fees: float = 0.0002) -> np.ndarray:
    """Your original quick backtest core, vectorized."""
    signals = np.asarray(signals, dtype=int)
    if len(signals) != len(close):
        raise ValueError("signals and close length mismatch")

    price_ret = close.pct_change().fillna(0.0).to_numpy()
    # base pnl
    ret = signals * price_ret

    # transaction costs when position changes
    pos_change = np.zeros_like(signals, dtype=bool)
    pos_change[1:] = signals[1:] != signals[:-1]
    ret[pos_change] -= fees
    return ret


def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0, periods_per_year: Optional[int] = None) -> float:
    r = np.asarray(returns) - risk_free
    mu = r.mean()
    sd = r.std()
    if sd == 0:
        return float("nan")
    sr = float(mu / sd)
    return sr * np.sqrt(periods_per_year) if periods_per_year else sr


def model_and_transform(
    model_or_tuple: Union[Any, Tuple[Any, Callable[[Any], Any]]]
) -> Tuple[Any, Optional[Callable[[Any], Any]]]:
    """
    Accept either a Pipeline (no transform callable) or (model, transform) pair.
    """
    if isinstance(model_or_tuple, tuple) and len(model_or_tuple) == 2 and callable(model_or_tuple[1]):
        return model_or_tuple[0], model_or_tuple[1]
    # assume pipeline or model that can .predict on raw X
    return model_or_tuple, None


# ============== TIER 1: quick (no vectorbt) ================================

def quick_backtest_from_signals(
    signals: Iterable[int],
    close: pd.Series,
    fees: float = 0.0002,
    periods_per_year: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Fast, dependency-light backtest. Returns a few core metrics.
    """
    signals = np.asarray(signals, dtype=int)
    close = close.astype(float)

    rets = simple_returns_from_signals(signals, close, fees=fees)
    total_return = float((1 + rets).prod() - 1)
    sr = sharpe_ratio(rets, periods_per_year=periods_per_year)

    # naive win-rate (positive bar pnl)
    wins = (rets > 0).mean() if len(rets) else float("nan")

    return dict(
        total_return=total_return,   # decimal (0.12 = 12%)
        sharpe=sr,
        win_rate=wins,
        returns=rets,                # array, if caller wants more stats
    )


def quick_backtest_from_model(
    model_or_tuple: Union[Any, Tuple[Any, Callable[[Any], Any]]],
    X: pd.DataFrame,
    y_shifted: Optional[pd.Series],
    close: pd.Series,
    threshold: float = 0.0,
    fees: float = 0.0002,
    periods_per_year: Optional[int] = None,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Quick evaluation on a given slice: no CV, no vectorbt. Good for speed.
    Returns: (accuracy, sharpe, stats_dict).
    """
    model, transform = model_and_transform(model_or_tuple)
    X_in = transform(X) if transform else X

    preds = model.predict(X_in)

    # accuracy expects same scale as y_shifted (0/1/2). If y_shifted is None, skip accuracy.
    acc = float("nan")
    if y_shifted is not None:
        # ensure shapes align
        y_s = y_shifted.loc[X.index] if isinstance(y_shifted, pd.Series) else y_shifted
        acc = accuracy_score(y_s, preds_to_labels012(preds))

    signals = to_signals_from_scores(preds, threshold=threshold)
    close_aligned = close.loc[X.index]
    stats = quick_backtest_from_signals(signals, close_aligned, fees=fees, periods_per_year=periods_per_year)
    return acc, stats["sharpe"], stats


def cv_quick(
    models: dict,
    X: pd.DataFrame,
    y_shifted: pd.Series,
    close: pd.Series,
    n_splits: int = 5,
    threshold: float = 0.0,
    fees: float = 0.0002,
    periods_per_year: Optional[int] = None,
) -> list[dict]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    out = []

    y_all = y_shifted.loc[X.index]
    c_all = close.loc[X.index]

    for name, model_or_tuple in models.items():
        fold_acc, fold_sr = [], []

        for tr, te in tscv.split(X):
            X_tr, X_te = X.iloc[tr], X.iloc[te]
            y_tr, y_te = y_all.iloc[tr], y_all.iloc[te]
            c_te = c_all.iloc[te]

            base_model, transform = model_and_transform(model_or_tuple)

            # fresh estimator per fold to avoid class-state carryover
            model = clone(base_model)

            X_tr_in = transform(X_tr) if transform else X_tr
            X_te_in = transform(X_te) if transform else X_te

            # require at least 2 classes in the training fold
            uniq = np.unique(y_tr)
            if len(uniq) < 2:
                # not enough classes to fit a classifier – skip this fold
                continue

            try:
                model.fit(X_tr_in, y_tr)
            except ValueError as e:
                # e.g., "Invalid classes inferred..." – skip this fold
                print(f"⏭️  Skipping fold for {name}: {e}")
                continue

            preds = preds_to_labels012(model.predict(X_te_in))

            # y_te must be a 1-D array of integer classes
            y_vec = np.asarray(y_te).ravel().astype(int)

            acc = accuracy_score(y_vec, preds)

            signals = labels012_to_signals(preds)
            stats = quick_backtest_from_signals(signals, c_te, fees=fees, periods_per_year=periods_per_year)

            fold_acc.append(acc)
            fold_sr.append(stats["sharpe"])

        out.append(dict(
            Model=name,
            Accuracy=float(np.nanmean(fold_acc)) if len(fold_acc) else float("nan"),
            Sharpe=float(np.nanmean(fold_sr)) if len(fold_sr) else float("nan"),
        ))

    return out


# ============== TIER 2: vectorbt (rich) ====================================

def vbt_backtest_from_signals(
    close: pd.Series,
    signals: Iterable[int],
    *,
    init_cash: float = 10_000.0,
    freq: str = "1h",
    fees: float = 0.0002,
):
    """
    Rich backtest with vectorbt (supports long & short). Returns vbt.Portfolio.
    """
    if vbt is None:
        raise ImportError("vectorbt is not installed")
    sig = pd.Series(np.asarray(signals, dtype=int), index=close.index)
    close, sig = close.align(sig, join="inner")

    long_entries  = (sig.shift(1).fillna(0) <= 0) & (sig > 0)
    long_exits    = sig <= 0
    short_entries = (sig.shift(1).fillna(0) >= 0) & (sig < 0)
    short_exits   = sig >= 0

    pf = vbt.Portfolio.from_signals(
        close,
        entries=long_entries,
        exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        init_cash=init_cash,
        freq=freq,
        fees=fees,
    )
    return pf


def vbt_backtest_from_model(
    model_or_tuple: Union[Any, Tuple[Any, Callable[[Any], Any]]],
    X: pd.DataFrame,
    close: pd.Series,
    *,
    init_cash: float = 10_000.0,
    freq: str = "1h",
    threshold: float = 0.0,
    fees: float = 0.0002,
):
    """
    Rich backtest on a fitted model (or pipeline) over X, aligned to close.
    """
    if vbt is None:
        raise ImportError("vectorbt is not installed")
    model, transform = model_and_transform(model_or_tuple)
    X_in = transform(X) if transform else X
    preds = preds_to_labels012(model.predict(X_in))
    signals = labels012_to_signals(preds)
    c = close.loc[X.index]
    return vbt_backtest_from_signals(c, signals, init_cash=init_cash, freq=freq, fees=fees)
