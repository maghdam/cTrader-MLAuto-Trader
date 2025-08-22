import numpy as np
import pandas as pd


def create_labels_multi_bar(df: pd.DataFrame, horizon: int = 5, threshold: float = 0.005) -> pd.DataFrame:
    """
    Multi-bar (fixed-horizon) labeling in ML space **0/1/2**:
      0 = down, 1 = flat, 2 = up
    """
    out = df.copy()
    out["future_return_h"] = out["close"].pct_change(periods=horizon).shift(-horizon)
    lab = np.full(len(out), 1, dtype=int)  # default flat
    lab[out["future_return_h"] >= threshold] = 2
    lab[out["future_return_h"] <= -threshold] = 0
    out["multi_bar_label"] = lab
    return out.dropna(subset=["future_return_h"])


def create_labels_double_barrier(df: pd.DataFrame, up: float = 0.005, down: float = 0.005, horizon: int = 20) -> pd.DataFrame:
    """
    Double-barrier first-hit labeling in ML space **0/1/2**:
      0 = hit lower first (down), 1 = none (flat), 2 = hit upper first (up)
    """
    out = df.copy()
    c = out["close"].to_numpy()
    lab = np.full(len(out), 1, dtype=int)  # default flat
    for i in range(len(c)):
        upper, lower = c[i] * (1 + up), c[i] * (1 - down)
        end = min(i + horizon, len(c))
        cur = 1
        for j in range(i + 1, end):
            if c[j] >= upper:
                cur = 2
                break
            if c[j] <= lower:
                cur = 0
                break
        lab[i] = cur
    out["barrier_label"] = lab
    return out


def create_labels_regime_detection(df: pd.DataFrame, short_window=20, long_window=50) -> pd.DataFrame:
    out = df.copy()
    out["ma_short"] = out["close"].rolling(short_window).mean()
    out["ma_long"] = out["close"].rolling(long_window).mean()
    # Keep regime label as -1/0/1 for analytics, but NOT used for ML in this repo
    reg = np.zeros(len(out), dtype=int)
    reg[out["ma_short"] > out["ma_long"]] = 1
    reg[out["ma_short"] < out["ma_long"]] = -1
    out["regime_label"] = reg
    return out.dropna(subset=["ma_short", "ma_long"])
