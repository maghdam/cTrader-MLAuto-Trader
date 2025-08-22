# feature_engineering.py
import numpy as np
import pandas as pd
import math
import ta
from statsmodels.tsa.stattools import adfuller

# --------------------------------------------------------------------
# 1) TA-LIB FEATURES (ta lib)
# --------------------------------------------------------------------
def add_all_ta_features(df: pd.DataFrame, volume_col: str = "volume") -> pd.DataFrame:
    """Add TA features via `ta.add_all_ta_features`. Expects: open,high,low,close, {volume_col}."""
    vol = volume_col
    if vol not in df.columns:
        # fallback to typical cTrader naming
        vol = "tick_volume" if "tick_volume" in df.columns else volume_col
    return ta.add_all_ta_features(df.copy(), open="open", high="high", low="low", close="close",
                                  volume=vol, fillna=True)

# --------------------------------------------------------------------
# 2) MISC FEATURES
# --------------------------------------------------------------------
def spread(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["spread"] = out["high"] - out["low"]
    return out

def auto_corr_multi(df: pd.DataFrame, col: str, n: int = 50, lags=(1, 5, 10)) -> pd.DataFrame:
    out = df.copy()
    for lag in lags:
        out[f"autocorr_{lag}"] = (
            out[col].rolling(window=n, min_periods=n).apply(lambda x: x.autocorr(lag=lag), raw=False)
        )
    return out

def candle_information(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["candle_way"] = (out["close"] > out["open"]).astype(int)
    rng = (out["high"] - out["low"]).replace(0, np.nan)
    out["fill"] = (np.abs(out["close"] - out["open"]) / (rng + 1e-5))
    out["amplitude"] = np.abs(out["close"] - out["open"]) / (out["open"].replace(0, np.nan) + 1e-5)
    return out

# --------------------------------------------------------------------
# 3) VOLATILITY ESTIMATORS (vectorized via rolling.apply)
# --------------------------------------------------------------------
def _parkinson_estimator(window: pd.DataFrame) -> float:
    n = len(window)
    if n < 1: return np.nan
    sum_sq = np.sum(np.log(window["high"] / window["low"]) ** 2)
    return math.sqrt(sum_sq / (4 * math.log(2) * n))

def moving_parkinson_estimator(df: pd.DataFrame, window_size: int = 30) -> pd.DataFrame:
    out = df.copy()
    out["rolling_volatility_parkinson"] = (
        out[["high","low"]].rolling(window_size).apply(
            lambda w: _parkinson_estimator(pd.DataFrame(w.reshape(-1,2), columns=["high","low"])),
            raw=True
        )
    )
    return out

def _yang_zhang_estimator(window: pd.DataFrame) -> float:
    n = len(window)
    if n < 1: return np.nan
    term1 = np.log(window["high"] / window["low"]) ** 2
    term2 = np.log(window["close"] / window["open"]) ** 2
    return math.sqrt(np.mean(term1 + term2))

def moving_yang_zhang_estimator(df: pd.DataFrame, window_size: int = 30) -> pd.DataFrame:
    out = df.copy()
    out["rolling_volatility_yang_zhang"] = (
        out[["open","high","low","close"]].rolling(window_size).apply(
            lambda w: _yang_zhang_estimator(pd.DataFrame(w.reshape(-1,4), columns=["open","high","low","close"])),
            raw=True
        )
    )
    return out

# --------------------------------------------------------------------
# 4) MARKET REGIME
# --------------------------------------------------------------------
def kama_market_regime(df: pd.DataFrame, col: str = "close", n1: int = 10, n2: int = 30) -> pd.DataFrame:
    out = df.copy()
    short_kama = out[col].ewm(span=n1, adjust=False).mean()
    long_kama  = out[col].ewm(span=n2, adjust=False).mean()
    out["kama_diff"] = short_kama - long_kama
    out["kama_trend"] = (out["kama_diff"] >= 0).astype(int)
    return out

# --------------------------------------------------------------------
# 5) ROLLING ADF (stationarity)
# --------------------------------------------------------------------
def rolling_adf_with_flag(df: pd.DataFrame, col: str = "close", window_size: int = 50, p_value_threshold=0.05) -> pd.DataFrame:
    out = df.copy()
    adf_stat = pd.Series(np.nan, index=out.index, dtype="float64")
    adf_pval = pd.Series(np.nan, index=out.index, dtype="float64")
    flag     = pd.Series(np.nan, index=out.index, dtype="float64")

    for i in range(window_size, len(out)):
        series = out[col].iloc[i-window_size:i].values
        try:
            stat, pval = adfuller(series, autolag="AIC")[:2]
            adf_stat.iat[i] = stat
            adf_pval.iat[i] = pval
            flag.iat[i]     = 1.0 if pval < p_value_threshold else 0.0
        except Exception:
            pass

    out["rolling_adf_stat"] = adf_stat
    out["rolling_adf_pval"] = adf_pval
    out["stationary_flag"]  = flag
    return out

# --------------------------------------------------------------------
# 6) CORE FEATURE SET (used by both notebook & live)
# --------------------------------------------------------------------
def add_core_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Trend & Momentum
    out["sma_20"] = out["close"].rolling(20).mean()
    out["ema_20"] = out["close"].ewm(span=20, adjust=False).mean()
    out["kama_10"] = out["close"].ewm(span=10, adjust=False).mean()  # simple proxy
    rsi = ta.momentum.rsi(out["close"], window=14)
    macd = ta.trend.macd(out["close"])
    macd_signal = ta.trend.macd_signal(out["close"])
    out["rsi_14"] = rsi
    out["macd_diff"] = macd - macd_signal

    # Vol & Volume
    out["atr_14"] = ta.volatility.average_true_range(out["high"], out["low"], out["close"], window=14)
    vol_col = "volume" if "volume" in out.columns else ("tick_volume" if "tick_volume" in out.columns else None)
    out["obv"] = ta.volume.on_balance_volume(out["close"], out[vol_col]) if vol_col else np.nan
    out["rolling_std_20"] = out["close"].rolling(20).std()

    # Candle & structure
    out = spread(out)
    out = candle_information(out)

    # Autocorrelation
    out = auto_corr_multi(out, col="close", n=50, lags=(1, 5, 10))

    # Regime
    out = kama_market_regime(out, col="close", n1=10, n2=30)
    out["ma_short"] = out["close"].rolling(20).mean()
    out["ma_long"]  = out["close"].rolling(50).mean()
    out["market_regime"] = 0
    out.loc[out["ma_short"] > out["ma_long"], "market_regime"] = 1
    out.loc[out["ma_short"] < out["ma_long"], "market_regime"] = -1

    # Stationarity
    out = rolling_adf_with_flag(out, col="close", window_size=50)

    return out.dropna()

def get_core_feature_cols() -> list[str]:
    return [
        "sma_20","ema_20","kama_10","rsi_14","macd_diff",
        "atr_14","obv","rolling_std_20","spread","fill","amplitude",
        "autocorr_1","autocorr_5","autocorr_10","market_regime","stationary_flag"
    ]
