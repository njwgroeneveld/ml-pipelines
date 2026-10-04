"""Every feature in FEATURES plus the label, computed from the raw snapshot.

Adding a feature = one function + one entry in FEATURES + a set in FEATURE_SETS.
Every feature is relative (percent, ratio or 0-100), never in price points:
NQ and XYZ100 trade at different levels, and trees cannot extrapolate.

Features are never stored: each step that needs them computes them from the raw snapshot
(about a second). A stored copy would go stale as soon as a feature changes, and Supabase
Storage has no lifecycle rules to clean such copies up.
"""
import numpy as np
import pandas as pd

from dataset import Dataset, raw_path
from mlkit import io
from mlkit.labels import forward_direction, forward_return


def rsi_14(df: pd.DataFrame) -> pd.Series:
    delta = df["c"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def rel_volume(df: pd.DataFrame) -> pd.Series:
    return df["v"] / df["v"].rolling(24).mean()


def atr_pct(df: pd.DataFrame) -> pd.Series:
    prev = df["c"].shift()
    tr = pd.concat([df["h"] - df["l"], (df["h"] - prev).abs(), (df["l"] - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / 14, adjust=False).mean() / df["c"]


def macd_hist_pct(df: pd.DataFrame) -> pd.Series:
    c = df["c"]
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    return (macd - macd.ewm(span=9, adjust=False).mean()) / c


def dist_ema10(df: pd.DataFrame) -> pd.Series:
    ema = df["c"].ewm(span=10, adjust=False).mean()
    return (df["c"] - ema) / ema


FEATURES = {
    "rsi_14": rsi_14,
    "rel_volume": rel_volume,
    "atr_pct": atr_pct,
    "macd_hist_pct": macd_hist_pct,
    "dist_ema10": dist_ema10,
}

FEATURE_SETS = {
    "base": ["rsi_14", "rel_volume", "atr_pct", "macd_hist_pct"],
    "base+ema10": ["rsi_14", "rel_volume", "atr_pct", "macd_hist_pct", "dist_ema10"],
}


def build_features(raw: pd.DataFrame, horizon: int, weekdays_only: bool) -> pd.DataFrame:
    df = raw.sort_values("t").reset_index(drop=True)
    # Weekend out BEFORE the features: Monday's candle then sees the jump since
    # Friday, exactly as it does on the future.
    if weekdays_only:
        df = df[df["t"].dt.dayofweek < 5].reset_index(drop=True)
    for name, fn in FEATURES.items():
        df[name] = fn(df)
    df["label"] = forward_direction(df["c"], horizon)
    df["fwd_ret"] = forward_return(df["c"], horizon)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[*FEATURES, "label"])
    return df[["t", "c", *FEATURES, "label", "fwd_ret"]].reset_index(drop=True)


def features_by_source(ds: Dataset, root: str) -> dict[str, pd.DataFrame]:
    """Read each source's raw snapshot and compute its features."""
    return {
        key: build_features(io.read_parquet(raw_path(root, ds, key)), ds.horizon, ds.weekdays_only)
        for key in ds.sources
    }
