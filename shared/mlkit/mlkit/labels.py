"""The label every project uses: is the close higher `horizon` candles later?"""
import pandas as pd


def forward_return(close: pd.Series, horizon: int) -> pd.Series:
    """Return from this close to the close `horizon` rows later; NaN at the end."""
    return close.shift(-horizon) / close - 1


def forward_direction(close: pd.Series, horizon: int) -> pd.Series:
    """1.0 if the close `horizon` rows later is higher, 0.0 if not, NaN at the end."""
    fwd = forward_return(close, horizon)
    return (fwd > 0).astype(float).where(fwd.notna())
