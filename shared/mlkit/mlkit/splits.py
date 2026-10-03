"""Train / validation / test periods with a purge between them.

Periods are whole UTC days, both ends inclusive. A row's label looks `horizon`
rows ahead, so the last `horizon` rows of train and of validation are dropped:
their labels would otherwise peek into the next period.
"""
from dataclasses import dataclass

import pandas as pd


def _utc_day(value) -> pd.Timestamp:
    return pd.Timestamp(value).tz_localize("UTC")


@dataclass(frozen=True)
class Periods:
    train_end: pd.Timestamp
    val_from: pd.Timestamp
    val_to: pd.Timestamp
    test_from: pd.Timestamp
    test_to: pd.Timestamp

    @classmethod
    def from_dict(cls, d: dict) -> "Periods":
        return cls(
            train_end=_utc_day(d["train"]["end"]),
            val_from=_utc_day(d["val"]["from"]),
            val_to=_utc_day(d["val"]["to"]),
            test_from=_utc_day(d["test"]["from"]),
            test_to=_utc_day(d["test"]["to"]),
        )


def _days(df: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp) -> pd.DataFrame:
    mask = df["t"] < end + pd.Timedelta(days=1)
    if start is not None:
        mask &= df["t"] >= start
    return df[mask]


def split_periods(df: pd.DataFrame, periods: Periods, horizon: int) -> dict[str, pd.DataFrame]:
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    df = df.sort_values("t").reset_index(drop=True)
    return {
        "train": _days(df, None, periods.train_end).iloc[:-horizon],
        "val": _days(df, periods.val_from, periods.val_to).iloc[:-horizon],
        "test": _days(df, periods.test_from, periods.test_to),
    }


def common_timestamps(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only the candles whose timestamp is in both frames, row-aligned by time."""
    shared = pd.Index(a["t"]).intersection(pd.Index(b["t"]))
    a = a[a["t"].isin(shared)].sort_values("t").reset_index(drop=True)
    b = b[b["t"].isin(shared)].sort_values("t").reset_index(drop=True)
    return a, b
