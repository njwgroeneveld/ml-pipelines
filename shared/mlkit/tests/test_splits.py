import pandas as pd
import pytest

from mlkit.splits import Periods, common_timestamps, split_periods

PERIODS = Periods.from_dict({
    "train": {"end": "2026-01-02"},
    "val": {"from": "2026-01-03", "to": "2026-01-03"},
    "test": {"from": "2026-01-04", "to": "2026-01-04"},
})


def hourly(start: str, end: str) -> pd.DataFrame:
    t = pd.date_range(start, end, freq="h", tz="UTC")
    return pd.DataFrame({"t": t, "c": range(len(t))})


def test_periods_are_whole_utc_days_inclusive():
    parts = split_periods(hourly("2026-01-01", "2026-01-04 23:00"), PERIODS, horizon=1)

    assert parts["test"]["t"].min() == pd.Timestamp("2026-01-04 00:00", tz="UTC")
    assert parts["test"]["t"].max() == pd.Timestamp("2026-01-04 23:00", tz="UTC")
    assert len(parts["test"]) == 24


def test_last_horizon_rows_of_train_and_val_are_purged():
    parts = split_periods(hourly("2026-01-01", "2026-01-04 23:00"), PERIODS, horizon=4)

    # train ends 2026-01-02 23:00; the last 4 hours are dropped
    assert parts["train"]["t"].max() == pd.Timestamp("2026-01-02 19:00", tz="UTC")
    assert parts["val"]["t"].max() == pd.Timestamp("2026-01-03 19:00", tz="UTC")
    assert len(parts["val"]) == 20
    assert len(parts["test"]) == 24  # test is not purged


def test_periods_never_overlap():
    parts = split_periods(hourly("2026-01-01", "2026-01-04 23:00"), PERIODS, horizon=4)
    assert parts["train"]["t"].max() < parts["val"]["t"].min()
    assert parts["val"]["t"].max() < parts["test"]["t"].min()


def test_horizon_zero_is_rejected():
    with pytest.raises(ValueError):
        split_periods(hourly("2026-01-01", "2026-01-02"), PERIODS, horizon=0)


def test_common_timestamps_keeps_shared_hours_aligned():
    a = hourly("2026-01-01 00:00", "2026-01-01 05:00")
    b = hourly("2026-01-01 03:00", "2026-01-01 08:00").sample(frac=1, random_state=0)  # shuffled

    a2, b2 = common_timestamps(a, b)

    assert a2["t"].tolist() == b2["t"].tolist()
    assert len(a2) == 3
