import pandas as pd
import pytest

import fetch
from dataset import raw_path
from mlkit import io
from mlkit.errors import DataCheckError


def candles(times: list[str], volumes: list[float]) -> pd.DataFrame:
    t = pd.to_datetime(times, utc=True)
    return pd.DataFrame({"t": t, "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": volumes})


def test_clean_drops_unaligned_incomplete_zero_volume_and_duplicates():
    df = candles(
        ["2026-01-05 10:00", "2026-01-05 10:30", "2026-01-05 11:00", "2026-01-05 11:00",
         "2026-01-05 12:00", "2026-01-05 13:00"],
        [5, 5, 0, 0, 5, 5],
    )
    # 13:00 candle is still running at 13:20
    now = pd.Timestamp("2026-01-05 13:20", tz="UTC")

    clean, dropped = fetch.clean_candles(df, now, "1h")

    assert clean["t"].dt.strftime("%H:%M").tolist() == ["10:00", "12:00"]
    assert dropped == {"not_aligned": 1, "incomplete": 1, "zero_volume": 2, "duplicate": 0}


def test_clean_sorts_and_removes_duplicates():
    df = candles(["2026-01-05 12:00", "2026-01-05 10:00", "2026-01-05 10:00"], [5, 5, 5])
    clean, dropped = fetch.clean_candles(df, pd.Timestamp("2026-01-06", tz="UTC"), "1h")
    assert clean["t"].is_monotonic_increasing
    assert dropped["duplicate"] == 1


def test_coverage_passes_on_enough_data(small_dataset, raw_by_source):
    fetch.check_coverage(raw_by_source["nq"], small_dataset, "nq")
    fetch.check_coverage(raw_by_source["xyz"], small_dataset, "xyz")


def test_coverage_fails_when_validation_is_missing(small_dataset, raw_by_source):
    xyz = raw_by_source["xyz"]
    no_val = xyz[xyz["t"] >= pd.Timestamp("2026-05-01", tz="UTC")]
    with pytest.raises(DataCheckError, match="val"):
        fetch.check_coverage(no_val, small_dataset, "xyz")


def test_coverage_checks_train_only_for_the_train_source(small_dataset, raw_by_source):
    xyz = raw_by_source["xyz"]
    only_recent = xyz[xyz["t"] >= pd.Timestamp("2026-04-01", tz="UTC")]
    fetch.check_coverage(only_recent, small_dataset, "xyz")      # no train needed
    with pytest.raises(DataCheckError, match="train"):
        fetch.check_coverage(only_recent, small_dataset, "nq")


def test_fetch_source_writes_once_then_skips(tmp_path, small_dataset, raw_by_source, monkeypatch):
    calls = []

    def fake_fetcher(symbol, interval):
        calls.append(symbol)
        return raw_by_source["nq"]

    monkeypatch.setitem(fetch.FETCHERS, "yfinance", fake_fetcher)
    root = tmp_path.as_posix()
    now = pd.Timestamp("2026-06-10", tz="UTC")

    fetch.fetch_source(small_dataset, "nq", root, now)
    first = io.read_parquet(raw_path(root, small_dataset, "nq"))
    fetch.fetch_source(small_dataset, "nq", root, now)

    assert calls == ["NQ=F"]          # second call did not fetch
    pd.testing.assert_frame_equal(io.read_parquet(raw_path(root, small_dataset, "nq")), first)
