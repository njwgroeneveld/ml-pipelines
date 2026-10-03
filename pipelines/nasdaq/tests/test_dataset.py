import pandas as pd
import pytest

from dataset import load_dataset, raw_path


def test_real_dataset_loads(real_dataset):
    assert real_dataset.name == "2026-10"
    assert real_dataset.train_on == "nq" and real_dataset.eval_on == "xyz"
    assert real_dataset.sources["xyz"].symbol == "xyz:XYZ100"
    assert real_dataset.periods.val_from == pd.Timestamp("2026-06-01", tz="UTC")


def test_periods_are_in_order(real_dataset):
    p = real_dataset.periods
    assert p.train_end < p.val_from <= p.val_to < p.test_from <= p.test_to


def test_dataset_needs_two_sources(tmp_path):
    f = tmp_path / "bad.yaml"
    f.write_text(
        "name: bad\ninterval: 1h\nhorizon: 4\nweekdays_only: true\n"
        "sources: {nq: {source: yfinance, symbol: NQ=F}}\ntrain_on: nq\n"
        "periods: {train: {end: 2026-01-01}, val: {from: 2026-01-02, to: 2026-01-03},"
        " test: {from: 2026-01-04, to: 2026-01-05}}\n"
    )
    with pytest.raises(ValueError):
        load_dataset(f)


def test_paths(real_dataset):
    assert raw_path("s3://ml-data/nasdaq/datasets", real_dataset, "nq") == \
        "s3://ml-data/nasdaq/datasets/2026-10/raw_nq.parquet"
