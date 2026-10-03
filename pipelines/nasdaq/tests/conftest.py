from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from dataset import Dataset, Source, load_dataset
from mlkit.splits import Periods

HERE = Path(__file__).resolve().parent


def synthetic_candles(start: str, end: str, level: float, seed: int, weekends: bool) -> pd.DataFrame:
    """Hourly random-walk candles; level sets the price scale."""
    t = pd.date_range(start, end, freq="h", tz="UTC")
    if not weekends:
        t = t[t.dayofweek < 5]
    rng = np.random.default_rng(seed)
    c = level * np.exp(np.cumsum(rng.normal(0, 0.002, len(t))))
    o = np.r_[c[0], c[:-1]]
    spread = np.abs(rng.normal(0, 0.001, len(t))) * c
    return pd.DataFrame({
        "t": t, "o": o, "h": np.maximum(o, c) + spread, "l": np.minimum(o, c) - spread,
        "c": c, "v": rng.uniform(100, 1000, len(t)),
    })


@pytest.fixture
def small_dataset() -> Dataset:
    """Same shape as datasets/2026-10.yaml, but over a few synthetic months."""
    return Dataset(
        name="test", interval="1h", horizon=4, weekdays_only=True,
        sources={"nq": Source("yfinance", "NQ=F"), "xyz": Source("hyperliquid", "xyz:XYZ100")},
        train_on="nq",
        periods=Periods.from_dict({
            "train": {"end": "2026-03-31"},
            "val": {"from": "2026-04-01", "to": "2026-04-30"},
            "test": {"from": "2026-05-01", "to": "2026-05-31"},
        }),
    )


@pytest.fixture
def raw_by_source() -> dict[str, pd.DataFrame]:
    return {
        "nq": synthetic_candles("2025-10-01", "2026-06-05", level=25_000, seed=1, weekends=False),
        "xyz": synthetic_candles("2026-02-01", "2026-06-05", level=2_500, seed=2, weekends=True),
    }


@pytest.fixture
def real_dataset() -> Dataset:
    return load_dataset(HERE.parent / "datasets" / "2026-10.yaml")


@pytest.fixture
def local_mlflow(tmp_path, monkeypatch):
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.chdir(tmp_path)  # MLflow writes model files to ./mlruns
    mlflow.set_tracking_uri(uri)
    yield uri
