"""Load a snapshot definition from datasets/<name>.yaml."""
from dataclasses import dataclass
from pathlib import Path

import yaml

from mlkit.splits import Periods

PROJECT = "nasdaq"
MODEL_NAME = "nasdaq-direction"


@dataclass(frozen=True)
class Source:
    source: str   # yfinance | hyperliquid
    symbol: str


@dataclass(frozen=True)
class Dataset:
    name: str
    interval: str
    horizon: int
    weekdays_only: bool
    sources: dict[str, Source]
    train_on: str
    periods: Periods

    @property
    def eval_on(self) -> str:
        """The source that is only evaluated on, never trained on."""
        (other,) = [k for k in self.sources if k != self.train_on]
        return other


def load_dataset(path: str | Path) -> Dataset:
    raw = yaml.safe_load(Path(path).read_text())
    ds = Dataset(
        name=str(raw["name"]),
        interval=raw["interval"],
        horizon=int(raw["horizon"]),
        weekdays_only=bool(raw["weekdays_only"]),
        sources={k: Source(**v) for k, v in raw["sources"].items()},
        train_on=raw["train_on"],
        periods=Periods.from_dict(raw["periods"]),
    )
    if len(ds.sources) != 2 or ds.train_on not in ds.sources:
        raise ValueError("a dataset needs exactly two sources, one of them train_on")
    return ds


def raw_path(root: str, ds: Dataset, key: str) -> str:
    return f"{root}/{ds.name}/raw_{key}.parquet"


def feat_path(root: str, ds: Dataset, key: str) -> str:
    return f"{root}/{ds.name}/feat_{key}.parquet"
