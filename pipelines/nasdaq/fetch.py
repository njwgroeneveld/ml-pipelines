"""Step 1: fetch the candles of every source in the dataset into the snapshot.

A source whose file already exists is skipped: a snapshot is never overwritten.
"""
import os
import time

import pandas as pd
import requests

from dataset import Dataset, load_dataset, raw_path
from mlkit import io
from mlkit.errors import DataCheckError, run_main

COLUMNS = ["t", "o", "h", "l", "c", "v"]
MIN_ROWS = {"train": 1000, "val": 200, "test": 200}
HL_URL = "https://api.hyperliquid.xyz/info"
HL_MAX_CANDLES = 5000


def from_yfinance(symbol: str, interval: str) -> pd.DataFrame:
    import yfinance as yf

    # Free intraday history only goes back about 730 days.
    raw = yf.download(symbol, period="729d", interval=interval, auto_adjust=False, progress=False)
    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {symbol}")  # retried by Argo
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns={"Open": "o", "High": "h", "Low": "l", "Close": "c", "Volume": "v"})
    raw.index = pd.to_datetime(raw.index, utc=True)
    df = raw.reset_index(names="t")[COLUMNS]
    return df.astype({c: float for c in COLUMNS[1:]})


def from_hyperliquid(symbol: str, interval: str) -> pd.DataFrame:
    # Hyperliquid keeps only the most recent ~5000 candles per interval.
    end = int(time.time() * 1000)
    start = end - 2 * 365 * 24 * 3600 * 1000
    rows = []
    while start < end:
        r = requests.post(HL_URL, timeout=30, json={
            "type": "candleSnapshot",
            "req": {"coin": symbol, "interval": interval, "startTime": start, "endTime": end},
        })
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows += batch
        start = batch[-1]["t"] + 1
        if len(batch) < HL_MAX_CANDLES:
            break
    if not rows:
        raise RuntimeError(f"hyperliquid returned no data for {symbol}")
    df = pd.DataFrame(rows)[COLUMNS]
    df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    return df.astype({c: float for c in COLUMNS[1:]})


FETCHERS = {"yfinance": from_yfinance, "hyperliquid": from_hyperliquid}


def clean_candles(df: pd.DataFrame, now: pd.Timestamp, interval: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop candles that would mislead the features; report how many per reason."""
    step = pd.Timedelta(interval)
    dropped = {}

    aligned = df["t"].dt.floor(step) == df["t"]
    dropped["not_aligned"] = int((~aligned).sum())
    df = df[aligned]

    complete = df["t"] + step <= now
    dropped["incomplete"] = int((~complete).sum())
    df = df[complete]

    traded = df["v"] > 0
    dropped["zero_volume"] = int((~traded).sum())
    df = df[traded]

    before = len(df)
    df = df.drop_duplicates("t").sort_values("t").reset_index(drop=True)
    dropped["duplicate"] = before - len(df)
    return df, dropped


def check_coverage(df: pd.DataFrame, ds: Dataset, key: str) -> None:
    """Fail without retry if a period the experiment needs has too few candles."""
    p = ds.periods
    windows = {
        "val": (p.val_from, p.val_to),
        "test": (p.test_from, p.test_to),
    }
    if key == ds.train_on:
        windows["train"] = (df["t"].min(), p.train_end)
    for name, (start, end) in windows.items():
        n = int(((df["t"] >= start) & (df["t"] < end + pd.Timedelta(days=1))).sum())
        if n < MIN_ROWS[name]:
            raise DataCheckError(f"{key}: {n} candles in {name}, need {MIN_ROWS[name]}")


def fetch_source(ds: Dataset, key: str, root: str, now: pd.Timestamp) -> str:
    out = raw_path(root, ds, key)
    if io.exists(out):
        print(f"{key}: {out} exists, skipping")
        return out
    src = ds.sources[key]
    df, dropped = clean_candles(FETCHERS[src.source](src.symbol, ds.interval), now, ds.interval)
    check_coverage(df, ds, key)
    io.write_parquet(df, out)
    print(f"{key}: {len(df)} candles {df['t'].min()} -> {df['t'].max()}, dropped {dropped} -> {out}")
    return out


def main() -> None:
    ds = load_dataset(os.environ["DATASET_FILE"])
    now = pd.Timestamp.now(tz="UTC")
    for key in ds.sources:
        fetch_source(ds, key, os.environ["DATA_ROOT"], now)


if __name__ == "__main__":
    run_main(main)
