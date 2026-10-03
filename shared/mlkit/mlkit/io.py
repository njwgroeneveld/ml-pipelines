"""Read and write Parquet on a local path or on S3-compatible storage (Supabase Storage).

A path starting with s3:// goes to the endpoint in $S3_ENDPOINT; anything else is a
local file. The same code therefore runs on a laptop and on the cluster.
"""
import os
from pathlib import Path

import pandas as pd


def _is_s3(path: str) -> bool:
    return path.startswith("s3://")


def _storage_options(path: str) -> dict | None:
    if not _is_s3(path):
        return None
    return {"client_kwargs": {"endpoint_url": os.environ["S3_ENDPOINT"]}}


def read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path, storage_options=_storage_options(path))


def write_parquet(df: pd.DataFrame, path: str) -> None:
    if not _is_s3(path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, storage_options=_storage_options(path))


def exists(path: str) -> bool:
    if _is_s3(path):
        import s3fs

        fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": os.environ["S3_ENDPOINT"]})
        return fs.exists(path)
    return Path(path).exists()
