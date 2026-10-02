import pandas as pd

from mlkit import io


def test_write_then_read_local_roundtrip(tmp_path):
    path = str(tmp_path / "nested" / "dir" / "x.parquet")
    df = pd.DataFrame({"t": pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC"), "c": [1.0, 2.0, 3.0]})

    io.write_parquet(df, path)

    pd.testing.assert_frame_equal(io.read_parquet(path), df)


def test_exists_local(tmp_path):
    path = str(tmp_path / "x.parquet")
    assert not io.exists(path)
    io.write_parquet(pd.DataFrame({"a": [1]}), path)
    assert io.exists(path)
