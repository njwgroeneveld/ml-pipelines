import numpy as np
import pandas as pd

from dataset import raw_path
from features import FEATURE_SETS, FEATURES, build_features, features_by_source
from mlkit import io


def test_every_feature_set_uses_known_features():
    for name, cols in FEATURE_SETS.items():
        assert set(cols) <= set(FEATURES), name


def test_output_has_no_missing_or_infinite_values(raw_by_source):
    feats = build_features(raw_by_source["xyz"], horizon=4, weekdays_only=True)
    values = feats[list(FEATURES)].to_numpy()
    assert np.isfinite(values).all()
    assert set(feats["label"].unique()) <= {0.0, 1.0}


def test_weekends_are_removed(raw_by_source):
    feats = build_features(raw_by_source["xyz"], horizon=4, weekdays_only=True)
    assert (feats["t"].dt.dayofweek < 5).all()


def test_weekends_kept_when_asked(raw_by_source):
    feats = build_features(raw_by_source["xyz"], horizon=4, weekdays_only=False)
    assert (feats["t"].dt.dayofweek >= 5).any()


def test_features_do_not_depend_on_price_level_or_volume_unit(raw_by_source):
    """The reason every feature is relative: NQ and XYZ100 trade at different levels."""
    raw = raw_by_source["nq"]
    scaled = raw.copy()
    scaled[["o", "h", "l", "c"]] *= 10
    scaled["v"] *= 3

    a = build_features(raw, horizon=4, weekdays_only=True)
    b = build_features(scaled, horizon=4, weekdays_only=True)

    pd.testing.assert_frame_equal(a[list(FEATURES)], b[list(FEATURES)], rtol=1e-9)


def test_features_only_look_backwards(raw_by_source):
    """Changing a future candle must not change any feature before it."""
    raw = raw_by_source["nq"].reset_index(drop=True)
    cut = len(raw) // 2
    changed = raw.copy()
    changed.loc[cut:, ["o", "h", "l", "c"]] *= 1.5

    a = build_features(raw, horizon=4, weekdays_only=True)
    b = build_features(changed, horizon=4, weekdays_only=True)
    before = a["t"] < raw.loc[cut, "t"]

    pd.testing.assert_frame_equal(a.loc[before, list(FEATURES)], b.loc[before, list(FEATURES)])


def test_label_matches_close_horizon_rows_later(raw_by_source):
    feats = build_features(raw_by_source["nq"], horizon=4, weekdays_only=True)
    raw = raw_by_source["nq"].set_index("t")["c"]
    row = feats.iloc[100]
    later = raw.index.get_loc(row["t"]) + 4
    assert row["label"] == float(raw.iloc[later] > row["c"])


def test_features_by_source_reads_the_raw_snapshot(tmp_path, small_dataset, raw_by_source):
    root = tmp_path.as_posix()
    for key, raw in raw_by_source.items():
        io.write_parquet(raw, raw_path(root, small_dataset, key))

    feats = features_by_source(small_dataset, root)

    assert set(feats) == {"nq", "xyz"}
    expected = build_features(raw_by_source["nq"], small_dataset.horizon, small_dataset.weekdays_only)
    pd.testing.assert_frame_equal(feats["nq"], expected)
