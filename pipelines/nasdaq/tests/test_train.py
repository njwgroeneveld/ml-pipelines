"""Training variants under one parent run, on synthetic data with a local MLflow."""
import pytest
from mlflow import MlflowClient

from features import build_features
from start import start_experiment
from train import train_validate


@pytest.fixture
def feats(raw_by_source, small_dataset):
    return {k: build_features(df, small_dataset.horizon, small_dataset.weekdays_only)
            for k, df in raw_by_source.items()}


def test_variants_hang_under_one_parent_with_val_metrics(local_mlflow, feats, small_dataset):
    parent = start_experiment(small_dataset)
    run_ids = [train_validate(feats, small_dataset, fs, parent) for fs in ("base", "base+ema10")]

    client = MlflowClient()
    for run_id in run_ids:
        run = client.get_run(run_id)
        assert run.data.tags["mlflow.parentRunId"] == parent
        for name in ("val_nq_auc", "val_xyz_auc", "val_agreement", "val_xyz_n_rows"):
            assert name in run.data.metrics
        assert not any(k.startswith("test_") for k in run.data.metrics)   # test is untouched
