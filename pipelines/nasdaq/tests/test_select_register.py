"""Choosing, testing and registering the winner, on synthetic data with a local MLflow."""
import mlflow
import pytest
from mlflow import MlflowClient

from dataset import MODEL_NAME
from features import build_features
from register import register_winner
from select_winner import AlreadyTested, evaluate_winner
from start import start_experiment
from train import train_validate


@pytest.fixture
def feats(raw_by_source, small_dataset):
    return {k: build_features(df, small_dataset.horizon, small_dataset.weekdays_only)
            for k, df in raw_by_source.items()}


def test_only_the_winner_is_tested_and_only_once(local_mlflow, feats, small_dataset):
    parent = start_experiment(small_dataset)
    runs = {fs: train_validate(feats, small_dataset, fs, parent) for fs in ("base", "base+ema10")}

    winner = evaluate_winner(feats, small_dataset, parent)

    client = MlflowClient()
    val = {fs: client.get_run(r).data.metrics["val_nq_auc"] for fs, r in runs.items()}
    assert winner == runs[max(val, key=val.get)]
    loser = next(r for r in runs.values() if r != winner)
    assert "test_nq_auc" in client.get_run(winner).data.metrics
    assert "test_nq_auc" not in client.get_run(loser).data.metrics
    with pytest.raises(AlreadyTested):
        evaluate_winner(feats, small_dataset, parent)


def test_register_points_candidate_at_the_winner(local_mlflow, feats, small_dataset):
    parent = start_experiment(small_dataset)
    train_validate(feats, small_dataset, "base", parent)
    evaluate_winner(feats, small_dataset, parent)

    version = register_winner(parent)

    mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, "candidate")
    assert mv.version == version
    assert mv.tags["feature_set"] == "base" and mv.tags["dataset"] == "test"
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@candidate")
    sample = feats["xyz"][["rsi_14", "rel_volume", "atr_pct", "macd_hist_pct"]].head(3)
    assert model.predict_proba(sample).shape == (3, 2)


def test_register_refuses_an_untested_experiment(local_mlflow, feats, small_dataset):
    parent = start_experiment(small_dataset)
    train_validate(feats, small_dataset, "base", parent)
    with pytest.raises(RuntimeError, match="no tested winner"):
        register_winner(parent)
