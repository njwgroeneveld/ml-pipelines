"""Choosing, testing and registering the winner, on synthetic data with a local MLflow."""
import mlflow
import pytest
from mlflow import MlflowClient

from dataset import MODEL_NAME
from features import build_features
from mlkit.errors import EXIT_DATA_CHECK, run_main
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


def test_a_failed_variant_is_never_the_winner(local_mlflow, feats, small_dataset):
    parent = start_experiment(small_dataset)
    good = train_validate(feats, small_dataset, "base", parent)
    # A retried train step leaves its first, failed attempt behind under the same parent.
    client = MlflowClient()
    experiment_id = client.get_run(parent).info.experiment_id
    failed = client.create_run(experiment_id, tags={"mlflow.parentRunId": parent, "feature_set": "base"})
    client.log_metric(failed.info.run_id, "val_nq_auc", 0.99)
    client.set_terminated(failed.info.run_id, "FAILED")

    assert evaluate_winner(feats, small_dataset, parent) == good


def test_a_crash_before_the_last_tag_leaves_the_experiment_retryable(local_mlflow, feats, small_dataset, monkeypatch):
    parent = start_experiment(small_dataset)
    run = train_validate(feats, small_dataset, "base", parent)
    real_set_tag = MlflowClient.set_tag

    def crash_on_winner_tag(self, run_id, key, value, *args, **kwargs):
        if key == "winner_feature_set":
            raise ConnectionError("tracking server went away")
        return real_set_tag(self, run_id, key, value, *args, **kwargs)

    monkeypatch.setattr(MlflowClient, "set_tag", crash_on_winner_tag)
    with pytest.raises(ConnectionError):
        evaluate_winner(feats, small_dataset, parent)
    monkeypatch.setattr(MlflowClient, "set_tag", real_set_tag)

    assert evaluate_winner(feats, small_dataset, parent) == run  # what Argo's retry does


def test_an_already_tested_experiment_fails_without_retry():
    def main():
        raise AlreadyTested("experiment abc was already tested")

    with pytest.raises(SystemExit) as exc:
        run_main(main)
    assert exc.value.code == EXIT_DATA_CHECK
