import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from mlkit import tracking
from mlkit.models import make_model


@pytest.fixture
def local_mlflow(tmp_path, monkeypatch):
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.chdir(tmp_path)  # MLflow writes model files to ./mlruns
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("test/kind")
    yield


def test_experiment_name():
    assert tracking.experiment_name("nasdaq", "features") == "nasdaq/features"


def test_provenance_defaults_to_local(monkeypatch):
    monkeypatch.delenv("IMAGE_TAG", raising=False)
    monkeypatch.delenv("GIT_SHA", raising=False)
    assert tracking.provenance_tags("2026-10") == {"dataset": "2026-10", "image_tag": "local", "git_sha": "local"}


def test_eval_metrics_names_carry_the_prefix():
    rng = np.random.default_rng(0)
    y, p, fwd = rng.integers(0, 2, 200), rng.random(200), rng.normal(0, 0.01, 200)
    values = tracking.eval_metrics("val_nq", y, p, fwd, horizon=4, cost_bps=10)
    assert set(values) == {
        "val_nq_auc", "val_nq_auc_ci_low", "val_nq_auc_ci_high", "val_nq_baseline_up_rate",
        "val_nq_n_rows", "val_nq_strat_t55_return", "val_nq_strat_t55_trades",
    }


def _fitted_model():
    X = pd.DataFrame(np.random.default_rng(0).random((200, 2)), columns=["f1", "f2"])
    return make_model().fit(X, (X["f1"] > 0.5).astype(int)), X


def test_logged_model_loads_back_and_predicts_the_same(local_mlflow):
    model, X = _fitted_model()
    with mlflow.start_run():
        uri = tracking.log_model(model, X)

    loaded = mlflow.sklearn.load_model(uri)

    np.testing.assert_allclose(loaded.predict_proba(X)[:, 1], model.predict_proba(X)[:, 1])


def test_register_candidate_sets_alias(local_mlflow):
    model, X = _fitted_model()
    with mlflow.start_run():
        uri = tracking.log_model(model, X)

    version = tracking.register_candidate("demo-direction", uri, {"feature_set": "base"})

    mv = MlflowClient().get_model_version_by_alias("demo-direction", "candidate")
    assert mv.version == version
    assert mv.tags["feature_set"] == "base"


def _register_versions(name: str, n: int) -> list[str]:
    """n versions of one model; each new one becomes @candidate, like a pipeline run."""
    model, X = _fitted_model()
    versions = []
    for _ in range(n):
        with mlflow.start_run():
            versions.append(tracking.register_candidate(name, tracking.log_model(model, X), {}))
    return versions


def test_promote_defaults_to_the_candidate(local_mlflow):
    _, latest = _register_versions("demo-direction", 2)

    assert tracking.promote_champion("demo-direction") == (latest, None)
    assert MlflowClient().get_model_version_by_alias("demo-direction", "champion").version == latest


def test_promote_reports_the_previous_champion_for_a_rollback(local_mlflow):
    first, latest = _register_versions("demo-direction", 2)
    tracking.promote_champion("demo-direction")

    assert tracking.promote_champion("demo-direction", first) == (first, latest)
    assert MlflowClient().get_model_version_by_alias("demo-direction", "champion").version == first


def test_promote_refuses_an_unknown_version(local_mlflow):
    _register_versions("demo-direction", 1)

    with pytest.raises(MlflowException):
        tracking.promote_champion("demo-direction", "99")
    with pytest.raises(MlflowException):
        MlflowClient().get_model_version_by_alias("demo-direction", "champion")
