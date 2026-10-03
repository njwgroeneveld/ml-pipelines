"""MLflow conventions shared by every project: names, metrics, provenance, registry."""
import os

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature

from . import metrics
from .models import SKOPS_TRUSTED_TYPES

CANDIDATE_ALIAS = "candidate"


def experiment_name(project: str, kind: str) -> str:
    return f"{project}/{kind}"


def provenance_tags(dataset: str) -> dict[str, str]:
    """Which data, image and code produced a run. Argo sets IMAGE_TAG and GIT_SHA."""
    return {
        "dataset": dataset,
        "image_tag": os.environ.get("IMAGE_TAG", "local"),
        "git_sha": os.environ.get("GIT_SHA", "local"),
    }


def eval_metrics(prefix: str, y, p, fwd_ret, horizon: int, cost_bps: float) -> dict[str, float]:
    auc, lo, hi = metrics.block_bootstrap_auc(y, p)
    strat = metrics.strategy_result(p, fwd_ret, threshold=0.55, cost_bps=cost_bps, step=horizon)
    return {
        f"{prefix}_auc": auc,
        f"{prefix}_auc_ci_low": lo,
        f"{prefix}_auc_ci_high": hi,
        f"{prefix}_baseline_up_rate": metrics.baseline_up_rate(y),
        f"{prefix}_n_rows": float(len(y)),
        f"{prefix}_strat_t55_return": strat["return"],
        f"{prefix}_strat_t55_trades": float(strat["trades"]),
    }


def log_model(model, X_example) -> str:
    """Log a fitted classifier with its input signature; returns the model URI."""
    signature = infer_signature(X_example, model.predict_proba(X_example)[:, 1])
    info = mlflow.sklearn.log_model(
        model,
        name="model",
        signature=signature,
        input_example=X_example.head(5),
        skops_trusted_types=SKOPS_TRUSTED_TYPES,
    )
    return info.model_uri


def register_candidate(model_name: str, model_uri: str, tags: dict[str, str]) -> str:
    """Register a model version and point the 'candidate' alias at it."""
    version = mlflow.register_model(model_uri, model_name, tags=tags).version
    MlflowClient().set_registered_model_alias(model_name, CANDIDATE_ALIAS, version)
    return version
