"""Step 3: train one feature set on the train period and validate it on both sources.

One call = one nested MLflow run under the experiment's parent run.
"""
import os

import mlflow
import pandas as pd

from dataset import PROJECT, Dataset, load_dataset
from features import FEATURE_SETS, features_by_source
from mlkit.errors import DataCheckError, run_main
from mlkit.metrics import direction_agreement
from mlkit.models import RF_PARAMS, make_model
from mlkit.splits import common_timestamps, split_periods
from mlkit.tracking import eval_metrics, experiment_name, log_model, provenance_tags

COST_BPS = 10.0
MIN_TRAIN, MIN_EVAL = 1000, 200


def paired_period(feats: dict[str, pd.DataFrame], ds: Dataset, period: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The rows of `period` for train_on and eval_on, limited to the hours both have."""
    a = split_periods(feats[ds.train_on], ds.periods, ds.horizon)[period]
    b = split_periods(feats[ds.eval_on], ds.periods, ds.horizon)[period]
    a, b = common_timestamps(a, b)
    if len(a) < MIN_EVAL:
        raise DataCheckError(f"{period}: only {len(a)} shared candles, need {MIN_EVAL}")
    return a, b


def evaluate(model, cols: list[str], a: pd.DataFrame, b: pd.DataFrame, ds: Dataset, prefix: str) -> dict[str, float]:
    """Metrics for one period on both sources, plus how often they agree."""
    p_a = model.predict_proba(a[cols])[:, 1]
    p_b = model.predict_proba(b[cols])[:, 1]
    values = {
        **eval_metrics(f"{prefix}_{ds.train_on}", a["label"], p_a, a["fwd_ret"], ds.horizon, COST_BPS),
        **eval_metrics(f"{prefix}_{ds.eval_on}", b["label"], p_b, b["fwd_ret"], ds.horizon, COST_BPS),
        f"{prefix}_agreement": direction_agreement(p_a, p_b),
    }
    return values


def train_validate(feats: dict[str, pd.DataFrame], ds: Dataset, feature_set: str, parent_run_id: str) -> str:
    cols = FEATURE_SETS[feature_set]
    train = split_periods(feats[ds.train_on], ds.periods, ds.horizon)["train"]
    if len(train) < MIN_TRAIN:
        raise DataCheckError(f"train: only {len(train)} rows, need {MIN_TRAIN}")
    val_a, val_b = paired_period(feats, ds, "val")

    model = make_model().fit(train[cols], train["label"].astype(int))

    mlflow.set_experiment(experiment_name(PROJECT, "features"))
    tags = {**provenance_tags(ds.name), "feature_set": feature_set}
    with mlflow.start_run(run_name=feature_set, parent_run_id=parent_run_id, tags=tags) as run:
        mlflow.log_params({
            **RF_PARAMS, "model": "rf", "feature_set": feature_set, "features": ",".join(cols),
            "train_on": ds.train_on, "eval_on": ds.eval_on, "horizon": ds.horizon, "cost_bps": COST_BPS,
            "train_start": str(train["t"].min()), "train_end": str(train["t"].max()), "train_rows": len(train),
        })
        mlflow.log_metrics(evaluate(model, cols, val_a, val_b, ds, "val"))
        mlflow.log_metrics({f"importance_{c}": float(i) for c, i in zip(cols, model.feature_importances_, strict=True)})
        mlflow.set_tag("model_uri", log_model(model, train[cols]))
        return run.info.run_id


def main() -> None:
    ds = load_dataset(os.environ["DATASET_FILE"])
    feats = features_by_source(ds, os.environ["DATA_ROOT"])
    run_id = train_validate(feats, ds, os.environ["FEATURE_SET"], os.environ["PARENT_RUN_ID"])
    print(f"{os.environ['FEATURE_SET']}: run {run_id}")


if __name__ == "__main__":
    run_main(main)
