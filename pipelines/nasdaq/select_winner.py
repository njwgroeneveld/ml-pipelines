"""Step 4: pick the variant with the best validation AUC and test only that one, once.

The test period is read here and nowhere else. A parent run that was already
tested is refused, so the test set cannot be used to choose between variants.
"""
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient

from dataset import Dataset, load_dataset
from features import FEATURE_SETS
from mlkit.errors import run_main
from train import evaluate, load_features, paired_period


class AlreadyTested(Exception):
    pass


def select_winner(parent_run_id: str, ds: Dataset):
    client = MlflowClient()
    parent = client.get_run(parent_run_id)
    if parent.data.tags.get("test_evaluated") == "true":
        raise AlreadyTested(f"experiment {parent_run_id} was already tested")
    metric = f"val_{ds.train_on}_auc"
    children = client.search_runs(
        [parent.info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
        order_by=[f"metrics.{metric} DESC"],
    )
    if not children:
        raise RuntimeError(f"no variants under {parent_run_id}")
    return children[0]


def evaluate_winner(feats: dict[str, pd.DataFrame], ds: Dataset, parent_run_id: str) -> str:
    winner = select_winner(parent_run_id, ds)
    model = mlflow.sklearn.load_model(winner.data.tags["model_uri"])
    cols = FEATURE_SETS[winner.data.tags["feature_set"]]
    test_a, test_b = paired_period(feats, ds, "test")

    values = evaluate(model, cols, test_a, test_b, ds, "test")
    client = MlflowClient()
    for name, value in values.items():
        client.log_metric(winner.info.run_id, name, value)
    client.set_tag(parent_run_id, "test_evaluated", "true")
    client.set_tag(parent_run_id, "winner_run_id", winner.info.run_id)
    client.set_tag(parent_run_id, "winner_feature_set", winner.data.tags["feature_set"])
    return winner.info.run_id


def main() -> None:
    ds = load_dataset(os.environ["DATASET_FILE"])
    feats = load_features(ds, os.environ["DATA_ROOT"])
    print(f"winner: run {evaluate_winner(feats, ds, os.environ['PARENT_RUN_ID'])}")


if __name__ == "__main__":
    run_main(main)
