"""Run the whole experiment on a laptop: local files, local MLflow database, no cluster.

    python run_local.py                         # dataset 2026-10, every feature set
    python run_local.py --feature-sets base     # a subset

Data goes to ./data and MLflow to ./mlflow.db. Look at the runs with:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import argparse
import os
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient

from dataset import load_dataset
from features import FEATURE_SETS, features_by_source
from fetch import fetch_source
from register import register_winner
from select_winner import evaluate_winner
from start import start_experiment
from train import train_validate

HERE = Path(__file__).resolve().parent


def summary(parent: str, ds) -> list[str]:
    client = MlflowClient()
    experiment_id = client.get_run(parent).info.experiment_id
    lines = []
    for run in client.search_runs([experiment_id], filter_string=f"tags.mlflow.parentRunId = '{parent}'"):
        m = run.data.metrics
        line = f"{run.data.tags['feature_set']:<12}"
        for name in (f"{p}_{k}_auc" for p in ("val", "test") for k in (ds.train_on, ds.eval_on)):
            if name in m:
                line += f"  {name} {m[name]:.3f} ({m[name + '_ci_low']:.3f}-{m[name + '_ci_high']:.3f})"
        lines.append(line)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="2026-10")
    parser.add_argument("--feature-sets", nargs="+", default=list(FEATURE_SETS))
    args = parser.parse_args()

    os.environ.setdefault("MLFLOW_TRACKING_URI", f"sqlite:///{(HERE / 'mlflow.db').as_posix()}")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    root = (HERE / "data").as_posix()
    ds = load_dataset(HERE / "datasets" / f"{args.dataset}.yaml")

    now = pd.Timestamp.now(tz="UTC")
    for key in ds.sources:
        fetch_source(ds, key, root, now)
    feats = features_by_source(ds, root)

    parent = start_experiment(ds)
    for feature_set in args.feature_sets:
        train_validate(feats, ds, feature_set, parent)
    winner = evaluate_winner(feats, ds, parent)
    version = register_winner(parent)

    print(f"\nexperiment {parent} -> winner {winner}, registered as version {version} (@candidate)\n")
    print("\n".join(summary(parent, ds)))


if __name__ == "__main__":
    main()
