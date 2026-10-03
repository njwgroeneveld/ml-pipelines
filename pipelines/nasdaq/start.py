"""Step 0: open the parent MLflow run that every variant of this experiment hangs under."""
import os
from pathlib import Path

import mlflow

from dataset import PROJECT, Dataset, load_dataset
from mlkit.errors import run_main
from mlkit.tracking import experiment_name, provenance_tags


def start_experiment(ds: Dataset) -> str:
    mlflow.set_experiment(experiment_name(PROJECT, "features"))
    with mlflow.start_run(run_name=f"experiment-{ds.name}", tags=provenance_tags(ds.name)) as run:
        return run.info.run_id


def main() -> None:
    run_id = start_experiment(load_dataset(os.environ["DATASET_FILE"]))
    # Argo reads this file as the step's output parameter.
    Path(os.environ.get("OUTPUT_FILE", "/tmp/parent_run_id")).write_text(run_id)
    print(run_id)


if __name__ == "__main__":
    run_main(main)
