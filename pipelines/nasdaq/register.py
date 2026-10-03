"""Step 5: register the tested winner as a new version of the model, alias @candidate."""
import os

from mlflow import MlflowClient

from dataset import MODEL_NAME
from mlkit.errors import run_main
from mlkit.tracking import register_candidate

COPIED_TAGS = ["feature_set", "dataset", "image_tag", "git_sha"]


def register_winner(parent_run_id: str) -> str:
    client = MlflowClient()
    parent = client.get_run(parent_run_id)
    if parent.data.tags.get("test_evaluated") != "true":
        raise RuntimeError(f"experiment {parent_run_id} has no tested winner yet")
    winner = client.get_run(parent.data.tags["winner_run_id"])
    tags = {k: winner.data.tags[k] for k in COPIED_TAGS}
    return register_candidate(MODEL_NAME, winner.data.tags["model_uri"], tags)


def main() -> None:
    version = register_winner(os.environ["PARENT_RUN_ID"])
    print(f"{MODEL_NAME} version {version} -> @candidate")


if __name__ == "__main__":
    run_main(main)
