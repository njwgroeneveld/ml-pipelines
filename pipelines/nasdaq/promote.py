"""Promote a registered version of the model to @champion. Always run by hand, never by the pipeline.

    python promote.py               # the current @candidate becomes @champion
    python promote.py --version 4   # a named version, e.g. to roll back

Needs MLFLOW_TRACKING_URI pointing at the MLflow server.
"""
import argparse

from dataset import MODEL_NAME
from mlkit.tracking import promote_champion


def main() -> None:
    parser = argparse.ArgumentParser(description="Point @champion at a model version.")
    parser.add_argument("--version", help="version to promote (default: the current @candidate)")
    args = parser.parse_args()

    new, previous = promote_champion(MODEL_NAME, args.version)
    print(f"{MODEL_NAME}@champion -> version {new} (was {previous or 'none'})")
    if previous and previous != new:
        print(f"roll back with: python promote.py --version {previous}")


if __name__ == "__main__":
    main()
