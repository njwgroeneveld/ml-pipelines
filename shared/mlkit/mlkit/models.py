"""The model, with fixed settings, so that experiments differ only in their features."""
from sklearn.ensemble import RandomForestClassifier

RF_PARAMS = {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 50, "n_jobs": -1}

# MLflow 3.16 saves sklearn models with skops, which refuses types it cannot vet.
# Tree is the node storage of every tree model; we only load files we wrote ourselves.
SKOPS_TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]


def make_model(kind: str = "rf", seed: int = 42) -> RandomForestClassifier:
    if kind != "rf":
        raise ValueError(f"unknown model kind: {kind}")
    return RandomForestClassifier(**RF_PARAMS, random_state=seed)
