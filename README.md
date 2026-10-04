# ml-pipelines

Reproducible ML experiment pipelines for a home Kubernetes cluster (two Raspberry Pis).
The focus is the pipeline, not a clever model: fixed data snapshots, a purge between
train / validation / test, one test evaluation per experiment, and every run tracked in MLflow.

> Status: phase 1 — the platform runs on the cluster and experiments are tracked there.
> Running the pipeline itself as an Argo workflow follows in phase 2.

## Layout

| Path | What |
|---|---|
| `shared/mlkit/` | shared by every project: splits + purge, labels, AUC with block-bootstrap interval, MLflow conventions |
| `pipelines/nasdaq/` | first project: train on CME Nasdaq futures (`NQ=F`), validate and test on both NQ and the Hyperliquid perpetual `xyz:XYZ100` |
| `platform/` | Argo Workflows and MLflow, installed with Helm; see [platform/README.md](platform/README.md) |

## Platform

| Component | Where | State |
|---|---|---|
| Argo Workflows v4.1.4 | namespace `argo`, runs workflows in `ml` | — |
| MLflow 3.16.0 | namespace `mlflow`, UI on port 30500 (home network only) | metadata in Supabase Postgres (schema `mlflow`) |
| Data and models | Supabase Storage bucket `ml-data`, S3 API | outside the cluster |

The cluster holds no volumes: a pod can be replaced, upgraded or rolled back (`helm rollback`) without
losing runs or models. Things learned installing MLflow 3 on a Raspberry Pi (a read-only root
filesystem, ~2 GB of background job workers, presigned downloads) are in the platform values file.

## The Nasdaq experiment

A snapshot is defined in `pipelines/nasdaq/datasets/<name>.yaml` and fetched once.

```
NQ      |──────── train ────────|p|── val ──|p|── test ──|
XYZ100                             |── val ──|  |── test ──|
```

1. **fetch** — candles for both sources, cleaned (unaligned, unfinished and zero-volume candles dropped)
2. **features** — every feature in `features.FEATURES`; all relative, so price level does not matter
3. **train + validate** — one run per feature set, same Random Forest, same seed
4. **select + test** — best validation AUC wins; only the winner sees the test period, once
5. **register** — the winner becomes a new version of `nasdaq-direction`, alias `@candidate`

Validation and test are scored only on hours both sources have, so NQ and XYZ100 are compared on
the same moments.

## First result (snapshot `2026-10`)

Train up to 2026-05-31, validation June–July, test August–September 2026. AUC with 95% interval:

| Feature set | val NQ | val XYZ100 | test NQ | test XYZ100 |
|---|---|---|---|---|
| `base+ema10` (winner) | 0.495 (0.427–0.549) | 0.510 (0.452–0.557) | 0.489 (0.432–0.536) | 0.506 (0.446–0.553) |
| `base` | 0.493 (0.420–0.549) | 0.494 (0.444–0.538) | — | — |

Every interval contains 0.5: **these features carry no detectable edge**, and adding the distance to
EMA(10) does not change that. A second run on the same snapshot reproduces these numbers exactly.

## Run it locally

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e shared/mlkit -r pipelines/nasdaq/requirements.txt -r requirements-dev.txt
cd pipelines/nasdaq
../../.venv/Scripts/python run_local.py
../../.venv/Scripts/mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Adding a feature

1. Write a function in `pipelines/nasdaq/features.py` and add it to `FEATURES`.
2. Add a feature set that uses it to `FEATURE_SETS`.
3. Run the experiment; compare the variants in MLflow.

The tests check that every feature is independent of price level and never looks ahead.

## Disclaimer

Learning and portfolio project. Not financial advice, not used for live trading.
