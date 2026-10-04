# ml-pipelines

Reproducible ML experiment pipelines for a home Kubernetes cluster (two Raspberry Pis).
The focus is the pipeline, not a clever model: fixed data snapshots, a purge between
train / validation / test, one test evaluation per experiment, and every run tracked in MLflow.

> Status: phase 2 — the experiment runs as an Argo workflow on the cluster.
> Next: presenting the result, and promoting a candidate to `@champion` by hand.

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

1. **fetch** — candles for both sources, cleaned (unaligned, unfinished and zero-volume candles dropped);
   skipped when the snapshot already exists
2. **train + validate** — one run per feature set, same Random Forest, same seed; features are computed
   in-process from the raw snapshot (never stored) and are all relative, so price level does not matter
3. **select + test** — best validation AUC wins; only the winner sees the test period, once
4. **register** — the winner becomes a new version of `nasdaq-direction`, alias `@candidate`

Validation and test are scored only on hours both sources have, so NQ and XYZ100 are compared on
the same moments.

## First result (snapshot `2026-10`)

Train up to 2026-05-31, validation June–July, test August–September 2026. AUC with 95% interval
(block bootstrap), from the runs on the cluster:

| Feature set | val NQ | val XYZ100 | test NQ | test XYZ100 |
|---|---|---|---|---|
| `base+ema10` (winner) | 0.499 (0.431–0.556) | 0.508 (0.454–0.553) | 0.485 (0.427–0.533) | 0.507 (0.449–0.556) |
| `base` | 0.494 (0.424–0.551) | 0.499 (0.446–0.545) | — | — |

Every interval contains 0.5: **these features carry no detectable edge**, and adding the distance to
EMA(10) does not change that. Three workflow runs in a row gave identical numbers to six decimals.
(The first local run used a snapshot fetched two days earlier; its AUCs differ in the third decimal.)

## Run it locally

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e shared/mlkit -r pipelines/nasdaq/requirements.txt -r requirements-dev.txt
cd pipelines/nasdaq
../../.venv/Scripts/python run_local.py
../../.venv/Scripts/mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Run it on the cluster

A tag `nasdaq-v<version>` makes GitHub Actions run the tests and build the arm64 image
`ghcr.io/njwgroeneveld/ml-pipelines-nasdaq:<version>`. Then, on a machine with `kubectl` and
the [argo CLI](https://github.com/argoproj/argo-workflows/releases):

```bash
argo submit -n ml pipelines/nasdaq/image-check.yaml -p image_tag=0.2.0 --log   # once per image, on both nodes
kubectl apply -f pipelines/nasdaq/workflow-template.yaml
argo submit -n ml --from workflowtemplate/nasdaq-experiment -p image_tag=0.2.0
argo get -n ml @latest
```

```
start -> fetch -> train-validate (one pod per feature set, two at a time) -> select-winner -> register
```

A failed step is retried twice with backoff, except a failed data check (exit code 2). A run takes
about five minutes on the Raspberry Pis. Deleting a training pod halfway through is survived: Argo
retries the step, and only finished MLflow runs can become the winner.

## Adding a feature

1. Write a function in `pipelines/nasdaq/features.py` and add it to `FEATURES`.
2. Add a feature set that uses it to `FEATURE_SETS`.
3. Run the experiment; compare the variants in MLflow.

The tests check that every feature is independent of price level and never looks ahead.

## Disclaimer

Learning and portfolio project. Not financial advice, not used for live trading.
