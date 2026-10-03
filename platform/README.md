# Platform

Shared by every pipeline in this repo: Argo Workflows runs the steps, MLflow tracks runs and
models. Both are installed with Helm on the home cluster; state lives outside it (Supabase for
MLflow metadata, Cloudflare R2 for data and models), so the cluster holds no volumes.

| Component | Chart | Namespace | Reach it |
|---|---|---|---|
| Argo Workflows v4.1.4 | `argo/argo-workflows` 2.0.9 | `argo` | `kubectl -n argo port-forward svc/argo-workflows-server 2746:2746` |
| MLflow 3.16.0 | `community-charts/mlflow` 1.11.7 | `mlflow` | `http://<node-ip>:30500` (home network / Tailscale only, no login) |
| Pipeline pods | – | `ml` | ServiceAccount `ml-workflow` |

## Install

Run on a machine with `kubectl` and `helm` for the cluster, from this directory:

```bash
kubectl apply -f namespaces.yaml
bash create-secrets.sh                       # asks for every value
bash check-supabase.sh <pooler-host>         # one login; must succeed before MLflow
bash check-r2.sh write
helm repo add argo https://argoproj.github.io/argo-helm
helm repo add community-charts https://community-charts.github.io/helm-charts
helm repo update
helm upgrade --install argo-workflows argo/argo-workflows --version 2.0.9 -n argo -f argo/values.yaml --wait
cp mlflow/values.local.example.yaml mlflow/values.local.yaml   # then fill it in
helm upgrade --install mlflow community-charts/mlflow --version 1.11.7 -n mlflow \
  -f mlflow/values.yaml -f mlflow/values.local.yaml --wait --timeout 10m
```

## Upgrade and roll back

Change a values file or a `--version`, run the same `helm upgrade --install` command, then:

```bash
helm history mlflow -n mlflow
helm rollback mlflow <revision> -n mlflow --wait
```

## If MLflow cannot log in to Supabase

Stop it at once: every restart is another failed login, and after too many the pooler blocks
the whole project.

```bash
kubectl -n mlflow scale deployment/mlflow --replicas=0
```

Fix the secret, run `check-supabase.sh` again, then scale back to 1.
