#!/bin/bash
# Log in to Supabase ONCE as mlflow_user, from a temporary pod, with the mlflow-db secret.
# Run before installing MLflow: a crash-looping MLflow with a wrong password means repeated
# failed logins, and the Supabase pooler then blocks the whole project (ECIRCUITBREAKER).
#     bash check-supabase.sh <pooler-host>
set -euo pipefail
host="${1:?usage: check-supabase.sh <pooler-host>}"
pod=check-supabase

kubectl -n mlflow delete pod "$pod" --ignore-not-found >/dev/null
kubectl -n mlflow apply -f - >/dev/null <<YAML
apiVersion: v1
kind: Pod
metadata: {name: $pod}
spec:
  restartPolicy: Never          # exactly one login attempt
  containers:
  - name: psql
    image: postgres:17-alpine
    command:
    - psql
    - -v
    - ON_ERROR_STOP=1
    - -c
    - >-
      select current_user,
      current_setting('search_path') as search_path,
      (select count(*) from information_schema.tables where table_schema = 'mlflow') as mlflow_tables,
      has_table_privilege(current_user, 'public.trades', 'SELECT') as can_read_trades
    env:
    - {name: PGHOST, value: "$host"}
    - {name: PGPORT, value: "5432"}
    - {name: PGDATABASE, value: postgres}
    - {name: PGSSLMODE, value: require}
    - name: PGUSER
      valueFrom: {secretKeyRef: {name: mlflow-db, key: username}}
    - name: PGPASSWORD
      valueFrom: {secretKeyRef: {name: mlflow-db, key: password}}
YAML

phase=""
for _ in $(seq 1 30); do
  phase=$(kubectl -n mlflow get pod "$pod" -o jsonpath='{.status.phase}')
  if [ "$phase" = Succeeded ] || [ "$phase" = Failed ]; then break; fi
  sleep 3
done
echo "phase: $phase"
kubectl -n mlflow logs "$pod"
kubectl -n mlflow delete pod "$pod" --wait=false >/dev/null
[ "$phase" = Succeeded ]
