#!/bin/bash
# Create the Kubernetes secrets of the ML platform. Run this yourself on pi5:
#     bash ~/ml-pipelines/platform/create-secrets.sh
# It asks for every value; passwords are not echoed and nothing ends up in shell history
# or on a command line (values reach kubectl through a file descriptor, not as arguments).
set -euo pipefail

read -rp  "Supabase project ref (from the pooler user postgres.<ref>): " PROJECT_REF
read -rsp "Password you gave mlflow_user: " DB_PASSWORD; echo
read -rp  "Cloudflare account id: " ACCOUNT_ID
read -rp  "R2 access key id: " R2_KEY_ID
read -rsp "R2 secret access key: " R2_SECRET; echo

endpoint="https://${ACCOUNT_ID}.r2.cloudflarestorage.com"

# Supabase login for the MLflow server (pooler user = <role>.<project ref>)
kubectl -n mlflow create secret generic mlflow-db \
  --from-file=username=<(printf %s "mlflow_user.${PROJECT_REF}") \
  --from-file=password=<(printf %s "$DB_PASSWORD") \
  --dry-run=client -o yaml | kubectl apply -f -

# R2 keys for the MLflow server (models)
kubectl -n mlflow create secret generic mlflow-r2 \
  --from-file=AWS_ACCESS_KEY_ID=<(printf %s "$R2_KEY_ID") \
  --from-file=AWS_SECRET_ACCESS_KEY=<(printf %s "$R2_SECRET") \
  --dry-run=client -o yaml | kubectl apply -f -

# R2 keys + endpoint for the pipeline pods (snapshots, features)
kubectl -n ml create secret generic r2-credentials \
  --from-file=AWS_ACCESS_KEY_ID=<(printf %s "$R2_KEY_ID") \
  --from-file=AWS_SECRET_ACCESS_KEY=<(printf %s "$R2_SECRET") \
  --from-file=S3_ENDPOINT=<(printf %s "$endpoint") \
  --dry-run=client -o yaml | kubectl apply -f -

unset DB_PASSWORD R2_SECRET R2_KEY_ID PROJECT_REF
echo "Secrets created (names only):"
kubectl -n mlflow get secret mlflow-db mlflow-r2
kubectl -n ml get secret r2-credentials
