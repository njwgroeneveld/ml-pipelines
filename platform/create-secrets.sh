#!/bin/bash
# Create the Kubernetes secrets of the ML platform. Run this yourself on pi5:
#     bash ~/ml-pipelines/platform/create-secrets.sh
# It asks for every value; passwords are not echoed and nothing ends up in shell history
# or on a command line (values reach kubectl through a file descriptor, not as arguments).
set -euo pipefail

read -rp  "Supabase project ref (from the pooler user postgres.<ref>): " PROJECT_REF
read -rp  "Supabase project region (e.g. eu-central-1): " REGION
read -rsp "Password you gave mlflow_user: " DB_PASSWORD; echo
read -rp  "Storage S3 access key id: " S3_KEY_ID
read -rsp "Storage S3 secret access key: " S3_SECRET; echo

endpoint="https://${PROJECT_REF}.storage.supabase.co/storage/v1/s3"

# Supabase login for the MLflow server (pooler user = <role>.<project ref>)
kubectl -n mlflow create secret generic mlflow-db \
  --from-file=username=<(printf %s "mlflow_user.${PROJECT_REF}") \
  --from-file=password=<(printf %s "$DB_PASSWORD") \
  --dry-run=client -o yaml | kubectl apply -f -

# Storage keys for the MLflow server (models)
kubectl -n mlflow create secret generic mlflow-s3 \
  --from-file=AWS_ACCESS_KEY_ID=<(printf %s "$S3_KEY_ID") \
  --from-file=AWS_SECRET_ACCESS_KEY=<(printf %s "$S3_SECRET") \
  --dry-run=client -o yaml | kubectl apply -f -

# Storage keys, endpoint and region for the pipeline pods (snapshots, features)
kubectl -n ml create secret generic s3-credentials \
  --from-file=AWS_ACCESS_KEY_ID=<(printf %s "$S3_KEY_ID") \
  --from-file=AWS_SECRET_ACCESS_KEY=<(printf %s "$S3_SECRET") \
  --from-file=AWS_DEFAULT_REGION=<(printf %s "$REGION") \
  --from-file=S3_ENDPOINT=<(printf %s "$endpoint") \
  --dry-run=client -o yaml | kubectl apply -f -

unset DB_PASSWORD S3_SECRET S3_KEY_ID PROJECT_REF REGION
echo "Secrets created (names only):"
kubectl -n mlflow get secret mlflow-db mlflow-s3
kubectl -n ml get secret s3-credentials
