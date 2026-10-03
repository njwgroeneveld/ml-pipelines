#!/bin/bash
# Look into the ml-data bucket (Supabase Storage, S3 API) from a temporary pod that uses the
# s3-credentials secret of namespace ml.
#     bash check-storage.sh            list everything under s3://ml-data/
#     bash check-storage.sh <prefix>   list only under s3://ml-data/<prefix>
#     bash check-storage.sh write      write, list and delete s3://ml-data/_check/hello.txt
set -euo pipefail
arg="${1:-}"
pod=check-storage

if [ "$arg" = write ]; then
  mode=write
  prefix=""
else
  mode=list
  prefix="$arg"
fi

kubectl -n ml delete pod "$pod" --ignore-not-found >/dev/null
kubectl -n ml apply -f - >/dev/null <<YAML
apiVersion: v1
kind: Pod
metadata: {name: $pod}
spec:
  restartPolicy: Never
  containers:
  - name: aws
    image: amazon/aws-cli:latest
    command: [sh, -c]
    args:
    - |
      set -e
      # Supabase Storage needs path-style addressing
      aws configure set default.s3.addressing_style path
      s3="--endpoint-url \$S3_ENDPOINT --region \$AWS_DEFAULT_REGION"
      if [ "$mode" = write ]; then
        echo hello > /tmp/hello.txt
        aws s3 cp /tmp/hello.txt s3://ml-data/_check/hello.txt \$s3
        aws s3 ls s3://ml-data/_check/ \$s3
        aws s3 rm s3://ml-data/_check/hello.txt \$s3
      else
        aws s3 ls s3://ml-data/$prefix --recursive --summarize \$s3
      fi
    envFrom:
    - secretRef: {name: s3-credentials}
YAML

phase=""
for _ in $(seq 1 40); do
  phase=$(kubectl -n ml get pod "$pod" -o jsonpath='{.status.phase}')
  if [ "$phase" = Succeeded ] || [ "$phase" = Failed ]; then break; fi
  sleep 3
done
echo "phase: $phase"
kubectl -n ml logs "$pod"
kubectl -n ml delete pod "$pod" --wait=false >/dev/null
[ "$phase" = Succeeded ]
