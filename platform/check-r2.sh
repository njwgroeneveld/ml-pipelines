#!/bin/bash
# Look into the R2 bucket from a temporary pod that uses the r2-credentials secret of namespace ml.
#     bash check-r2.sh            list everything under s3://ml-data/
#     bash check-r2.sh <prefix>   list only under s3://ml-data/<prefix>
#     bash check-r2.sh write      write, list and delete s3://ml-data/_check/hello.txt
set -euo pipefail
arg="${1:-}"
pod=check-r2

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
      r2="--endpoint-url \$S3_ENDPOINT --region auto"
      if [ "$mode" = write ]; then
        echo hello > /tmp/hello.txt
        aws s3 cp /tmp/hello.txt s3://ml-data/_check/hello.txt \$r2
        aws s3 ls s3://ml-data/_check/ \$r2
        aws s3 rm s3://ml-data/_check/hello.txt \$r2
      else
        aws s3 ls s3://ml-data/$prefix --recursive \$r2
      fi
    envFrom:
    - secretRef: {name: r2-credentials}
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
