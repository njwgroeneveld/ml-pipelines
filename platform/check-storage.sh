#!/bin/bash
# Look into the ml-data bucket (Supabase Storage, S3 API) from a temporary pod that uses the
# s3-credentials secret of namespace ml. It uses s3fs with default settings, the same library
# and configuration as mlkit.io in the pipelines.
#     bash check-storage.sh            list everything under ml-data/
#     bash check-storage.sh <prefix>   list only under ml-data/<prefix>
#     bash check-storage.sh write      write, read back and delete ml-data/_check/hello.txt
# (amazon/aws-cli is not used: its Amazon Linux base needs ARMv8.2, which the Pi 4 lacks.)
set -euo pipefail
arg="${1:-}"
pod=check-storage

kubectl -n ml delete pod "$pod" --ignore-not-found >/dev/null
kubectl -n ml apply -f - >/dev/null <<YAML
apiVersion: v1
kind: Pod
metadata: {name: $pod}
spec:
  restartPolicy: Never
  containers:
  - name: s3
    image: python:3.13-slim
    env:
    - {name: ARG, value: "$arg"}
    - {name: PIP_ROOT_USER_ACTION, value: ignore}
    - {name: PIP_DISABLE_PIP_VERSION_CHECK, value: "1"}
    envFrom:
    - secretRef: {name: s3-credentials}
    command: [sh, -c]
    args:
    - |
      pip install -q s3fs==2026.9.0 && python - <<'PY'
      import os
      import s3fs
      fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": os.environ["S3_ENDPOINT"]})
      arg = os.environ["ARG"]
      if arg == "write":
          path = "ml-data/_check/hello.txt"
          fs.pipe(path, b"hello")
          print("wrote", path)
          print("read back:", fs.cat(path))
          print("listing:", fs.ls("ml-data/_check"))
          fs.rm_file(path)   # single DeleteObject; Supabase rejects the bulk DeleteObjects
          print("deleted", path, "-> exists:", fs.exists(path))
      else:
          files = fs.find("ml-data/" + arg, detail=True)
          for name, info in sorted(files.items()):
              print(f"{info['size']:>12}  {name}")
          print(f"{len(files)} files, {sum(i['size'] for i in files.values()) / 1e6:.1f} MB")
      PY
YAML

phase=""
for _ in $(seq 1 60); do
  phase=$(kubectl -n ml get pod "$pod" -o jsonpath='{.status.phase}')
  if [ "$phase" = Succeeded ] || [ "$phase" = Failed ]; then break; fi
  sleep 3
done
echo "phase: $phase"
kubectl -n ml logs "$pod"
kubectl -n ml delete pod "$pod" --wait=false >/dev/null
[ "$phase" = Succeeded ]
