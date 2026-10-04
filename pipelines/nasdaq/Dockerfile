# One image for every step of the Nasdaq pipeline; Argo picks the script per step.
# Build context is the repo root, because the image needs shared/mlkit:
#     docker build -f pipelines/nasdaq/Dockerfile --build-arg IMAGE_TAG=0.2.0 --build-arg GIT_SHA=$(git rev-parse HEAD) .
# Debian-based on purpose: the Pi 4 node (ARMv8.0) cannot run Amazon Linux 2023 images.
FROM python:3.13-slim
LABEL org.opencontainers.image.source=https://github.com/njwgroeneveld/ml-pipelines

# GIT_PYTHON_REFRESH: there is no git in the image; without it MLflow warns about that on every run.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GIT_PYTHON_REFRESH=quiet \
    MLFLOW_DISABLE_AGENT_HINT=1

# Dependencies first: pip only reruns when shared/mlkit or requirements.txt change, not for a script.
COPY shared/mlkit /tmp/mlkit
COPY pipelines/nasdaq/requirements.txt /tmp/requirements.txt
RUN pip install /tmp/mlkit -r /tmp/requirements.txt && rm -rf /tmp/mlkit /tmp/requirements.txt

# A numeric user, so a pod with runAsNonRoot can verify it is not root.
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY pipelines/nasdaq/*.py ./
COPY pipelines/nasdaq/datasets ./datasets

# Fail the build, not the first workflow, if a step cannot even be imported.
RUN python -c "import fetch, register, select_winner, start, train, yfinance"

# Provenance: mlkit.tracking.provenance_tags logs these on every MLflow run.
ARG IMAGE_TAG=dev
ARG GIT_SHA=unknown
ENV IMAGE_TAG=$IMAGE_TAG \
    GIT_SHA=$GIT_SHA

USER 10001
