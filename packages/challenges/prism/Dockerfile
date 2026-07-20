# syntax=docker/dockerfile:1
#
# Prism challenge images (monorepo-aware).
#
# Build from Base monorepo ROOT with BuildKit named context `monorepo` (repo root)
# and primary context = this package directory:
#
#   docker buildx build \
#     -f packages/challenges/prism/Dockerfile \
#     --build-context monorepo=. \
#     --target service \
#     -t ghcr.io/baseintelligence/prism:local \
#     packages/challenges/prism
#
#   docker buildx build \
#     -f packages/challenges/prism/Dockerfile \
#     --build-context monorepo=. \
#     --target evaluator \
#     -t ghcr.io/baseintelligence/prism-evaluator:local \
#     packages/challenges/prism
#
# GHCR names are INVARIANT (do not rename):
#   ghcr.io/baseintelligence/prism
#   ghcr.io/baseintelligence/prism-evaluator
#
# OCI source label points at the Base monorepo (not the archived standalone remote).
#
# Multi-stage targets (mirrors pre-monorepo standalone layout):
#   service   — uvicorn API on :8080 (default final stage)
#   evaluator — CUDA cu128 torchrun runner

############################################################
# evaluator target — CUDA-capable image (cu128 series)
############################################################
ARG CUDA_BASE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
FROM ${CUDA_BASE} AS evaluator

LABEL org.opencontainers.image.source="https://github.com/BaseIntelligence/base" \
      org.opencontainers.image.title="prism-evaluator" \
      org.opencontainers.image.description="Prism CUDA evaluator image (Base monorepo)"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

ARG TORCH_CUDA_CHANNEL=cu128

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "$VIRTUAL_ENV"

# Workspace Base first (path dep; not release wheel / floating git).
COPY --from=monorepo pyproject.toml README.md /tmp/base-src/
COPY --from=monorepo src/base /tmp/base-src/src/base
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir /tmp/base-src \
    && rm -rf /tmp/base-src

# Challenge package from primary context (packages/challenges/prism).
COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/${TORCH_CUDA_CHANNEL} \
    && pip install --no-cache-dir . \
    && pip install --no-cache-dir sentencepiece tiktoken

ENV PRISM_REFERENCE_TOKENIZER_DIR=/opt/reference-tokenizers \
    TIKTOKEN_CACHE_DIR=/opt/reference-tokenizers/tiktoken-cache
RUN mkdir -p "$TIKTOKEN_CACHE_DIR" \
    && python -m prism_challenge.evaluator.reference_tokenizers \
        --output-dir "$PRISM_REFERENCE_TOKENIZER_DIR"

ENV HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN useradd --create-home --shell /usr/sbin/nologin prism \
    && mkdir -p /workspace /artifacts \
    && chown -R prism:prism /workspace /artifacts /opt/venv /opt/reference-tokenizers

USER prism
ENV HOME=/home/prism

CMD ["python", "-c", "import torch; print('prism-evaluator ready: torch', torch.__version__, 'cuda', torch.version.cuda)"]


############################################################
# service target — uvicorn API on :8080 (default final stage)
############################################################
FROM python:3.12-slim AS service

LABEL org.opencontainers.image.source="https://github.com/BaseIntelligence/base" \
      org.opencontainers.image.title="prism" \
      org.opencontainers.image.description="Prism challenge service image (Base monorepo)"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends docker-cli \
    && rm -rf /var/lib/apt/lists/*

# Workspace Base first.
COPY --from=monorepo pyproject.toml README.md /tmp/base-src/
COPY --from=monorepo src/base /tmp/base-src/src/base
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir /tmp/base-src \
    && rm -rf /tmp/base-src

COPY pyproject.toml README.md ./
COPY src ./src

# bittensor extra: miner sr25519 signature verification (auth.verify_hotkey_signature).
RUN pip install --no-cache-dir ".[bittensor]"

RUN useradd --create-home --shell /usr/sbin/nologin prism \
    && mkdir -p /data/tmp \
    && chown -R prism:prism /app /data

USER prism
ENV HOME=/home/prism \
    TMPDIR=/data/tmp \
    TEMP=/data/tmp \
    TMP=/data/tmp

EXPOSE 8080

CMD ["uvicorn", "prism_challenge.app:app", "--host", "0.0.0.0", "--port", "8080"]
