# syntax=docker/dockerfile:1
#
# Multi-stage build for the Prism challenge with two independently buildable
# targets (mirrors the agent-challenge pattern):
#
#   docker build --target service   -t prism-svc  .   # uvicorn API on :8080
#   docker build --target evaluator -t prism-eval .   # CUDA cu128 torchrun runner
#
# A plain `docker build .` (no --target) yields the `service` image, preserving
# the previous single-image consumer (uvicorn app on port 8080).
#
# The Base SDK is resolved from the immutable, hash-pinned wheel declared in
# pyproject.toml. No repository checkout is required.

############################################################
# evaluator target — CUDA-capable image (cu128 series) that
# runs the torchrun runner. Matches the proven local image
# prism-evaluator:smoke-local-cu128-nonroot (cu128, non-root).
############################################################
ARG CUDA_BASE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
FROM ${CUDA_BASE} AS evaluator

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# torch CUDA channel: cu128 wheels carry their own CUDA 12.8 runtime libs.
ARG TORCH_CUDA_CHANNEL=cu128

WORKDIR /workspace

# python3.12 ships with ubuntu24.04.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Isolated venv (ubuntu24.04 python is externally-managed).
RUN python3 -m venv "$VIRTUAL_ENV"

COPY pyproject.toml ./
COPY src ./src

# Install the CUDA-enabled torch from the cu128 channel FIRST, so the package
# install below resolves `torch>=2.3` against the GPU build (not the CPU wheel),
# then install the package (brings numpy, base and runner deps).
# sentencepiece + tiktoken back the llama / gpt2 reference tokenizers (also pinned in
# pyproject.toml, installed here explicitly so the eval image always carries them).
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/${TORCH_CUDA_CHANNEL} \
    && pip install --no-cache-dir . \
    && pip install --no-cache-dir sentencepiece tiktoken

# Bake the OFFLINE reference tokenizers into the image at a fixed path (network is available
# at build time; the eval container runs network=none). gpt2 -> tiktoken BPE cache blobs read
# back via TIKTOKEN_CACHE_DIR; llama -> a non-gated sentencepiece .model. See
# evaluator/reference_tokenizers.py and architecture.md sections 3 + 9.
ENV PRISM_REFERENCE_TOKENIZER_DIR=/opt/reference-tokenizers \
    TIKTOKEN_CACHE_DIR=/opt/reference-tokenizers/gpt2
RUN python -m prism_challenge.evaluator.reference_tokenizers \
        --output-dir "$PRISM_REFERENCE_TOKENIZER_DIR"

# Enforce Hugging Face offline mode for the eval container so no reference tokenizer / dataset
# load can reach the network (VAL-DATA-012).
ENV HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Non-root runtime user.
RUN useradd --create-home --shell /usr/sbin/nologin prism \
    && mkdir -p /workspace /artifacts \
    && chown -R prism:prism /workspace /artifacts /opt/venv /opt/reference-tokenizers

USER prism
ENV HOME=/home/prism

# The runner.py + payload.json are mounted into /workspace at run time and driven
# with: torchrun --standalone --nnodes=1 --nproc-per-node=N \
#                 /workspace/runner.py /workspace/payload.json
# (see src/prism_challenge/evaluator/container.py::_runner_launch_command).
CMD ["python", "-c", "import torch; print('prism-evaluator ready: torch', torch.__version__, 'cuda', torch.version.cuda)"]


############################################################
# service target — uvicorn API on :8080 (non-root prism).
# This is the final stage, so `docker build .` (no --target)
# reproduces the previous single-image service behavior.
############################################################
FROM python:3.12-slim AS service

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# docker-cli: the service shells out to the docker broker.
RUN apt-get update \
    && apt-get install -y --no-install-recommends docker-cli \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src

# Security: the service performs miner sr25519 signature verification via bittensor
# (auth.verify_hotkey_signature). Without the bittensor extra every signed submission
# is rejected, locking out all miners; install it so production verifies real signatures.
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
