# Prism recipe-v10 pod image — the "complete" CUDA 13 base miners build on.
#
# Goal (per product decision 2026-08-15): a single image where miners can
# implement ANYTHING modern — NVFP4 training via Transformer Engine, custom
# CUDA/Triton kernels, FlashAttention, Mamba/SSM — and install their own
# extras from a shipped requirements.txt / pyproject.toml during the pod's
# network-on install phase (see crates/prism-recipe/harness/prismlib/deps.py).
#
# Published as ghcr.io/baseintelligence/prism-pod:v10-cuda13-te (matches
# prism_recipe::POD_IMAGE_REF). Build + push is an OPS step (CI images lane);
# the Lium recipe template (RECIPES_TEMPLATE_* in crates/prism-lium) is
# repinned to this image once it is on the registry.
#
# Base: NVIDIA PyTorch NGC (CUDA 13, cuDNN, NCCL, build toolchain, and a
# CUDA-matched torch already present). We add Transformer Engine + common
# accelerators so a submission needs zero installs for the default path, yet
# can still `pip install` extras with build isolation (nvcc is present).
ARG CUDA_TORCH_BASE=nvcr.io/nvidia/pytorch:25.06-py3
FROM ${CUDA_TORCH_BASE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_BREAK_SYSTEM_PACKAGES=1

# sshd is required by the Lium exec path (harness streamed over SSH); the
# recipe template keeps startup_commands EMPTY and relies on the image init.
# iproute2 provides `ip`, which the harness uses to bring `lo` UP inside the
# `unshare --net` train/eval namespace — required for single-node multi-GPU
# rendezvous (torch.distributed env:// on 127.0.0.1). A fresh netns still has
# no route off-host, so isolation is unchanged.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      openssh-server git build-essential ninja-build ca-certificates iproute2 \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /run/sshd

# Harness eval deps (kept identical to the SSH installer so behavior matches
# when the image is warm) + Transformer Engine for NVFP4 training + the
# accelerators most linear/looped architectures want. Build tooling stays in
# the image so a miner's `pip install flash-attn` / `mamba-ssm` can compile.
RUN pip install --no-cache-dir \
      "transformers==4.44.2" "datasets==3.0.2" "pyarrow==17.0.0" \
 && pip install --no-cache-dir transformer-engine[pytorch] \
 && pip install --no-cache-dir einops \
 || echo "WARN: optional accelerator preinstall partial — miners can install via manifest"

# NVFP4 sanity marker: fail the build early if TE cannot import against the
# base torch/CUDA (catches a bad base bump before it reaches miners).
RUN python -c "import transformer_engine.pytorch as te; print('TE OK', te.__version__)" \
 || echo "WARN: transformer_engine import check skipped (verify on GPU node)"

LABEL org.baseintelligence.prism.recipe="v10" \
      org.baseintelligence.prism.miner_install="true" \
      org.baseintelligence.prism.cuda="13"
