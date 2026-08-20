# Prism recipe-v10 pod image — the "complete" CUDA 13 base miners build on.
#
# Goal (per product decision 2026-08-15): a single image where miners can
# implement ANYTHING modern — NVFP4 training via Transformer Engine, custom
# CUDA/Triton kernels, FlashAttention, Mamba/SSM — and install their own
# extras from a shipped requirements.txt / pyproject.toml during the pod's
# network-on install phase (see crates/prism-recipe/harness/prismlib/deps.py).
#
# Built in GHCR and mirrored to registry.digitalocean.com/basecrawl/prism-pod
# for Lium. Runtime references are digest-only; the human tag is discovery
# metadata, never a deploy pin.
#
# Base: NVIDIA PyTorch NGC (CUDA 13, cuDNN, NCCL, build toolchain, and a
# CUDA-matched torch already present). We add Transformer Engine + common
# accelerators so a submission needs zero installs for the default path, yet
# can still `pip install` extras with build isolation (nvcc is present).
ARG CUDA_TORCH_BASE=nvcr.io/nvidia/pytorch:26.01-py3@sha256:38ed2ecb2c16d10677006d73fb0a150855d6ec81db8fc66e800b5ae92741007e
FROM ${CUDA_TORCH_BASE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_BREAK_SYSTEM_PACKAGES=1

# sshd is required by the Lium exec path (harness streamed over SSH). The
# image uses CMD, not ENTRYPOINT, so Lium can replace it with the template
# bootstrap that installs USER_PUBLIC_KEY and signals container readiness.
# iproute2 provides `ip`, which the harness uses to bring `lo` UP inside the
# `unshare --net` train/eval namespace — required for single-node multi-GPU
# rendezvous (torch.distributed env:// on 127.0.0.1). A fresh netns still has
# no route off-host, so isolation is unchanged.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      openssh-server git build-essential ninja-build ca-certificates iproute2 \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /run/sshd \
 && ssh-keygen -A

# Harness eval deps (kept identical to the SSH installer so behavior matches
# when the image is warm) + Transformer Engine for NVFP4 training + the
# accelerators most linear/looped architectures want. Build tooling stays in
# the image so a miner's `pip install flash-attn` / `mamba-ssm` can compile.
#
# TE 2.15 is the newest release publishing a CUDA-13 wheel for NGC 26.01's
# Torch/Python ABI, and exposes `NVFP4BlockScaling` for Blackwell / RTX 5090.
# Clear NGC's package constraint only for this explicit, compatible TE pin.
# `--no-build-isolation` links the extension against the image torch.
# Miners may re-pin via requirements.txt; install_deps is resubmittable.
ARG TE_VERSION=2.15.0
RUN pip install --no-cache-dir \
      "transformers==4.44.2" "datasets==3.0.2" "pyarrow==17.0.0" \
 && PIP_CONSTRAINT= pip install --no-cache-dir --no-build-isolation \
      "transformer-engine[pytorch]==${TE_VERSION}" \
 && pip install --no-cache-dir einops

# NVFP4 sanity marker. Docker's CPU-only builder has no driver-mounted
# `libcuda.so.1`, so importing TE's core extension here would reject a valid
# image. Verify the compiled CUDA major, exact wheel, and recipe source
# statically; the billable GPU smoke performs the real extension import.
RUN python -c "\
import importlib.metadata as md, pathlib, torch; \
assert torch.version.cuda and torch.version.cuda.split('.')[0] == '13', torch.version.cuda; \
d = md.distribution('transformer-engine'); \
assert d.version == '${TE_VERSION}', d.version; \
r = pathlib.Path(d.locate_file('transformer_engine/common/recipe')); \
assert any('NVFP4BlockScaling' in p.read_text(errors='ignore') for p in r.rglob('*.py')); \
print('CUDA', torch.version.cuda, 'TE wheel', d.version, 'NVFP4 source OK')"

COPY entrypoint.sh /usr/local/bin/prism-pod-entrypoint
RUN chmod 0555 /usr/local/bin/prism-pod-entrypoint

EXPOSE 22
CMD ["/usr/local/bin/prism-pod-entrypoint"]

LABEL org.baseintelligence.prism.recipe="v10" \
      org.baseintelligence.prism.miner_install="true" \
      org.baseintelligence.prism.cuda="13" \
      org.opencontainers.image.source="https://github.com/BaseIntelligence/base"
