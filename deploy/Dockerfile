# syntax=docker/dockerfile:1.7
# Multi-stage Rust build for base service binaries
#
# Source build (default, CI/prod):
#   docker build -f deploy/Dockerfile --target validator -t validator:0.1.0 .
#
# Fast local path (host already ran `cargo build --release -p …`):
#   docker build -f deploy/Dockerfile --target validator \
#     --build-arg BUILD_FROM=prebuilt -t validator:0.1.0 .
#
# Digests only for base images — never floating tags.

ARG DEBIAN_IMAGE=debian@sha256:7b140f374b289a7c2befc338f42ebe6441b7ea838a042bbd5acbfca6ec875818
ARG RUST_IMAGE=rust@sha256:e51d0265072d2d9d5d320f6a44dde6b9ef13653b035098febd68cce8fa7c0bc4
# python:3.12.11-slim-bookworm (multi-arch index digest)
ARG PYTHON_IMAGE=python@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7
ARG BUILD_FROM=source

# -----------------------------------------------------------------------------
# Toolchain: workspace pin is Rust 1.96.0 (rust-toolchain.toml)
# -----------------------------------------------------------------------------
FROM ${RUST_IMAGE} AS rust-base
RUN rustup toolchain install 1.96.0 --profile minimal \
    && rustup default 1.96.0 \
    && rustc --version
WORKDIR /src

# -----------------------------------------------------------------------------
# Source builder — compiles all service bins once
# -----------------------------------------------------------------------------
FROM rust-base AS source-builder
COPY Cargo.toml Cargo.lock rust-toolchain.toml clippy.toml ./
COPY crates ./crates
COPY bins ./bins
COPY xtask ./xtask
# Keep workspace metadata members that Docker context needs
COPY config ./config
# design-challenge embeds the published baseline agent as the agentic review
# corpus anchor (include_str! in crates/design-challenge/src/orchestrator.rs).
COPY docs/external-miner/examples/design-baseline ./docs/external-miner/examples/design-baseline
# validator-bin/dcap turns on real Intel DCAP quote verification; without it the
# validator parks every attestation instead of verifying it.
RUN cargo build --release \
      --features validator-bin/dcap \
      -p validator-bin \
      -p gateway-bin \
      -p updater-bin \
      -p prism-challenge-bin \
      -p design-challenge-bin \
      -p design-egress-proxy-bin \
 && cargo build --release -p challenge-review-bin \
 && strip target/release/validator \
           target/release/gateway \
           target/release/updater \
           target/release/prism-challenge \
           target/release/design-challenge \
           target/release/design-egress-proxy \
           target/release/challenge-review \
 && mkdir -p /out \
 && cp target/release/validator \
       target/release/gateway \
       target/release/updater \
       target/release/prism-challenge \
       target/release/design-challenge \
       target/release/design-egress-proxy \
       target/release/challenge-review /out/

# -----------------------------------------------------------------------------
# Prebuilt path — copy host target/release artifacts (local verify only)
# -----------------------------------------------------------------------------
FROM ${DEBIAN_IMAGE} AS prebuilt-builder
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /out
COPY target/release/validator \
     target/release/gateway \
     target/release/updater \
     target/release/prism-challenge \
     target/release/design-challenge \
     target/release/design-egress-proxy \
     target/release/challenge-review \
     /out/

# -----------------------------------------------------------------------------
# Select artifact source
# -----------------------------------------------------------------------------
FROM source-builder AS artifacts-source
FROM prebuilt-builder AS artifacts-prebuilt

FROM artifacts-${BUILD_FROM} AS artifacts

# -----------------------------------------------------------------------------
# Runtime base (non-root)
# -----------------------------------------------------------------------------
FROM ${DEBIAN_IMAGE} AS runtime-base
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libssl3 curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 65532 base \
    && useradd --system --uid 65532 --gid base --home-dir /home/base --create-home base \
    && mkdir -p /var/lib/base \
    && chown -R base:base /var/lib/base
USER base
WORKDIR /home/base

FROM runtime-base AS validator
COPY --from=artifacts --chown=base:base /out/validator /usr/local/bin/validator
# Owner-signed trust root for /v1/attest allowlist (todo 34 staging Verified path).
COPY --chown=base:base config /etc/base/config
ENV BASE_TRUST_ROOT_DIR=/etc/base/config
ENTRYPOINT ["/usr/local/bin/validator"]

FROM runtime-base AS gateway
COPY --from=artifacts --chown=base:base /out/gateway /usr/local/bin/gateway
ENTRYPOINT ["/usr/local/bin/gateway"]

FROM runtime-base AS updater
COPY --from=artifacts --chown=base:base /out/updater /usr/local/bin/updater
ENTRYPOINT ["/usr/local/bin/updater"]

FROM runtime-base AS prism-challenge
# prism-lium drives pod exec over the `ssh` binary; runtime-base has none.
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends openssh-client \
    && rm -rf /var/lib/apt/lists/*
USER base
COPY --from=artifacts --chown=base:base /out/prism-challenge /usr/local/bin/prism-challenge
ENTRYPOINT ["/usr/local/bin/prism-challenge"]

FROM runtime-base AS design-challenge
USER root
# Headless Chromium for full-page design screenshots (DESIGN_CHROME_BIN).
RUN apt-get update \
    && apt-get install -y --no-install-recommends chromium fonts-liberation \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/lib/design/staging \
    && chown -R base:base /var/lib/design
ENV DESIGN_CHROME_BIN=/usr/bin/chromium
USER base
COPY --from=artifacts --chown=base:base /out/design-challenge /usr/local/bin/design-challenge
ENTRYPOINT ["/usr/local/bin/design-challenge"]

FROM runtime-base AS design-egress-proxy
COPY --from=artifacts --chown=base:base /out/design-egress-proxy /usr/local/bin/design-egress-proxy
ENTRYPOINT ["/usr/local/bin/design-egress-proxy"]

# -----------------------------------------------------------------------------
# design-runtime — pinned Python for design-challenge sandboxes (install/run).
# No challenge secrets; sandboxes reach egress only via design-sandbox-egress.
# -----------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS design-runtime
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 65532 base \
    && useradd --system --uid 65532 --gid base --home-dir /home/base --create-home base \
    && mkdir -p /work /out \
    && chown -R base:base /work /out /home/base
USER base
WORKDIR /work
# Harness entry is staged into /work by design-challenge; image only supplies Python.
CMD ["python", "--version"]

# -----------------------------------------------------------------------------
# design-review — containerized anti-cheat review (challenge-agentic + AST).
# design-challenge stages /work (submitted agent + _similar harness, read-only)
# and /out (verdict) per run; the OpenRouter key arrives via container env.
# Python is present so the sandboxed run_command tool can run AST probes.
# -----------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS design-review
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 65532 base \
    && useradd --system --uid 65532 --gid base --home-dir /home/base --create-home base \
    && mkdir -p /work /out \
    && chown -R base:base /work /out /home/base
USER base
WORKDIR /work
COPY --from=artifacts --chown=base:base /out/challenge-review /usr/local/bin/challenge-review
ENTRYPOINT ["/usr/local/bin/challenge-review"]
