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
# validator-bin/dcap turns on real Intel DCAP quote verification; without it the
# validator parks every attestation instead of verifying it.
RUN cargo build --release \
--features validator-bin/dcap \
-p validator-bin \
-p gateway-bin \
-p updater-bin \
      -p agent-challenge-bin \
      -p hypertraining-challenge-bin \
      -p prism-challenge-bin \
-p agent-runner-bin \
&& strip target/release/validator \
target/release/gateway \
target/release/updater \
              target/release/agent-challenge \
              target/release/hypertraining-challenge \
              target/release/prism-challenge \
target/release/agent-runner \
&& mkdir -p /out \
&& cp target/release/validator \
target/release/gateway \
target/release/updater \
          target/release/agent-challenge \
          target/release/hypertraining-challenge \
          target/release/prism-challenge \
target/release/agent-runner /out/

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
target/release/agent-challenge \
     target/release/hypertraining-challenge \
     target/release/prism-challenge \
target/release/agent-runner \
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
    && mkdir -p /var/lib/base/pack-source /var/lib/base/packs \
    && chown -R base:base /var/lib/base
# Docker seeds a named volume from the image directory it covers, ownership
# included. It only does that when the directory exists: mounting over a path
# the image never created yields a root-owned empty volume the nonroot user
# cannot write. agent-challenge mounts pack-source, the miner CVM mounts packs,
# so both have to be here.
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

FROM runtime-base AS agent-challenge
COPY --from=artifacts --chown=base:base /out/agent-challenge /usr/local/bin/agent-challenge
ENTRYPOINT ["/usr/local/bin/agent-challenge"]

FROM runtime-base AS hypertraining-challenge
COPY --from=artifacts --chown=base:base /out/hypertraining-challenge /usr/local/bin/hypertraining-challenge
ENTRYPOINT ["/usr/local/bin/hypertraining-challenge"]

# Miner runner binary (agent-runner) published as image target base-agent.
FROM runtime-base AS prism-challenge
COPY --from=artifacts --chown=base:base /out/prism-challenge /usr/local/bin/prism-challenge
ENTRYPOINT ["/usr/local/bin/prism-challenge"]

# Miner runner binary (agent-runner) published as image target base-agent.
FROM runtime-base AS base-agent
COPY --from=artifacts --chown=base:base /out/agent-runner /usr/local/bin/agent-runner
ENTRYPOINT ["/usr/local/bin/agent-runner"]
