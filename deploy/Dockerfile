# syntax=docker/dockerfile:1.7
# Multi-stage Rust build for gbase service binaries (validator | gateway | updater).
#
# Source build (default, CI/prod):
#   docker build -f deploy/Dockerfile --target validator -t gbase-validator:0.1.0 .
#
# Fast local path (host already ran `cargo build --release -p …`):
#   docker build -f deploy/Dockerfile --target validator \
#     --build-arg BUILD_FROM=prebuilt -t gbase-validator:0.1.0 .
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
# Source builder — compiles all three service bins once
# -----------------------------------------------------------------------------
FROM rust-base AS source-builder
COPY Cargo.toml Cargo.lock rust-toolchain.toml clippy.toml ./
COPY crates ./crates
COPY bins ./bins
COPY xtask ./xtask
# Keep workspace metadata members that Docker context needs
COPY config ./config
RUN cargo build --release \
      -p gbase-validator-bin \
      -p gbase-gateway-bin \
      -p gbase-updater-bin \
    && strip target/release/gbase-validator \
              target/release/gbase-gateway \
              target/release/gbase-updater \
    && mkdir -p /out \
    && cp target/release/gbase-validator \
          target/release/gbase-gateway \
          target/release/gbase-updater /out/

# -----------------------------------------------------------------------------
# Prebuilt path — copy host target/release artifacts (local verify only)
# -----------------------------------------------------------------------------
FROM ${DEBIAN_IMAGE} AS prebuilt-builder
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /out
COPY target/release/gbase-validator \
     target/release/gbase-gateway \
     target/release/gbase-updater \
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
    && apt-get install -y --no-install-recommends ca-certificates libssl3 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 65532 gbase \
    && useradd --system --uid 65532 --gid gbase --home-dir /home/gbase --create-home gbase
USER gbase
WORKDIR /home/gbase

FROM runtime-base AS validator
COPY --from=artifacts --chown=gbase:gbase /out/gbase-validator /usr/local/bin/gbase-validator
ENTRYPOINT ["/usr/local/bin/gbase-validator"]

FROM runtime-base AS gateway
COPY --from=artifacts --chown=gbase:gbase /out/gbase-gateway /usr/local/bin/gbase-gateway
ENTRYPOINT ["/usr/local/bin/gbase-gateway"]

FROM runtime-base AS updater
COPY --from=artifacts --chown=gbase:gbase /out/gbase-updater /usr/local/bin/gbase-updater
ENTRYPOINT ["/usr/local/bin/gbase-updater"]
