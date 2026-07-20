# syntax=docker/dockerfile:1
#
# Agent-challenge images (monorepo-aware).
#
# Build from Base monorepo ROOT with BuildKit named context `monorepo` (repo root)
# and primary context = this package directory:
#
#   docker buildx build \
#     -f packages/challenges/agent-challenge/Dockerfile \
#     --build-context monorepo=. \
#     --target runtime \
#     -t ghcr.io/baseintelligence/agent-challenge:local \
#     packages/challenges/agent-challenge
#
#   docker buildx build \
#     -f packages/challenges/agent-challenge/Dockerfile \
#     --build-context monorepo=. \
#     --target terminal-bench-runner \
#     -t ghcr.io/baseintelligence/agent-challenge-terminal-bench-runner:local \
#     packages/challenges/agent-challenge
#
# GHCR names are INVARIANT (do not rename):
#   ghcr.io/baseintelligence/agent-challenge
#   ghcr.io/baseintelligence/agent-challenge-terminal-bench-runner
#
# OCI source label points at the Base monorepo (not the archived standalone remote).

# ---------------------------------------------------------------------------
# dcap-qvl (Phala dcap-qvl-cli): trustless Intel PCS quote verifier used by
# review + key-release paths (DcapQvlVerifier). Baking the binary into the
# shipping runtime image keeps live AC independent of the ops-only host bind
# /var/lib/base/tools/dcap-qvl (ac-dcap-pcs-ready interim). Pin crate version
# to the verified ops binary family (dcap-qvl-cli 0.5.2); do not invent
# offline trust roots or ship collateral here.
# ---------------------------------------------------------------------------
FROM rust:1-bookworm AS dcap-qvl-builder

ARG DCAP_QVL_CLI_VERSION=0.5.2

RUN cargo install dcap-qvl-cli --version "${DCAP_QVL_CLI_VERSION}" \
    && test -x /usr/local/cargo/bin/dcap-qvl \
    && /usr/local/cargo/bin/dcap-qvl --help >/dev/null \
    && strip /usr/local/cargo/bin/dcap-qvl

FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.source="https://github.com/BaseIntelligence/base" \
      org.opencontainers.image.title="agent-challenge" \
      org.opencontainers.image.description="Agent-challenge service image (Base monorepo)"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends docker.io git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=dcap-qvl-builder /usr/local/cargo/bin/dcap-qvl /usr/local/bin/dcap-qvl
RUN chmod 0755 /usr/local/bin/dcap-qvl \
    && test -x /usr/local/bin/dcap-qvl

# Workspace Base first (path dep; not floating git+base).
COPY --from=monorepo pyproject.toml README.md /tmp/base-src/
COPY --from=monorepo src/base /tmp/base-src/src/base
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir /tmp/base-src \
    && rm -rf /tmp/base-src

COPY pyproject.toml README.md ./
COPY .rules ./.rules
COPY src ./src
# Phala Cloud pre-launch helper is measured into review/eval app-compose.
# Offline compose generators resolve REPO_ROOT=/app (PYTHONPATH=/app/src), so
# the shipping runtime image must package the vendor script at this exact path
# or create_review_session fails with image_package_prelaunch_script_missing.
COPY docker/review/phala_pre_launch.sh /app/docker/review/phala_pre_launch.sh

RUN pip install --no-cache-dir .

RUN useradd --create-home --uid 10001 challenge \
    && mkdir -p /data/agents \
    && chown -R challenge:challenge /app /data

USER 10001:10001

EXPOSE 8000

CMD ["uvicorn", "agent_challenge.app:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.12-slim AS terminal-bench-runner

LABEL org.opencontainers.image.source="https://github.com/BaseIntelligence/base" \
      org.opencontainers.image.title="agent-challenge-terminal-bench-runner" \
      org.opencontainers.image.description="Agent-challenge terminal-bench runner (Base monorepo)"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg git iptables fuse-overlayfs; \
    install -m 0755 -d /etc/apt/keyrings; \
    curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc; \
    chmod a+r /etc/apt/keyrings/docker.asc; \
    . /etc/os-release; \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian ${VERSION_CODENAME} stable" > /etc/apt/sources.list.d/docker.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        docker-ce docker-ce-cli containerd.io docker-compose-plugin docker-buildx-plugin; \
    update-alternatives --set iptables /usr/sbin/iptables-legacy || true; \
    apt-get clean; rm -rf /var/lib/apt/lists/*

COPY --from=monorepo pyproject.toml README.md /tmp/base-src/
COPY --from=monorepo src/base /tmp/base-src/src/base
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir /tmp/base-src \
    && rm -rf /tmp/base-src

COPY pyproject.toml README.md ./
COPY .rules ./.rules
COPY src ./src

RUN pip install --no-cache-dir .

# Pre-bake the toolchain a submitted agent needs so its install resolves fully
# offline. Runner jobs run on an egress-free network, so own_runner installs the
# agent with `--no-index --no-build-isolation` (see runner._own_runner_script).
RUN pip install --no-cache-dir \
        "setuptools>=61" \
        "wheel>=0.40" \
        "hatchling>=1.25" \
        "hatch-vcs>=0.4" \
        "poetry-core>=1.9" \
        "flit-core>=3.9" \
        "pdm-backend>=2.3" \
        "editables>=0.5" \
        "httpx>=0.27.0" \
        "pydantic>=2.0" \
        "tomli>=2.0" \
        "tomli-w>=1.0" \
        "rich>=13.0" \
        "typer>=0.12.0"

# no CMD: the own_runner broker supplies the command at launch
