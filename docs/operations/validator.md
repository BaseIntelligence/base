# Validator Operations

![Platform Banner](../../assets/banner.jpg)

## Scope

Run these commands from the repository root. They are the local validation surfaces used for the corrected Python `platform-network` work. If Docker, Helm, kubeconform, kind, kubectl, or a Python tool is missing, record the blocker in evidence and don't mark that surface as tested.

## Python Validation

```bash
uv sync --extra dev --extra master
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest --cov=platform_network --cov-report=term-missing --cov-fail-under=80
```

Known baseline notes belong in evidence. Current Task 12 evidence records these Python quality gates as passing without changing the documented gates: Ruff check, Ruff format check, mypy, and full coverage. Historical Task 11 evidence recorded Ruff format and mypy blockers, but those blockers are resolved in the current validation state.

## Docker Compose Validation

Validate the base stack, the development stack, and the local or staging Watchtower overlay:

```bash
docker compose -f docker/compose.yml config --quiet
docker compose -f docker/compose.dev.yml config --quiet
docker compose -f docker/compose.yml -f docker/compose.watchtower.yml config --quiet
```

Clean up local Compose validation resources with:

```bash
docker compose -f docker/compose.yml -f docker/compose.watchtower.yml down --remove-orphans
```

The Compose files intentionally expose local development defaults. Production safety comes from the production policy checks, production Helm values, external PostgreSQL, and digest-pinned images.

## Local and Staging Watchtower Overlay

`docker/compose.watchtower.yml` is an explicit local and staging overlay for updating the Docker Compose control plane. It uses the maintained `nickfedor/watchtower:1.17.1` image for Docker 29 API compatibility and runs Watchtower with `--label-enable`, so containers are ignored unless they carry `com.centurylinklabs.watchtower.enable=true`.

The only Compose services opted in are:

- `master-admin`
- `master-proxy`
- `platform-docker-broker`
- `validator`
- `gpu-agent`

Challenge containers, broker-created job containers, database services, and Kubernetes manifests must not receive Watchtower labels. Production Kubernetes uses Helm and Kubernetes rollout controls instead of Watchtower.

Before using the overlay in local or staging environments, render it with the Compose command above and confirm the watched services are healthy after each update. Keep a rollback image tag available. Watchtower can replace containers, but it doesn't prove application health or perform production rollback orchestration.

## Docker Socket Risk

The local Compose control plane mounts `/var/run/docker.sock` for services that need to create local Docker containers or inspect/update local Compose services. The host Docker socket is root-equivalent host access. Treat these socket mounts as local control-plane risk, not production isolation.

The Compose labels `platform.security.docker-socket` and `platform.security.docker-socket-risk` must stay on socket-owning services. Broker-created challenge containers and Kubernetes jobs must not receive the host Docker socket.

## Helm and Kubernetes Validation

Validate the chart with default values and the production policy fixture:

```bash
helm lint deploy/helm/platform
helm template platform deploy/helm/platform > /tmp/platform-default.yaml
kubeconform -strict -summary /tmp/platform-default.yaml
helm template platform deploy/helm/platform -f deploy/helm/platform/values.production.example.yaml > /tmp/platform-production.yaml
kubeconform -strict -summary /tmp/platform-production.yaml
```

Run kind-backed server dry-run validation when Docker and kind are available:

```bash
kind delete cluster --name platform-validation
kind create cluster --name platform-validation
kind get kubeconfig --name platform-validation > /tmp/platform-validation-kubeconfig
KUBECONFIG=/tmp/platform-validation-kubeconfig kubectl apply --dry-run=server -f /tmp/platform-default.yaml
KUBECONFIG=/tmp/platform-validation-kubeconfig kubectl apply --dry-run=server -f /tmp/platform-production.yaml
kind delete cluster --name platform-validation
```

Remove `/tmp/platform-validation-kubeconfig` after use if it contains live cluster access. Never commit kubeconfigs or paste them into evidence.

## Database, Image, and TLS Policy

Use different policy expectations for local validation and production operations:

- Local development and tests may use the default SQLite database URL and local mutable images. These defaults are intended for fast iteration only.
- Production and Kubernetes deployments must provide an external PostgreSQL database secret or URL before the control plane starts. Do not use SQLite for production or Kubernetes master state.
- Production images must be pinned with a semver tag and `sha256` digest. Do not deploy `latest`, untagged images, or images without a digest in production.
- Production remote GPU servers and Kubernetes targets must keep `verify_tls=true`. Disable TLS verification only for local test endpoints that are not part of production.
- Production Kubernetes agent targets must use HTTPS and `verify_tls=true`. Multi-server routing should trust only enabled, healthy, non-draining targets with available GPU capacity, and it should clear stale persisted assignments when those checks fail.

For Helm, render production values with `deploy/helm/platform/values.production.example.yaml` and verify failures for unsafe overrides such as `image.tag=latest`, missing `image.digest`, missing database secret references, or target `verify_tls=false`.

## Kubernetes PID and Swap Policy

Kubernetes jobs and challenge workloads map CPU and memory to PodSpec requests and limits. Docker-only `pids_limit`, `memory_swap`, and custom Docker network modes are rejected for Kubernetes requests because Kubernetes won't enforce those fields through this PodSpec path. If a production cluster needs PID or swap ceilings, document the cluster or admission policy that enforces them.

## Broker Archive and Cleanup Checks

Broker archive input is untrusted. Validation evidence for broker changes should show that Docker and Kubernetes paths reject absolute paths, parent traversal, links, device members, malformed images, and unsafe mount sources before creating runtime resources.

Kubernetes broker cleanup evidence should cover deletion attempts for the Job, NetworkPolicy, and mount Secret on success, failure, timeout, apply-error, wait-error, and log-error paths. Keep archive payloads and bearer credentials out of evidence logs.

## Evidence Expectations

Save validation output in a local, gitignored evidence directory with task-scoped names. Evidence should include:

- command logs for Python, Compose, Helm, kubeconform, kind, and kubectl dry-run surfaces that were actually executed;
- policy guard output showing Watchtower scope, Docker socket risk wording, production PostgreSQL, semver plus digest images, `verify_tls=true`, Kubernetes PID boundary, multi-server target trust, and cleanup commands are documented;
- explicit limitations for unavailable tools or historical blockers, including the resolved Task 11 Ruff format and mypy blockers only when labeled historical or resolved, plus current Task 12 evidence showing Ruff check, Ruff format check, mypy, and full coverage passing;
- a redaction or grep check showing evidence does not contain bearer tokens, private keys, kubeconfigs, credentialed database URLs, private registry credentials, or Docker registry auth.

## Master Deployment Checklist

1. Configure `config/master.example.yaml` or provide environment overrides.
2. Provide an admin token file.
3. Run Alembic migrations.
4. Start the master API.
5. Start the proxy API.
6. Register and activate challenge images.
7. Monitor logs, Sentry, and OpenTelemetry.
