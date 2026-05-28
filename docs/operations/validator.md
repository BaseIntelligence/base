# Validator Operations

Run these commands from the repository root. This runbook is only for normal
validator Kubernetes installation and operation.

## Install Or Update

Automatic Kubernetes install:

Kubernetes is not optional for this installer: it applies resources to the
current `kubectl` context. For a single validator VM, prefer `k3s`; use
`minikube` only for local smoke tests. Validation VMs should have at least 2
vCPUs and 8 GB RAM.

```bash
./scripts/install-validator.sh
```

Normal install performs real Kubernetes changes and prompts for the validator
hotkey mnemonic. Without `--database-url` or `PLATFORM_DATABASE_URL`, it creates namespace-scoped managed validator Postgres resources named `platform-validator-postgres` with `postgres:16-alpine`, 10Gi storage, optional `PLATFORM_VALIDATOR_POSTGRES_STORAGE_CLASS`, and a retained data claim. It stores the generated URL only in the configured URL Secret, never in ConfigMaps, CronJobs, stdout, or stderr. Supplying `--database-url` or `PLATFORM_DATABASE_URL` skips those managed DB resources and uses the provided external URL Secret. It creates `platform-validator-helm-upgrader`, a CronJob that periodically downloads the configured GitHub chart source and runs a full `helm upgrade --install platform-validator` with `--atomic`, `--wait`, and `--cleanup-on-fail`. It uses `HELM_DRIVER=configmap`, `concurrencyPolicy: Forbid`, and pins only non-secret live references for the database URL Secret, namespace, wallet Secret, wallet name/hotkey labels, `validator.deploymentNameOverride=platform-validator`, and `persistence.existingClaim=platform-validator-state`. It references the validator wallet Secret by name without printing its contents. Validate the full install flow only against a disposable cluster or
namespace with disposable test hotkey material.

Stop only installer-managed validator objects:

```bash
./scripts/install-validator.sh --cleanup
```

## Runtime Commands

```bash
kubectl -n platform-validator get deployment platform-validator
kubectl -n platform-validator get pods
kubectl -n platform-validator logs -f deployment/platform-validator
kubectl -n platform-validator describe deployment platform-validator
```

## Secret Handling

The only secret requested during install is the validator hotkey mnemonic. Never
enter coldkey material. Do not store mnemonics in `.env`, shell history, support
threads, screenshots, or evidence logs.

The installer creates or updates a Kubernetes Secret named `platform-validator-wallet`
from generated hotkey files and deletes the temporary local wallet directory when it
exits. Kubernetes Secrets are readable to cluster admins and any subject with
Secret read RBAC; use a dedicated namespace and enable encryption at rest for
production clusters.

## Registry, Wallet, And Optional Database Settings

```text
PLATFORM_VALIDATOR_REGISTRY_URL=https://chain.platform.network
PLATFORM_NAMESPACE=platform-validator
PLATFORM_WALLET_NAME=platform-validator
PLATFORM_WALLET_HOTKEY=validator
PLATFORM_DATABASE_URL=postgresql+asyncpg://platform:<password>@postgres.platform.svc.cluster.local/platform  # optional external override
PLATFORM_DATABASE_URL_SECRET_NAME=platform-validator-database-url
PLATFORM_DATABASE_URL_SECRET_KEY=url
PLATFORM_BROKER_ALLOWED_IMAGES=ghcr.io/platformnetwork/,registry.example.com/platform/
PLATFORM_AUTO_UPGRADE_SCHEDULE=*/5 * * * *
PLATFORM_AUTO_UPGRADE_HELM_IMAGE=alpine/helm:3.15.4
PLATFORM_AUTO_UPGRADE_REPO=PlatformNetwork/platform
PLATFORM_AUTO_UPGRADE_REF=main
```

The auto-upgrade image must contain the `helm` CLI and basic archive download tools. The default is `alpine/helm:3.15.4`.

Before relying on self-upgrades, verify Secret/PVC/key references without printing secret values:

```bash
kubectl -n platform-validator get secret platform-validator-database-url -o jsonpath='{.data.url}' >/dev/null
kubectl -n platform-validator get secret platform-validator-wallet -o jsonpath='{.data.hotkey}' >/dev/null
kubectl -n platform-validator get secret platform-validator-wallet -o jsonpath='{.data.hotkeypub\.txt}' >/dev/null
kubectl -n platform-validator get pvc platform-validator-state
kubectl -n platform-validator get cronjob platform-validator-helm-upgrader
```

If any prerequisite is missing, keep the Helm-upgrader CronJob suspended with `autoUpgrade.suspend=true` until the referenced Secret/PVC exists with the intended key or name. If the validator is already healthy, do not replace it just to bootstrap CronJob values; confirm the live references above and update only the CronJob/bootstrap manifests when needed.

The validator pod sees the hotkey at:

```text
/var/lib/platform/wallets/platform-validator/hotkeys/validator
```

## Kubernetes Scope

The installer applies only namespaced resources needed by the validator:
Namespace, validator ServiceAccount/RBAC, Helm-upgrader ServiceAccount/RBAC, PVC,
ConfigMap, Secret, Deployment, and Helm-upgrader CronJob. Cleanup removes only the
installer-managed Helm-upgrader CronJob/RBAC/ServiceAccount plus the validator
legacy image-updater CronJob/RBAC/ServiceAccount, managed Postgres StatefulSet and Service, Deployment, ConfigMap, database URL Secret, Role, RoleBinding, and ServiceAccount. The managed Postgres credential Secret, managed Postgres data claim/PVC, validator state PVC, and wallet Secret are
preserved intentionally so validator state and key material are not destroyed by an update; delete
them manually only when you intentionally want to erase local validator state and credentials.

Kubernetes mode requires PostgreSQL control-plane state. The installer provides managed validator Postgres by default; an external PostgreSQL `PLATFORM_DATABASE_URL` or `--database-url` overrides that default. Kubernetes mode also requires
registry-scoped `PLATFORM_BROKER_ALLOWED_IMAGES`. SQLite URLs, wildcards, and
broad prefixes such as `platformnetwork/` fail settings validation.


## Agent Challenge Platform SDK Execution Checks

Agent Challenge production Terminal-Bench rollout uses `platform_sdk` through the generic Platform broker. The public proxy must still expose only challenge public routes and must block `/internal/*`, `POST /internal/v1/submissions/{submission_id}/launch`, and generic benchmark execution-shaped routes such as `/benchmark-executions`; the broker is an internal execution substrate, not a public miner API.

Use placeholder commands only and avoid printing token values:

```bash
kubectl -n <validator-namespace> get pods -l app.kubernetes.io/name=agent-challenge
kubectl -n <validator-namespace> logs deployment/<agent-challenge-deployment> --since=30m | rg 'terminal_bench|platform_sdk|tb_running'
kubectl -n <validator-namespace> logs deployment/<platform-broker-deployment> --since=30m | rg 'run request|created job|agent-challenge-terminal-bench-runner'
kubectl -n <validator-namespace> logs deployment/<agent-challenge-deployment> --since=30m | rg -- '--environment-import-path agent_challenge_runner.platform_environment:PlatformEnvironment'
! kubectl -n <validator-namespace> logs deployment/<agent-challenge-deployment> --since=30m | rg --fixed-strings -- '--env daytona'
! kubectl -n <validator-namespace> logs deployment/<agent-challenge-deployment> --since=30m | rg --fixed-strings -- '--env platform'
curl -sS '<api-base-url>/submissions/<submission-id>/status' | rg '"status":"evaluating"|"phase":"evaluation"|"status":"valid"|"status":"error"'
```

Safe Agent Challenge knobs are `CHALLENGE_TERMINAL_BENCH_EXECUTION_BACKEND=platform_sdk`, broker URL plus token file, `CHALLENGE_PLATFORM_SDK_RUNNER_IMAGE=ghcr.io/platformnetwork/agent-challenge-terminal-bench-runner:latest`, `CHALLENGE_PLATFORM_SDK_ENVIRONMENT_IMPORT_PATH=agent_challenge_runner.platform_environment:PlatformEnvironment`, and a scoped allowed-image policy. Platform SDK Harbor commands use `--environment-import-path`, not `--env platform`, and production does not require Daytona credentials. Roll back to `harbor` only for non-production testing or for an explicitly credentialed legacy Harbor environment; production remains `platform_sdk` after rollout.

## Validation

```bash
bash -n scripts/install-validator.sh
uv run pytest tests/unit/test_validator_install_docs.py
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest --cov=platform_network --cov-report=term-missing --cov-fail-under=80
```

The full installer is an interactive real install. Run it only when the current
Kubernetes context, namespace, and hotkey material are safe to mutate. CI
publishes Docker images to GHCR only from trusted events: PRs build with
`push: false`, while `main`, `v*.*.*` tags, and confirmed manual runs publish.
Kubernetes does not notice GitHub or chart changes by itself; the installed CronJob
runs a full Helm upgrade from the configured repo/ref and lets Helm reconcile the namespace.

If Kubernetes or a Python tool is unavailable, record the missing tool as a
blocker instead of marking that surface as tested.
