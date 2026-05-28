# Validator Quick Start

This page is only for normal validator Kubernetes installation. The validator
fetches the public registry from `https://chain.platform.network/v1/registry`,
runs active challenge workloads through Kubernetes, and keeps them synchronized.

## Automatic Install

Run from the repository root:

```bash
./scripts/install-validator.sh
```

The installer asks only for the validator hotkey mnemonic. Do not enter coldkey
material. The mnemonic is read silently, converted to hotkey files in a temporary
local directory, and stored as a Kubernetes Secret.

Normal install performs a real Kubernetes installation and prompts for the
validator hotkey mnemonic. When `--database-url` and `PLATFORM_DATABASE_URL` are unset, it creates namespace-scoped managed validator Postgres resources named `platform-validator-postgres` with image `postgres:16-alpine`, 10Gi storage, optional `PLATFORM_VALIDATOR_POSTGRES_STORAGE_CLASS`, and a retained `platform-validator-postgres-data` claim. The generated URL is stored only in `secret/platform-validator-database-url` and is read by the Deployment through `PLATFORM_DATABASE__URL`; the script does not print database URLs or passwords. Supplying `--database-url` or `PLATFORM_DATABASE_URL` skips the managed validator Postgres Secret, Service, StatefulSet, and data claim and uses the provided external database URL Secret instead. It also installs `cronjob/platform-validator-helm-upgrader`,
a scoped CronJob that periodically downloads the configured GitHub chart source
and runs a full `helm upgrade --install platform-validator` with `--atomic`,
`--wait`, and `--cleanup-on-fail`. It uses `HELM_DRIVER=configmap`,
`concurrencyPolicy: Forbid`, and pins only non-secret live references for future
self-upgrades, including `validator.deploymentNameOverride=platform-validator`
so Helm manages the existing standalone Deployment instead of creating a second
validator Deployment. Use a disposable namespace and test mnemonic when
validating the full install flow.

Follow the validator:

```bash
kubectl -n platform-validator logs -f deployment/platform-validator
```

Stop only installer-managed validator objects:

```bash
./scripts/install-validator.sh --cleanup
```

Cleanup removes the configured database URL Secret because the installer creates it. It deletes the managed validator Postgres StatefulSet and Service, but preserves the managed Postgres credential Secret, its data claim/PVC, the validator wallet Secret, and the validator state PVC by default.

## Kubernetes Requirement

Kubernetes is mandatory for the first-party validator installer. Run the script
only after the VM or server already has a working Kubernetes cluster and
`kubectl` points at the target context. For a single validator VM, prefer `k3s`;
use `minikube` only for local smoke tests or disposable validation. A validator
VM used for validation should have at least 2 vCPUs and 8 GB RAM.

The script creates the validator Kubernetes resources and, by default, managed
validator Postgres. You only need `--database-url` when you intentionally want an
external PostgreSQL service instead of the installer-managed default.

## Manual Install

Create equivalent Kubernetes resources manually: namespace, service account,
namespaced runtime RBAC, state PVC, validator ConfigMap, hotkey Secret, and a
Deployment that runs:

```text
platform validator run --config config/validator.kubernetes.yaml
```

The ConfigMap must set `runtime.backend: kubernetes`,
`validator.registry_url: https://chain.platform.network`, `database.url` or `PLATFORM_DATABASE_URL` to a PostgreSQL URL such as the installer-managed `postgresql+asyncpg://platform:<password>@platform-validator-postgres:5432/platform` or an external URL, `docker.broker_allowed_images` or `PLATFORM_BROKER_ALLOWED_IMAGES` to registry-scoped prefixes such as `ghcr.io/platformnetwork/`, and `kubernetes.in_cluster: true`. SQLite URLs, wildcard prefixes, and broad prefixes such as `platformnetwork/` are rejected in Kubernetes mode.

The installer-managed validator database is only for validator control-plane state. `PLATFORM_DATABASE_URL` is the external override for that validator control-plane database credential. It is not a challenge database credential and must not be copied into challenge specs. In Kubernetes managed challenge mode, Platform creates per-challenge Postgres resources and injects each challenge's own `CHALLENGE_DATABASE_URL` from the matching Secret.

## Challenge database behavior

When a validator starts active challenges through Kubernetes managed mode, each challenge slug receives isolated managed Postgres resources. Platform injects `CHALLENGE_DATABASE_URL` automatically from the per-challenge Secret. The challenge `/data` PVC remains separate and is still used for artifacts, analyzer output, local files, and the generated SQLite fallback.

Managed Postgres Secrets and data claims are retained by default when a challenge is removed. Manual deletion of those retained objects is destructive. Automated destructive purge is not part of this implementation.

## Safety

- The installer never needs coldkey material.
- Cleanup is scoped to this validator Deployment and its installer-managed objects.
- The default registry URL is `https://chain.platform.network`.
- The validator runs in Kubernetes mode.
- The hotkey Secret is readable by cluster admins and any subject with Secret read RBAC; use a dedicated namespace and enable Kubernetes Secret encryption at rest.
