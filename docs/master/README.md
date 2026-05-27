# Cortex Foundation Master Installation Guide

Foundation-only installer for Cortex Foundation master infrastructure. Do not run this for validators or third-party operators.

This guide covers the committed Kubernetes installer for the master control plane. It installs the Platform master admin API, proxy, broker, shared master ConfigMap, active image auto-update CronJobs, and a suspended full Helm auto-upgrade CronJob in the master namespace. It does not install validator workloads, chain submission jobs, or any key material.

## Default Namespace

```text
PLATFORM_NAMESPACE=platform-master
```

Use a different namespace only for Cortex Foundation managed test clusters. Do not reuse the namespace reserved for normal operator installs.

## Automatic Install

Run from the repository root:

```bash
./scripts/install-master.sh --database-url postgresql+asyncpg://platform:<password>@postgres.platform.svc.cluster.local/platform
```

The script performs these actions:

1. Prints the foundation-only warning before it changes the cluster.
2. Applies Namespace, ServiceAccount/RBAC, ConfigMap, admin Deployment and Service, proxy Deployment and Service, broker Deployment and Service, image updater CronJobs, and `platform-master-helm-upgrader` without deleting healthy existing workloads.
3. Stores the required database URL in `secret/platform-master-database-url` and references it from Deployments.
4. Runs the master admin API with `platform master run --config config/master.kubernetes.yaml`.
5. Runs the proxy and broker with the same master config.

Useful options:

```bash
export PLATFORM_DATABASE_URL='postgresql+asyncpg://platform:<password>@postgres.platform.svc.cluster.local/platform'
./scripts/install-master.sh --namespace platform-master
./scripts/install-master.sh --image ghcr.io/platformnetwork/platform-master:v1.2.3@sha256:<digest>
./scripts/install-master.sh --image-update-schedule '*/1 * * * *'
./scripts/install-master.sh --image-updater-image ghcr.io/platformnetwork/platform:latest
./scripts/install-master.sh --auto-upgrade-schedule '*/5 * * * *'
./scripts/install-master.sh --auto-upgrade-helm-image alpine/helm:3.15.4
./scripts/install-master.sh --auto-upgrade-repo PlatformNetwork/platform --auto-upgrade-ref main
./scripts/install-master.sh --netuid 100
./scripts/install-master.sh --cleanup
```

Cleanup is scoped to installer-managed master objects and removes `secret/platform-master-database-url`. It does not delete unrelated workloads or namespaces.

## Full Helm Auto-Upgrade

The installer creates `cronjob/platform-master-admin-image-updater`, `cronjob/platform-master-proxy-image-updater`, and `cronjob/platform-master-broker-image-updater` for the normal mutable image channel. These jobs patch only their named Deployments to the latest digest of the configured master image tag.

The installer also creates `cronjob/platform-master-helm-upgrader`, suspended by default. The job uses a namespace-local ServiceAccount with ConfigMap-backed Helm release storage and runs a full Helm upgrade from GitHub when explicitly unsuspended:

```text
helm upgrade --install platform-master ... --atomic --wait --cleanup-on-fail
```

The upgrader downloads the configured repo/ref, reads the chart under `deploy/helm/platform`, and applies master-only values in the master namespace. It sets `HELM_DRIVER=configmap`, uses `concurrencyPolicy: Forbid`, and does not read or print Kubernetes Secret values. The master database URL must be supplied by the existing Secret referenced by the chart values. The installer pins only live-safe non-secret references for future self-upgrades, including `database.urlSecret.name=platform-master-database-url`, `database.urlSecret.key=url`, `security.existingSecret=platform-secrets`, `kubernetes.namespace`, and `kubernetes.serviceAccount`; never place database URLs, tokens, or other secret values in `autoUpgrade.extraSet`.

Before relying on self-upgrades, verify the referenced Secret and keys exist without printing their values:

```bash
kubectl -n platform-master get secret platform-master-database-url -o jsonpath='{.data.url}' >/dev/null
kubectl -n platform-master get secret platform-secrets -o jsonpath='{.data.admin_token}' >/dev/null
kubectl -n platform-master get cronjob platform-master-admin-image-updater platform-master-proxy-image-updater platform-master-broker-image-updater
kubectl -n platform-master get cronjob platform-master-helm-upgrader
```

If any prerequisite is missing, keep the Helm-upgrader CronJob suspended with `autoUpgrade.suspend=true` until the referenced Secret exists with the intended key. If the deployment is already healthy, use these checks to confirm the CronJob bootstrap references instead of replacing the deployment just to recreate the CronJob.

## Explicit Non Goals

- It does not create validator resources.
- It does not run the master weights CLI command.
- It does not create a master on-chain submission CronJob.
- It does not ask for, print, or store key material.
- It does not use external paste services as the canonical source.

## Runtime Checks

```bash
kubectl -n platform-master get deployment platform-master-admin platform-master-proxy platform-master-broker
kubectl -n platform-master get cronjob platform-master-helm-upgrader
kubectl -n platform-master logs -f deployment/platform-master-admin
```

## Validation Commands

Before changing the installer or docs, run:

```bash
bash -n scripts/install-master.sh
uv run pytest tests/unit/test_master_install_docs.py tests/unit/test_validator_install_docs.py -q
```

Run the full installer only when the current Kubernetes context and namespace are owned by Cortex Foundation master infrastructure.
