# base deploy (compose)

## Services

| Service | Profile | Image |
|---------|---------|--------|
| `postgres` | default | `postgres@sha256:33f9…` (16) |
| `validator` | default | build `deploy/Dockerfile` target `validator` |
| `updater` | default | build target `updater` |
| `socket-proxy` | default | `tecnativa/docker-socket-proxy@sha256:9e4b…` |
| `gateway` | **`master`** | build target `gateway` |

Default `docker compose up -d` starts **4** services and does **not** start gateway.
Owner host: `docker compose --profile master up -d` starts **5**.

## Hard rules

- No floating image tags (digest pins only).
- `/var/run/docker.sock` only on `socket-proxy` (read-only).
- socket-proxy allowlist: `CONTAINERS=1 IMAGES=1 POST=1` (matches `updater`).
- Secrets via age-decrypted env files mode **0600** under `deploy/env/*.env` — never in images or cloud-init.

## Quick start (local)

```bash
# 1) Release binaries (or set BASE_DOCKER_BUILD_FROM=source for full in-Docker rustc 1.96)
cargo build --release -p validator-bin -p gateway-bin -p updater-bin

# 2) Env files at 0600
./deploy/scripts/materialize-env.sh

# 3) Build service images + start default stack
export BASE_DOCKER_BUILD_FROM=prebuilt
docker compose build
docker compose up -d
docker compose ps

# 4) Master profile (gateway)
docker compose --profile master up -d
```

## Age secrets (production)

```bash
# On operator machine
age -r "$RECIPIENT" -o deploy/env/postgres.env.age deploy/env/postgres.env
# On droplet (identity delivered out of band)
export AGE_IDENTITY=/etc/base/age-identity.txt
./deploy/scripts/materialize-env.sh
```

## Infrastructure (DigitalOcean)

Terraform lives in [`terraform/`](./terraform/): two `s-8vcpu-16gb-amd` droplets
(`base-staging`, `base-prod`) in `nyc1` (nyc3 has no 8vCPU/16GB slug on this account) plus a firewall (SSH from operator IP
only; 80/443 open). Cloud-init installs Docker + Compose only.

Age delivery helpers:

```bash
# Encrypt (operator machine; recipient = age public key)
./deploy/scripts/age-encrypt-env.sh \
  --recipient "$(age-keygen -y /path/to/age-identity.txt)" \
  --src-dir deploy/env \
  --out-dir /tmp/base-env-age

# After OOB identity install on the droplet:
./deploy/scripts/age-push-env.sh --host root@DROPLET_IP --age-dir /tmp/base-env-age --materialize
```

See [`terraform/README.md`](./terraform/README.md) for apply steps and R11 notes.

## Test-only: evil-gateway profile (task 48)

**Never enable in production.** Adversarial staging harness:

```bash
docker compose --profile evil-gateway config --services   # must list evil-gateway
docker compose --profile master config --services         # must NOT list evil-gateway
./deploy/scripts/assert-evil-gateway-not-default.sh
```

Offline proofs (no live TAO): `cargo test -p validator a48_`


## Promotion pipeline (task 43)

Digest-only rollout with backup-before-pin and fail-closed prod.

```bash
# 1) CI (or local) records digests after build
./deploy/scripts/record-image-digests.sh

# 2) Promote known-good digest to staging (backs up Postgres first)
export PGHOST=... PGUSER=... PGPASSWORD=... PGDATABASE=base
export BASE_BACKUP_ENDPOINT=https://nyc3.digitaloceanspaces.com   # or local MinIO
export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...
export BASE_BACKUP_BUCKET=base-backups
./deploy/scripts/promote.sh \
  --env staging --service validator \
  --image ghcr.io/org/validator@sha256:<64-hex>

# 3) After staging is healthy, promote same digest to prod
./deploy/scripts/promote.sh \
  --env prod --service validator --confirm-prod \
  --image ghcr.io/org/validator@sha256:<64-hex>

# 4) Rollback = re-promote previous snapshot
./deploy/scripts/promote.sh --env staging --service validator --rollback

# 5) Restore drill (scratch DB row-count match)
./deploy/scripts/pg-restore-drill.sh --s3-uri s3://base-backups/pg/staging/<stamp>.sql.gz
```

Pin files: `deploy/pins/staging.json`, `deploy/pins/prod.json`.  
Staging promote **never** writes the prod pin. Prod promote requires staging ladder + `--confirm-prod`.  
Updater consumes `BASE_UPDATER_DESIRED_IMAGE` (also written to `deploy/pins/<env>.desired.env`).

Verify locally: `./deploy/scripts/verify-task-43.sh`
