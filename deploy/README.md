# gbase deploy (compose)

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
- socket-proxy allowlist: `CONTAINERS=1 IMAGES=1 POST=1` (matches `gbase-updater`).
- Secrets via age-decrypted env files mode **0600** under `deploy/env/*.env` — never in images or cloud-init.

## Quick start (local)

```bash
# 1) Release binaries (or set GBASE_DOCKER_BUILD_FROM=source for full in-Docker rustc 1.96)
cargo build --release -p gbase-validator-bin -p gbase-gateway-bin -p gbase-updater-bin

# 2) Env files at 0600
./deploy/scripts/materialize-env.sh

# 3) Build service images + start default stack
export GBASE_DOCKER_BUILD_FROM=prebuilt
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
export AGE_IDENTITY=/etc/gbase/age-identity.txt
./deploy/scripts/materialize-env.sh
```

## Infrastructure (DigitalOcean)

Terraform lives in [`terraform/`](./terraform/): two `s-8vcpu-16gb-amd` droplets
(`gbase-staging`, `gbase-prod`) in `nyc3` plus a firewall (SSH from operator IP
only; 80/443 open). Cloud-init installs Docker + Compose only.

Age delivery helpers:

```bash
# Encrypt (operator machine; recipient = age public key)
./deploy/scripts/age-encrypt-env.sh \
  --recipient "$(age-keygen -y /path/to/age-identity.txt)" \
  --src-dir deploy/env \
  --out-dir /tmp/gbase-env-age

# After OOB identity install on the droplet:
./deploy/scripts/age-push-env.sh --host root@DROPLET_IP --age-dir /tmp/gbase-env-age --materialize
```

See [`terraform/README.md`](./terraform/README.md) for apply steps and R11 notes.

## Test-only: evil-gateway profile (task 48)

**Never enable in production.** Adversarial staging harness:

```bash
docker compose --profile evil-gateway config --services   # must list evil-gateway
docker compose --profile master config --services         # must NOT list evil-gateway
./deploy/scripts/assert-evil-gateway-not-default.sh
```

Offline proofs (no live TAO): `cargo test -p gbase-validator a48_`
