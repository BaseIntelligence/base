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
