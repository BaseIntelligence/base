# Runbook: promote, rollback, restore

Digest-pinned rollouts for staging → prod. Updater behaviour: crate `updater` (D14). Compose layout: [`../../deploy/README.md`](../../deploy/README.md).

**Self-update of the updater is an operator one-shot, never automatic in prod.**

---

## 1. Preconditions

- [ ] CI green on the commit you are promoting.
- [ ] Images built and pushed as `repo@sha256:<64 hex>` only.
- [ ] Staging has run the new digest long enough to pass `/readyz` and smoke checks.
- [ ] You have SSH to the target host and age identity already on the box (R11).
- [ ] Postgres volume is healthy.

---

## 2. Backup Postgres **before** every prod promote

On the target host (paths assume `/opt/base` checkout; adjust if different):

```bash
set -euo pipefail
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
BACKUP_DIR="${BASE_BACKUP_DIR:-/var/backups/base}"
mkdir -p "$BACKUP_DIR"
chmod 700 "$BACKUP_DIR"

# Requires compose stack up with service name postgres
docker compose -f /opt/base/docker-compose.yml exec -T postgres \
  pg_dump -U "${POSTGRES_USER:-base}" -d "${POSTGRES_DB:-base}" \
  --no-owner --format=custom \
  > "$BACKUP_DIR/base-${STAMP}.dump"

ls -la "$BACKUP_DIR/base-${STAMP}.dump"
echo "BACKUP_OK=$BACKUP_DIR/base-${STAMP}.dump"
```

Record the path in the change ticket. An unexercised backup is not a backup (D17).

---

## 3. Promote (staging or prod)

### 3.1 Materialize env (if secrets changed)

```bash
cd /opt/base
export AGE_IDENTITY=/etc/base/age-identity.txt
./deploy/scripts/materialize-env.sh
# confirm modes
stat -c '%a %n' deploy/env/*.env
```

### 3.2 Set desired digest and let updater roll

Updater pulls **only** digest-pinned references, recreates the target container via socket-proxy, health-gates on `/readyz`, and rolls back on failure.

Operator pattern (illustrative env names; match your `deploy/env/updater.env`):

```bash
# Example: pin validator image (replace digest with the real one you built)
export BASE_UPDATER_DESIRED_IMAGE="ghcr.io/example/validator@sha256:REPLACE_WITH_64_HEX"

# Restart updater to pick config, or write pin file if your deploy uses pin_store paths
docker compose up -d updater
docker compose logs -f --tail=100 updater
```

Success signals:

- Updater log / state: desired digest adopted.
- Target container image digest matches desired.
- `curl -fsS http://127.0.0.1:<port>/readyz` exits 0 (port per service).

### 3.3 Master host only

```bash
docker compose --profile master up -d
docker compose ps
```

---

## 4. Rollback (bad digest)

Updater auto-rolls back when health fails after a swap. Manual path if you must force previous digest:

```bash
# Set desired image back to the last known-good digest
export BASE_UPDATER_DESIRED_IMAGE="ghcr.io/example/validator@sha256:PREVIOUS_64_HEX"
docker compose up -d updater
docker compose logs -f --tail=100 updater
docker compose ps
```

If the host is wedged:

```bash
docker compose stop validator gateway updater
# restore previous compose pin / image references in env
docker compose --profile master up -d   # on master; omit profile on pure validators
```

Prod must stay on the last good digest until staging proves the fix.

---

## 5. Restore Postgres from dump

```bash
set -euo pipefail
DUMP="${1:?usage: restore.sh /var/backups/base/base-....dump}"
test -f "$DUMP"

docker compose stop validator gateway updater
docker compose exec -T postgres \
  pg_restore -U "${POSTGRES_USER:-base}" -d "${POSTGRES_DB:-base}" \
  --clean --if-exists --no-owner \
  < "$DUMP"

docker compose --profile master up -d   # adjust profile for host role
docker compose ps
```

Then re-check `/readyz` and that validators reload without class-B storms.

---

## 6. Spot-check commands (clean shell, no live promote required)

From a developer checkout:

```bash
./deploy/scripts/materialize-env.sh
stat -c '%a' deploy/env/postgres.env | grep -qx 600
./deploy/scripts/assert-evil-gateway-not-default.sh
cargo run -q -p xtask -- external-docs-check
```

All must exit 0. Live `pg_dump` / promote requires a running stack on the droplet (tasks 41+).
