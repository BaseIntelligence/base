<!-- protocol_version: 1 -->

# Bounty challenge — HTTP video bug reports

**challenge_id:** `bounty`  
**scoring_version:** `1` (TARGET_BUGS=50 + UID-0 burn sink)  
**Path:** HTTP only — **no Phala/CVM**

Normative docs: [`../BOUNTY_CHALLENGE.md`](../BOUNTY_CHALLENGE.md).

## What you submit

A **multipart** bug report with a short video showing the issue:

| Field | Required | Notes |
|-------|----------|-------|
| `video` | yes | `mp4` / `webm` / `mov` (raw ≤ ~100 MiB) |
| `title` | yes | Short title |
| `description` | yes | What went wrong |
| `app_id` | yes | Target app slug |
| `steps` | no | Repro steps |

Header: `X-Miner-Hotkey` (64 lowercase hex). Your hotkey must be **registered
on the subnet** (metagraph). Duplicate reports of the same bug within 24h are
rejected by an automated similarity check; novel reports wait for operator
admin approve before earning epoch points.

You do **not** deploy a miner CVM. Scoring awards points for **approved
reports**, not for shipping a product fix.

## Submit

```bash
# Via gateway (preferred)
curl -sS -X POST "$BASE_GATEWAY/challenge/bounty/v1/bugs" \
  -H "X-Miner-Hotkey: <64 lowercase hex>" \
  -F "video=@bug.mp4;type=video/mp4" \
  -F "title=Checkout button does nothing" \
  -F "description=Clicking Pay on /pricing leaves the spinner forever." \
  -F "app_id=example-shop" \
  -F "steps=1) Open /pricing 2) Click Pay 3) Observe spinner"

# Local / direct (env-local host port)
curl -sS -X POST "http://127.0.0.1:28095/v1/bugs" \
  -H "X-Miner-Hotkey: <64 lowercase hex>" \
  -F "video=@bug.mp4;type=video/mp4" \
  -F "title=…" -F "description=…" -F "app_id=…"
```

## Status / video

```bash
curl -sS "$BASE_GATEWAY/challenge/bounty/v1/bugs/<id>"
curl -sS "$BASE_GATEWAY/challenge/bounty/v1/bugs/<id>/video" -o compressed.mp4
curl -sS "$BASE_GATEWAY/challenge/bounty/v1/bugs?mine=1"
curl -sS "$BASE_GATEWAY/challenge/bounty/v1/status"
```

Typical statuses: `uploaded` → processing → `pending_admin` | `rejected` |
`approved`.

## Scoring (epoch)

- Target: **50** admin-approved bugs per epoch = 100% of the bounty emission
  share (`emission_share_bps = 2500` today).
- Fewer than 50: remaining mass burns (UID 0).
- More than 50: full share is split proportionally among reporters (dilution).
- Your points this epoch = number of **approved** bugs for your hotkey.

Exact lattice: [`../BOUNTY_CHALLENGE.md`](../BOUNTY_CHALLENGE.md) §6.

## What not to do

- Do not paste challenge signing keys or wallet mnemonics into clients.
- Do not expect admin approve routes via the public gateway — those are
  master-local only.
- Do not spam near-duplicate videos; similarity rejects within 24h.

## Troubleshoot

See [`troubleshoot.md`](./troubleshoot.md) for shared HTTP / metagraph / quota
failures. Bounty-specific: oversized video → `413`; unknown hotkey →
`403 hotkey_not_in_metagraph`; duplicate → `rejected` with nearest id in detail.
