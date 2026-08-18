# PRISM — enable Real Lium + emission unlock

## Live Lium (no emission change)

1. Rotate any leaked API key; set `LIUM_API_KEY` in operator env only.
2. Mount SSH key for pod access (`~/.config/prism-mission/lium_ssh_ed25519`).
3. Pin public eval image digest in config when Real `exec_eval` is fully wired.
4. Run inventory probe → single rent smoke → terminate → `verify_terminated`.
5. Prod default is `PRISM_MAX_CONCURRENT_EVALS=8` (orchestrator worker count /
   semaphore). Dial down only if the Lium lease pool cannot absorb the load.
6. **Control-plane restart / redeploy (GPU-safe):** keep
   `PRISM_PAYER_VAULT_DIR` + `PRISM_PAYER_VAULT_KEY_FILE` on a durable volume.
   Seal TTL defaults to ≥**36h** (`PRISM_PAYER_VAULT_TTL_SECS`, floored by
   train wall + eval + skew); measure + heartbeats re-seal so full-budget
   runs survive a bounce. Healthy mid-flight pods are resumed (not
   terminated). Do not manually kill Lium pods after a routine
   `prism-challenge` bounce — only stop pods that surface
   `control_plane_restart` / `harness_detached` (dead pod or unrecoverable
   seal). Prefer rolling the challenge image when no pods are in
   `provisioning`/`running`, or accept resume after boot.

## Prism v2.1 safe operator environment

Keep the live `:28092` service on the established scoring/emission surface:

```bash
PRISM_SCORING_MODE=benchmarks
PRISM_ANCHOR_VERSION=0
PRISM_EMISSION_MODE=wta
PRISM_OWNER_ARCH_CREDIT_BPS=0
PRISM_EVAL_REQUIRE_PRIVATE=0
PRISM_POD_GPU_COUNT=4
```

Unknown emission modes fall back to WTA; unknown anchor versions fall back
to v0. Do not combine a pod/image change with a composite, private-required,
owner-credit, top3, or sig flip.

For an isolated v3 calibration wave (not the live service), use:

```bash
PRISM_FLOW=v3
PRISM_SCORING_MODE=benchmarks   # do not flip live or isolated :28092 to composite
PRISM_ANCHOR_VERSION=3          # measure only; placeholders until pre-register
PRISM_TRAIN_FLOPS_CAP=3.0e18
PRISM_MIN_SPEND_FRACTION=0.5
PRISM_EVAL_G2_TASKS=lambada,hellaswag,piqa,arc_easy
PRISM_EVAL_BATTERY_BUDGET_S=3600
PRISM_G6_BPB_THRESHOLD=1.5
PRISM_POD_GPU_COUNT=4
```

The 0.5 floor applies only when `binding_cap=none`; `steps`, `wall`, and
`flops` are protocol stops and exempt. v3 remains a placeholder anchor set:
the calibration wave measures references and does not authorize a live flip.

Build a private-tier assets directory (hard cap: 400 rows per JSONL file):

```bash
PACK_TIER=private \
PRISM_EVAL_ASSETS_DIR=/var/lib/prism/eval-assets-private \
python3 crates/prism-recipe/harness/eval/build_private_pack.py
```

Verify `tier.json` says `private`, `manifest.json` hashes every asset, and the
completed run reports `battery.mirror_defence.contamination_checked=true`.

## CUDA 13 + Transformer Engine pod image

The workflow is manual: it builds in GHCR, mirrors the execution artifact to
the existing private DigitalOcean registry, and reports the provider digest.
Runtime pins must use that digest:

```bash
gh workflow run images.yml \
  --ref prism-v2.1-scoring \
  -f prism_pod_only=true
```

Set the staged service with:

```bash
PRISM_POD_IMAGE_REF=registry.digitalocean.com/basecrawl/prism-pod@sha256:fe1197b26e30ebd88f200963cc8528533326666873880b62e676adb51663ff88
PRISM_POD_IMAGE_TAG=v10-cuda13-te
PRISM_POD_DOCKER_CREDENTIAL_ID=<lium-docker-credential-id>
```

The credential ID is a non-secret reference; the registry username/password
remain stored in Lium. It is needed to create a new provider template, whose
name is digest- and credential-scoped. Lium needs the tag as a pull locator,
but records and checks the digest separately; malformed or missing digest
refs fail closed. The image must use overridable Docker `CMD`; the Lium
bootstrap injects `USER_PUBLIC_KEY` through a metacharacter-free command, then
the image script writes `authorized_keys`, touches `/root/container_ready`,
and keeps sshd in the foreground. Before promotion, run the billable test with
`PRISM_LIVE_RUNNING_TIMEOUT_SECS=1800`; bound retries during diagnosis with
`PRISM_LIVE_MAX_ATTEMPTS=1` (accepted range 1–8). One 4-GPU rent must prove:

1. `torch.cuda.device_count() == 4`;
2. `transformer_engine.pytorch` and
   `transformer_engine.common.recipe.NVFP4BlockScaling` import;
3. the dependency install phase can build/install a harmless manifest;
4. a stream-owned 4-GPU train produces nonzero attested FLOPs and G6 probe
   points;
5. the fresh train netns has `lo` up and no external route;
6. all anchored v3 G1–G8 keys are present; and
7. the pod terminates and disappears from Lium inventory.

### Reversible 1-GPU live cutover

Drain active `provisioning`/`running` rows, change only
`PRISM_POD_GPU_COUNT=1`, restart `prism-challenge`, and submit one operator
smoke. Confirm the selected offer is exactly one RTX 5090 and G7 records one
attested GPU. Do not edit Compose topology or the `:28092` scoring/emission
knobs. Restore `PRISM_POD_GPU_COUNT=4` and restart to roll back; already
rented pods retain their original width.

## Failed measure with EVAL_OK but no metrics (ops)

If `error_detail` ends with cheatguard / `EVAL_OK` but has no parseable
`METRICS_JSON=` / `bpb` (historical 32 KiB log-tail harvest bug), the row is
**not** recoverable from Postgres alone — `metrics_json` was never written.
After deploying the sidecar/grep harvest fix:

1. `POST /v1/submissions/{id}/retry` with Prism admin Bearer (re-queues measure).
2. `POST /v1/admin/gating/{hotkey}/reset` if the miner is still 1-max gated.
3. Miner (or operator with sealed BYOK vault) must fund another Lium run.

## Emission ceremony (shared with design)

Trust root today: **`prism = 5000` bps**, **`design = 5000` bps** (sum must stay
exactly `10000`). See [`config/challenges.toml`](../../config/challenges.toml).

To rebalance shares:

1. Choose `emission_share_bps` for `prism` and `design` such that the sum is
   **10000**.
2. Edit `config/challenges.toml`, re-sign with owner key
   ([`trust-root-rotation.md`](./trust-root-rotation.md)).
3. Roll validators with the new trust root (dual-accept if rotating).
4. Prefer the design-facing checklist when changing design emission:
   [`design-enable-and-emission.md`](./design-enable-and-emission.md).

Current committed default: **prism = 5000 bps**, **design = 5000 bps**.
