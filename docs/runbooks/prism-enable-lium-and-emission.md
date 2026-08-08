# PRISM — enable Real Lium + emission unlock

## Live Lium (no emission change)

1. Rotate any leaked API key; set `LIUM_API_KEY` in operator env only.
2. Mount SSH key for pod access (`~/.config/prism-mission/lium_ssh_ed25519`).
3. Pin public eval image digest in config when Real `exec_eval` is fully wired.
4. Run inventory probe → single rent smoke → terminate → `verify_terminated`.
5. Prod default is `PRISM_MAX_CONCURRENT_EVALS=8` (orchestrator worker count /
   semaphore). Dial down only if the Lium lease pool cannot absorb the load.

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
