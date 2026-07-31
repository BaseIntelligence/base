# PRISM — enable Real Lium + emission unlock

## Live Lium (no emission)

1. Rotate any leaked API key; set `LIUM_API_KEY` in operator env only.
2. Mount SSH key for pod access (`~/.config/prism-mission/lium_ssh_ed25519`).
3. Pin public eval image digest in config when Real `exec_eval` is fully wired.
4. Run inventory probe → single rent smoke → terminate → `verify_terminated`.
5. Keep `max_concurrent_evals=1` until lease proven.

## Emission ceremony (separate)

1. Choose `emission_share_bps` for `prism` such that sum with `agent-v1` (+ hypertraining) = **10000**.
2. Edit `config/challenges.toml`, re-sign with owner key.
3. Roll validators with new trust root (dual-accept if rotating).
4. Do **not** change agent-v1 freeze beyond the bps line.

Default until ceremony: **prism = 0 bps**.
