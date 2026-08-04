# Design challenge — enable real backends + emission unlock

Design ships with `emission_share_bps = 0` until an explicit owner ceremony.
Until then **prism holds `10000` bps** (sole weight source). See
[`config/challenges.toml`](../../config/challenges.toml) and
[`DESIGN_CHALLENGE.md`](../DESIGN_CHALLENGE.md).

## A. Bring up design without emission (staging / local)

1. Materialize env: `./deploy/scripts/materialize-env.sh`
2. Ensure secrets (mode **0400**, uid **65532**):
   - `deploy/secrets/design_sk` — challenge signing mini-secret
   - `deploy/secrets/openrouter/api_key` — egress proxy LLM path + agentic review (SimAgent if absent; **never** host Sim in staging/prod)
   - `deploy/secrets/design/annotator_tokens` — operator **admin winners** bearers (one token per line; hashed at boot). Optional override: `DESIGN_ADMIN_TOKENS_FILE`
3. Confirm trust-root pubkey for `design` matches `design_sk` public key.
4. Start stack (compose wiring lands with deploy-wiring todo). Health:
   ```bash
   curl -sS http://127.0.0.1:28093/health
   curl -sS http://127.0.0.1:28092/health   # prism still required for weights
   ```
5. Register challenge with gateway after redeploy:
   ```json
   { "challenge_id": "design", "base_url": "http://design-challenge:8093", "weight": 1 }
   ```
6. Keep `emission_share_bps = 0` for design until ceremony. Leaves still emit
   (D24 exact-E); they simply carry zero emission share.

## B. Keygen (`design_sk`)

```bash
cargo run -p trustroot -- keygen --out deploy/secrets/design_sk
# record public_key hex into config/challenges.toml [[challenges]] id = "design"
```

Never bake the mini-secret into images or cloud-init. Sandbox and egress proxy
must **not** mount `design_sk`.

## C. Emission ceremony (owner)

Goal: move weight from prism → design (sum must remain exactly **10000** bps).

1. Choose new shares, e.g. `design = N`, `prism = 10000 - N` (both ≥ 0).
2. Edit `config/challenges.toml` emission lines only (plus pubkey if rotating).
3. Re-sign: follow [`../../config/CEREMONY.md`](../../config/CEREMONY.md) and
   [`trust-root-rotation.md`](./trust-root-rotation.md) (dual-accept if rotating).
4. Roll **all** validators with the new signed trust root before relying on the
   new shares.
5. Do **not** change design `challenge_scoring_version` or freeze pins in the
   same ceremony unless intentionally bumping the scoring contract.

Default until this ceremony: **design = 0 bps**, **prism = 10000 bps**.

## D. Admin winners tokens + OpenRouter

Human scoring role is **only** selecting 1 or 2 round winners (not prompt
approval; Elo annotate is deprecated / unused on the leaf path).

- Admin bearers: put one token per line in
  `deploy/secrets/design/annotator_tokens` (mode **0400**, uid **65532**).
  Hashed into challenge config at boot; optional `DESIGN_ADMIN_TOKENS_FILE`.
- Award path: `GET /v1/admin/rounds/{id}/candidates` →
  `POST /v1/admin/rounds/{id}/winners` with
  `{ "harness_ids": ["…"] }` (length 1 → `SCORE_MAX`, length 2 → each
  `SCORE_MAX / 2`). See [`DESIGN_CHALLENGE.md`](../DESIGN_CHALLENGE.md) §8.
- Without OpenRouter key: egress LLM path and agentic review use Sim backends
  in CI/local only; staging/prod must use Docker + keyed OpenRouter.
- Legacy annotate routes (`GET /v1/annotate/next`, `POST /v1/annotate`) are
  deprecated and unused for leaf scores.

## E. Rollback emission

Re-sign challenges.toml with previous bps (design 0 / prism 10000), roll
validators. Challenge service can keep running; emission share alone changes
aggregation weight.
