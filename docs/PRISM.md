# PRISM challenge (Base)

**challenge_id:** `prism`  
**scoring_version:** `1`  
**port:** `8092`  
**emission_share_bps:** `0` (until owner ceremony)  
**GPU path:** master-centralized **Lium** (no Phala CVM)

## What it is

PRISM on Base accepts miner two-script submissions (`architecture.py` + `training.py`), runs **operator-owned** eval jobs on Lium GPU pods (or Sim in CI), maps BPB → integer leaf scores, and emits D24-complete signed leaves for gateway ingest.

This is **not** agent-challenge Phala/TDX attestation and **not** hypertraining B300 tournament code.

## Crates

| Crate | Role |
|-------|------|
| `prism-challenge-task` | Identity constants / domains |
| `prism-lium` | Lium REST client + `SimLiumBackend` + `EvalReceipt` |
| `prism-challenge` | Submit API, pipeline, score, D24 leaves, gateway client |
| `bins/prism-challenge` | Operator binary `:8092` |

## Run (sim / local)

```bash
export PRISM_CHALLENGE_SK_FILE=/root/.base-secrets/challenge-prism.age
# or BASE_CHALLENGE_SK_FILE
cargo run -p prism-challenge-bin -- identity
cargo run -p prism-challenge-bin -- serve --bind 127.0.0.1:8092
curl -s http://127.0.0.1:8092/health
```

## Miner submit

`POST /v1/submissions`

```json
{
  "miner_hotkey": "<64 hex chars>",
  "architecture_py": "def build_model(ctx): ...",
  "training_py": "def train(model, ctx): ..."
}
```

## Live Lium

```bash
export LIUM_API_KEY=...   # never commit; rotate if pasted in chat
cargo run -p prism-challenge-bin --quiet  # not required
# inventory smoke:
cargo test -p prism-lium -- --ignored live_  # if live tests present
# or:
python3 - <<'PY'
# use credentials from /root/.config/prism-mission/credentials.env
PY
```

Cost guards: `max_lifetime_hours >= 1`, `max_price_per_hour > 0`, terminate + verify on every path.

## Trust root

`config/challenges.toml` lists `prism` with emission **0**. Re-sign after edits:

```bash
cargo run -p trustroot-bin -- sign \
  --key ~/.base-secrets/owner-throwaway.age \
  --age-identity ~/.base-secrets/age-identity.txt \
  --input config/challenges.toml --kind challenges
```

## Gateway registration

```json
{ "challenge_id": "prism", "base_url": "http://prism-challenge:8092", "weight": 1 }
```

## Tests

```bash
cargo test -p prism-challenge-task -p prism-lium -p prism-challenge -p prism-challenge-bin
```

## Must not

- Phala CVM / TDX path for PRISM GPUs  
- Non-zero emission without ceremony  
- Touch agent-v1 freeze or move its bps without explicit owner order  
- Commit `LIUM_API_KEY` or challenge secrets  
