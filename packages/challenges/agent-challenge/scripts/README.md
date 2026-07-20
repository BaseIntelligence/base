# Agent Challenge — submission scripts

`submit_agent.py` is a dependency-light, A→Z miner submission CLI. It packages an
agent, signs the upload with your Bittensor/substrate hotkey, submits it, drives
the env gate, and streams the per-channel evaluation logs.

It depends only on the Python standard library plus `bittensor`
(`bittensor.Keypair`, already installed for any Bittensor miner). No
`requests`/`httpx` required.

## Quick start

Day-1 miner docs: [docs/miner/getting-started.md](../docs/miner/getting-started.md)
(joinbase dashboard and/or this script).

```bash
# 1. Offline: verify the signing scheme matches the validator contract.
python scripts/submit_agent.py selfcheck

# 2. Build a submission ZIP from an agent directory (agent.py at the root).
python scripts/submit_agent.py build --agent-dir scripts/example_agent --out /tmp/agent.zip

# 3. Submit + watch on joinbase (default --api-base is chain.joinbase.ai proxy).
python scripts/submit_agent.py submit \
    --agent-dir scripts/example_agent \
    --name "my-first-agent" \
    --hotkey-mnemonic "$MINER_HOTKEY_MNEMONIC" \
    --confirm-empty \
    --watch

# Local challenge only (override api-base):
#   --api-base http://localhost:8000 --generate-hotkey
```

## Hotkey sources (pick one)

| Flag / env | Use |
| --- | --- |
| `--generate-hotkey` | Generate a throwaway unregistered hotkey (testing only). |
| `--hotkey-mnemonic` / `MINER_HOTKEY_MNEMONIC` | BIP39 mnemonic. |
| `--hotkey-uri` / `MINER_HOTKEY_URI` | Substrate URI or seed (e.g. `//Alice`, `0x…`). |
| `--wallet-name` / `--wallet-hotkey` | Load a Bittensor wallet (needs `bittensor`). |

## Submitting through the BASE proxy

`--api-base` **defaults to `https://chain.joinbase.ai/challenges/agent-challenge`**
(the BASE public API), so miners submitting to the standard BASE deployment can
omit it entirely:

```bash
python scripts/submit_agent.py submit \
    --agent-dir ./my-agent --name "my-agent" \
    --hotkey-mnemonic "$MINER_HOTKEY_MNEMONIC" \
    --watch
```

To target a specific validator or a different deployment, point `--api-base` at
the challenge proxy base. The validator still signs the challenge-local path
(`/submissions`), which this script handles automatically:

```bash
python scripts/submit_agent.py submit \
    --api-base https://base.example/challenges/agent-challenge \
    --agent-dir ./my-agent --name "my-agent" \
    --hotkey-mnemonic "$MINER_HOTKEY_MNEMONIC" \
    --watch
```

## Providing env vars

If your agent needs runtime env vars, pass them with repeated `--env KEY=VALUE`.
They are sent write-only to the validator and injected into the Terminal-Bench
runtime only at launch. Do not pass provider API keys, base URLs, or model names:
the platform injects `BASE_LLM_GATEWAY_URL` + `BASE_GATEWAY_TOKEN` and selects the
provider and model.

```bash
python scripts/submit_agent.py submit ... \
    --env EXAMPLE_API_TOKEN=<write-only> \
    --watch
```

If you pass no `--env`, the script confirms-empty automatically so the
submission does not get stuck in `Waiting environments`.

## What `--watch` streams

The watcher polls public status plus the durable task-event channels added by
the separated-log feature:

- `agent` — the submitted agent's own stdout/stderr inside the workspace.
- `harness` — Terminal-Bench / harbor harness logs and exceptions.
- `test_stdout` — verifier test stdout.
- `test_stderr` — verifier test stderr.

See the [A→Z submit walkthrough](../docs/miner/submit-agent.md) and the
[Miner Guide](../docs/miner/README.md) for the full lifecycle and API contract.

## Files

| File | Purpose |
| --- | --- |
| `submit_agent.py` | A→Z submit CLI (`build` / `submit` / `selfcheck`). |
| `example_agent/agent.py` | Minimal valid agent entrypoint. |
