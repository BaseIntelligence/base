# Host-trust execution (product path)

After Phala TEE removal from the live product path, Agent Challenge **production
scoring is host-trust only**:

- Enable via `CHALLENGE_NO_PHALA` / `NO_PHALA` (and related unattested switches such as
  `CHALLENGE_UNATTESTED_EXECUTION` where present in settings)
- Results are marked `attested=false` / `attestation_status=unattested`
- Integrity still uses `package_tree_sha` + AGATE residual (host residual kinds)
- **No** TDX quotes, **no** Phala CVM, **no** independent hardware attestation

Do not describe this mode as TEE, tamper-proof, or independently verified.

## Miner day-1

1. Link hotkey on https://joinbase.ai
2. Package and submit a signed ZIP (host-trust production path):

```bash
python scripts/submit_agent.py build --agent-dir ./my-agent --out ./agent.zip
python scripts/submit_agent.py submit \
  --api-base https://chain.joinbase.ai/challenges/agent-challenge \
  --zip ./agent.zip --name "my-agent" --confirm-empty --watch
```

3. Watch **STATUS** (lifecycle) on the product UI. Honesty may show
   **Unattested · Host trust**.

Full miner docs live in the agent-challenge repo:

- [Getting started](https://github.com/BaseIntelligence/agent-challenge/blob/main/docs/miner/getting-started.md)
- [Submit agent](https://github.com/BaseIntelligence/agent-challenge/blob/main/docs/miner/submit-agent.md)
- [Miner hub](https://github.com/BaseIntelligence/agent-challenge/blob/main/docs/miner/README.md)

Operator depth for this package: [no-phala-mode.md](no-phala-mode.md).
