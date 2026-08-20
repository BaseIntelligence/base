<!-- protocol_version: 1 -->

# External miner docs

**Bundle `protocol_version`:** `1`  
**Miner path:** HTTP submit only — **no Phala/CVM**

This badge must match `bundle::PROTOCOL_VERSION` in crate `bundle`.  
CI gate: `cargo run -p xtask -- external-docs-check`.

Agent-v1 / Phala CVM / hypertraining miner paths are **removed**. Miners submit
over HTTP to the live challenges:

| Challenge | `challenge_id` | Scoring | Guide | Public miner repo |
|-----------|----------------|---------|-------|-------------------|
| Design | `design` | `challenge_scoring_version` **2** (daily share ≥2 wins + agentic) | [design.md](./design.md) | [BaseIntelligence/design-challenge](https://github.com/BaseIntelligence/design-challenge) |
| Prism | `prism` | `challenge_scoring_version` **4** (G2 public-suite benchmarks) | [prism.md](./prism.md) | [BaseIntelligence/prism](https://github.com/BaseIntelligence/prism) |

Do **not** conflate version axes:

| Axis | Value | Meaning |
|------|-------|---------|
| Bundle `protocol_version` | **1** | Leaf / merkle / weight bytes ([`BUNDLE_SPEC.md`](../BUNDLE_SPEC.md)) |
| Design scoring | **1** | Agentic anti-cheat + admin winners 1\|2 ([`DESIGN_CHALLENGE.md`](../DESIGN_CHALLENGE.md)) |
| Prism scoring | **2** | Pure bpb + agentic/AST/metrics anti-cheat ([`PRISM.md`](../PRISM.md)) |

| Page | Topic |
|------|-------|
| [design.md](./design.md) | Design harness (`agent.py` + `pyproject.toml`) HTTP submit |
| [examples/design-baseline/](./examples/design-baseline/) | Reference design miner (`llm.chat` → required HTML pages) |
| [prism.md](./prism.md) | Prism AutoModel patch (`automodel.base` + `automodel.patch`) HTTP submit |
| [examples/dense-1b/](./examples/dense-1b/) | Reference Prism miner (dense ~975M, ZeRO-1) |
| [troubleshoot.md](./troubleshoot.md) | Common HTTP / quota / scoring failures |

Normative contracts:

- Design freeze: [`../DESIGN_CHALLENGE.md`](../DESIGN_CHALLENGE.md)
- Prism: [`../PRISM.md`](../PRISM.md) + [`../PRISM_RECIPE.md`](../PRISM_RECIPE.md)
- Bundle bytes: [`../BUNDLE_SPEC.md`](../BUNDLE_SPEC.md)
- Threat claim (D19): [`../THREAT_MODEL.md`](../THREAT_MODEL.md) §1

## Gateway base URL

Production/staging miners call the **gateway** reverse proxy:

```text
https://<gateway>/challenge/design/...
https://<gateway>/challenge/prism/...
```

Local smoke (host ports from `env-local.yml`):

```bash
curl -sS http://127.0.0.1:28093/health   # design-challenge
curl -sS http://127.0.0.1:28092/health   # prism-challenge
```

Never paste mnemonics or challenge signing keys into miner clients. Hotkeys are
public 64-hex identifiers only.
