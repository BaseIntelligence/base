# Concepts (Prism miners)

Short mental model. Day-1 submit stays on [Getting started](getting-started.md).

## What Prism is

Prism is a BASE **research lab** challenge on https://joinbase.ai:

- **Norm:** try **new architectures** (`torch.nn.Module` under the AST sandbox).
- **Goal:** find **more performant** learners for the locked FineWeb-Edu target under
  fair **challenge-owned** re-execution.
- You submit **two scripts** (`architecture.py` + `training.py`). The challenge owns
  the dataset, forced random init, and the metric.

You do not bring FineWeb weights or self-reported scores. Miner-written manifests are
ignored.

## Emission vs scientific surfaces

| Surface | Role | Ranking |
|---------|------|---------|
| **Emission crown** (leaderboard → raw weights) | Subnet reward eligibility | **Held-out / generalization primary**, prequential bits-per-byte **secondary** |
| **Official Comparison / multimetric / Complete View** | Published **scientific** research grade | Multi-axis (held-out, bpb, long-ctx, reasoning, polar honesty) |

Multimetric scorecard `multimetric.v1.1` and Complete View do **not** silently replace
the emission scalar. Wall-clock **never** ranks emission.

Deep dives:

- [Scoring](../../../packages/challenges/prism/docs/scoring.md) — emission metric  
- [Official Comparison](../../../packages/challenges/prism/docs/official-comparison.md) — science grade  
- [Overview](../../../packages/challenges/prism/docs/overview.md) — lab identity  

## Dual param ladder

| Stage | Cap | Role |
| --- | ---: | --- |
| Explore / provisional | **124M** | Default continuous thrash; may provisional-crown |
| Promote / final | **350M** | Confirm or revoke provisional crown on same pin |

Start under 124M (`examples/tiny-1m`, `examples/mamba-tiny`, other seed families). Promote
only to confirm durable claims.

## Deterministic admission (no LLM gateway)

After static sandbox gates, Prism runs **deterministic** similarity / anti-cheat
admission. There is **no** Base LLM gateway hard gate on Prism. Gateway language in
older posts is obsolete.

## Provider trust and NO-TEE (Prism product)

Prism does **not** ship a TEE verifier package and does **not** require TEE evidence to
finalize production scores.

| Label | Meaning |
| --- | --- |
| **PROVIDER_TRUST** | Operators trust Lium/Targon compute; no Prism crypto TEE path |
| **IMAGE_PIN** | Pinned worker/eval image digest; elevates effective tier to max **1** |
| **LAB-GPU** | Fair CUDA lab scores under Official Comparison |
| **REAL-PROVIDER TEE** | **Retired for Prism product** (do not implement) |

Agent Challenge Phala/KR attestation is a **different** challenge and is out of scope
here.

## How rewards leave Prism

```text
You  --signed ZIP-->  chain.joinbase.ai bridge
                           |
                           v
                    challenge-prism (admit + re-exec + score)
                           |
                    raw hotkey weights push
                           v
                    BASE master aggregation
                    (absolute emission shares; Prism **50%** default)
                           |
                    GET /v1/weights/latest
                           v
                    validators set_weights (their wallets)
```

BASE never calls `set_weights` on the master. Missing or unscored challenges burn their
absolute share (uid0 policy on seal).

## Two-tier ownership inside Prism

Architecture pool **0.50** / training-variant pool **0.50** of Prism’s emission slice
(both use the emission rank metric). See [Scoring](../../../packages/challenges/prism/docs/scoring.md).

## Related

- [Getting started](getting-started.md)
- [Troubleshooting](troubleshooting.md)
- [Miner hub](README.md)
- [Security](../../../packages/challenges/prism/docs/security.md)
- BASE miner hub (emission 50/50 + multi-challenge honesty):
  https://github.com/BaseIntelligence/base/tree/main/docs/miner
- Sibling challenge (agents, attestation advanced):
  https://github.com/BaseIntelligence/agent-challenge/tree/main/docs/miner
