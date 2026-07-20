# Concepts (miners)

Short mental model. Details and edge cases stay in architecture and challenge docs.

## What BASE is (for miners)

BASE is a **multi-challenge Bittensor subnet platform**:

- One public **master** API: https://chain.joinbase.ai
- Many **challenge** services (Prism, Agent Challenge, …), each with its own repo, image,
  scoring, and SQLite `/data` volume
- Independent **validators** that fetch the master’s final weight vector and call
  `set_weights` under **their** wallets

BASE does **not**:

- Score your architecture or agent for you (challenges do)
- Host a public LLM gateway for miners
- Call `set_weights` on the master (validators do)

## Traffic model

```text
Miner  --signed request-->  chain.joinbase.ai (BASE proxy / bridge)
                                |
                                v
                         challenge-<slug> service
                                |
                     raw hotkey weights push
                                v
                         master aggregation
                     (absolute emission shares)
                                |
                     GET /v1/weights/latest
                                v
                         validators set_weights
```

Public challenge paths look like:

```text
https://chain.joinbase.ai/challenges/{slug}/...
https://chain.joinbase.ai/v1/challenges/{slug}/...
```

## Absolute emission shares (50 / 50)

Registry field: `emission_percent` (absolute). Production default:

| Challenge | Share |
|-----------|------:|
| Prism (`prism`) | **50%** |
| Agent Challenge (`agent-challenge`) | **50%** |

Policy highlights:

- Shares are **absolute**, not “relative renormalize of whoever reported.”
- Active percents should sum to **100**. Remainder (or a missing scorer side) burns
  according to master burn policy (uid0).
- Each seal snapshots registry shares into the aggregated vector.
- **Wall-clock is never the emission rank key** inside challenges. Prism emission is
  held-out primary / bpb secondary; Agent Challenge uses its own benchmark scores.
- Multimetric / Complete View / research panels on Prism are **scientific grade**, not
  a silent rewrite of the emission scalar.

You can confirm live shares:

```bash
curl -fsS https://chain.joinbase.ai/v1/registry
# each challenge object includes emission_percent (e.g. "50.0000")
```

## Division of responsibility

| Layer | Owns |
|-------|------|
| **BASE** | Public entry, slug routing, bridges, registry, raw-weight ingress, aggregation, final vector read API |
| **Challenge** | Artifact format, signatures, scoring, leaderboards, raw weight push |
| **Validator** | Assignments / verify paths they own, on-chain `set_weights` |
| **Miner** | Wallet, challenge work product, signed submits, optional GPU workers (Prism) |

## Trust labels (honesty)

| Label | Meaning for miners |
|-------|--------------------|
| Prism **NO-TEE** product | Prism does **not** require a Prism-side crypto TEE verifier for score finalization. Integrity levers include deterministic admission, challenge-owned eval, IMAGE_PIN / provider trust on GPU paths. |
| Agent Challenge **attestation** | Separate product path (Phala / KR). Advanced; day-1 submit does not require you to operate TEE research. |
| REAL-PROVIDER TEE PASS | **Not** a Prism day-1 claim. Do not expect marketing “REAL TEE PASS” from ordinary Prism mining. |
| No Base LLM gateway | There is no `BASE_LLM_GATEWAY_*` / `/llm/v1` miner path on Compose master. |

## Hotkey consistency

Raw weights and leaderboards key on **hotkey**. Switching hotkeys mid-challenge forks your
history. Keep one mining hotkey per strategy unless you intentionally run multiple UIDs.

## Optional Prism worker plane

When the master enables `compute.worker_plane_enabled`, Prism may require a **miner-funded
GPU worker** bound to your hotkey (`403 NO_ACTIVE_WORKER` if missing). That is advanced
ops: [worker-plane.md](worker-plane.md). Flag-off networks keep legacy submit behavior.

## Related reading

- [Getting started](getting-started.md)
- [../challenges.md](../challenges.md) — challenge lifecycle on Compose
- [../architecture.md](../architecture.md) — full topology
- Challenge-owned scoring docs in Prism and agent-challenge repositories
