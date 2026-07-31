# base threat model

This document states the **honest** security claim and the properties we deliberately do **not** claim. Overclaiming is treated as a project failure mode.

Frozen contracts: [`BUNDLE_SPEC.md`](./BUNDLE_SPEC.md), [`AGENT_CHALLENGE.md`](./AGENT_CHALLENGE.md).  
Architecture map: [`ARCHITECTURE.md`](./ARCHITECTURE.md).

CI enforces that §1 matches plan decision **D19** word for word (`cargo run -p xtask -- external-docs-check`).

---

## 1. D19 — honest security claim (verbatim)

The following paragraph is copied from plan decision D19 and MUST remain byte-identical to that decision (modulo a single trailing newline). Do not paraphrase in this section.

base guarantees *no equivocation between validators* and *no undetected deviation by the gateway from the owner-signed challenge and measurement artifacts*. It does **not** guarantee (i) that a challenge's scores are honest, (ii) that the owner is honest — the owner signs the trust roots and runs the gateway, so a malicious owner can authorize a dishonest challenge or a backdoored measurement, (iii) completeness beyond what D24 provides, nor (iv) **chain-anchored, third-party-auditable non-equivocation** — per D5 the property is peer-consensus plus local evidence, verifiable by the participating validators and not by an outside observer after the fact.

---

## 2. What attestation does **not** prove (D11)

Env integrity is scoped honestly:

- We prove the `allowed_envs` **name list** via compose-hash.
- We use a `LAUNCH_TOKEN` whose **hash** is in the measured compose.
- We do **not** claim env **values** are verified.
- Secrets are **mounted files**, never measured env values.

dstack does not measure env values. Any doc or pitch that says "TEE proves all secrets" is false for this system.

Related attestation bounds (see also AGENT_CHALLENGE and attest policy crates):

| Outcome | Meaning |
|---------|---------|
| Cryptographic failure | **Reject** |
| Verifier / collateral outage | **Park** (no attestation credit this epoch; never carries prior `Verified` forward) |
| `report_data` binding | Epoch, netuid, miner key, nonce, validator hotkey (D10) — not "the agent is smart" |

Attestation proves **which measured code** answered a **fresh, bound** challenge for this epoch. It does not prove score honesty (D19(i)).

---

## 3. What non-equivocation does **not** rest on (D5)

**Non-equivocation does NOT use the on-chain weight payload.**

`WeightsTlockPayload` is frozen to `{hotkey, uids, values, version_key}`:

- There is **no field** for a merkle root.
- `version_key` is 64 bits, far too small for a 256-bit root.

Non-equivocation therefore rests on:

1. **(a)** In-epoch signed peer root exchange over hotkey-authenticated HTTPS.
2. **(b)** Every validator durably persisting the signed bundle plus all peer root statements as **local evidence**.
3. **(c)** An on-chain announcement via the commitments pallet is **optional and conditional** — only if metadata snapshot confirms the pallet on the target network. The design does not depend on it.

**State this weakening loudly:** non-equivocation here is *peer-consensus plus local evidence*, **not** chain-anchored auditability.

Consequences:

- There is no public `(epoch → bundle_root)` anchor.
- A third party cannot verify after the fact from chain alone.
- Local evidence can be deleted by whoever holds it.
- A fully colluding set of reachable validators could agree on one root with no public counter-evidence.

**The merkle root is NOT committed in the on-chain weight payload.** Do not re-add it. SCALE would reject extra bytes; the field does not exist.

---

## 4. Owner-concentration caveat (R12)

**The owner is the trust root AND the gateway operator.**

This is not solvable inside the current design. It is bounded by:

- D19 (owner honesty out of scope).
- Trust-root rotation as a **signed, reviewable release**, never a hot push (D21).
- Dual-accept window so validators can adopt `v(n+1)` beside `v(n)` for `rotation_epochs` (default 3).

Future work (not claimed here): multi-sig owner keys, transparency log for trust-root releases.

A malicious owner can authorize a dishonest challenge or a backdoored measurement. Validators will still agree with each other and with the owner-signed artifacts. That is **not** a bug relative to D19; it is the stated trust boundary.

---

## 5. Assets and adversaries (summary)

| Asset | Primary protection |
|-------|--------------------|
| Weight vector integrity among honest validators | Bundle verify + recompute + peer roots (D4/D6/D18/D24) |
| Challenge key provenance | Local owner-signed `challenges.toml` only (D18) |
| Emission shares | Same trust root; gateway copy must match (D23) |
| Participant completeness | Validator-derived expected set (D24) |
| Miner code identity / liveness | TDX quote + D10 `report_data` + measurements allowlist |
| Operator secrets | age-encrypted files, mode 0600, never cloud-init / TF state (R11) |
| Host docker.sock | Only on `socket-proxy` with method allowlist |

| Adversary | Expected residual risk |
|-----------|------------------------|
| Deviant gateway | Detected by validators that verify local roots + recompute (task 48 class A/B) |
| Forged challenge key in gateway DB | Rejected: key absent from local trust root (D18) |
| Censorship / set shrinking | Rejected: incompleteness / proper-subset (D24) |
| Eclipsed validator | `Degraded`, no submit below `min_peer_sample` (D26) |
| Compromised challenge sk | Verifiable garbage; quarantine + rotation (R10, D6, D21); honesty not restored |
| Malicious owner | In scope of D19(ii) / R12 — not eliminated |
| Colluding validator set deleting evidence | D19(iv) / D5 — no public anchor |

---

## 6. Operational risks called out in the plan

| ID | Risk | Mitigation (claimed) |
|----|------|----------------------|
| R9 | Gateway death takes down registry, proxy, bundle serving | `restart: unless-stopped` + healthcheck; manual failover runbook. **HA not claimed.** |
| R12 | Owner = trust root + gateway operator | D19 + signed rotation releases |
| R4 | Zero emission possible | Extrinsic success + revealed weights match recompute is pass; emission is not |

---

## 7. Doc hygiene

- Never claim merkle is on-chain in the weight payload.
- Never claim owner honesty or public third-party auditability of non-equivocation.
- Never put secrets, tokens, mnemonics, or private keys in this tree.
- Miner docs must carry the same `protocol_version` badge as `bundle::PROTOCOL_VERSION` (CI-gated).
