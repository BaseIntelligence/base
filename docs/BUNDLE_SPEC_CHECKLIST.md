# BUNDLE_SPEC checklist (task 8)

Maps each required pin letter **(a)–(l)** from plan item 8 to a section heading in
[`BUNDLE_SPEC.md`](./BUNDLE_SPEC.md).

CI: `cargo run -p xtask -- spec-check` fails if any letter marker is missing from `BUNDLE_SPEC.md`.

| Letter | Requirement (plan) | Section heading in BUNDLE_SPEC.md | Anchor marker (must appear in spec) |
|--------|--------------------|-----------------------------------|-------------------------------------|
| (a) | SCALE only; JSON forbidden for hashed/signed bytes | ## 1. Encoding law (a) | `## 1. Encoding law (a)` |
| (b) | protocol_version: u16; schema change = major bump | ## 2. Protocol version (b) | `## 2. Protocol version (b)` |
| (c) | leaf layout, ScoreOrAbsence, sort, EMPTY_ROOT hex | ## 3. Merkle construction (RFC 6962) and leaves (c) | `## 3. Merkle construction (RFC 6962) and leaves (c)` |
| (d) | block_B, block_hash, metagraph_root | ## 4. Bundle body and block pin (d) | `## 4. Bundle body and block pin (d)` |
| (e) | full integer aggregation formula | ## 6. Aggregation formula, algorithm_version = 1 (e) | `## 6. Aggregation formula, algorithm_version = 1 (e)` |
| (f) | emission shares from owner-signed trust root | ## 5. Emission shares from owner-signed trust root (f) | `## 5. Emission shares from owner-signed trust root (f)` |
| (g) | expected participant set derivation + completeness | ## 7. Expected participant set derivation (g) (D24) | `## 7. Expected participant set derivation (g) (D24)` |
| (h) | final vector equality + sha256 of SCALE bytes | ## 8. Final vector comparison (h) | `## 8. Final vector comparison (h)` |
| (i) | distribution endpoints; no last-known-good | ## 9. Distribution and caching (i) | `## 9. Distribution and caching (i)` |
| (j) | DissentV1 + reason_code enum | ## 10. Dissent (j) | `## 10. Dissent (j)` |
| (k) | D19 verbatim; D6 quarantine; D26 peer-sample | ## 11. Security claim, quarantine, peer sample (k) | `## 11. Security claim, quarantine, peer sample (k)` |
| (l) | merkle root NOT in on-chain weight payload | ## 12. On-chain weight payload: no merkle root (l) (D5) | `## 12. On-chain weight payload: no merkle root (l) (D5)` |

## Extra pins verified by spec-check

| Pin | Marker substring required in BUNDLE_SPEC.md |
|-----|-----------------------------------------------|
| EMPTY_ROOT hex | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| D19 claim fragment | `no equivocation between validators` |
| WeightsTlockPayload freeze | `WeightsTlockPayload` |
| No last-known-good | `No last-known-good` |
| Hamilton house | `65_535` or `65535` |
| NoScore enum | `NoScoreReasonCode` |
| Dissent enum | `DissentReasonCode` |

## Maintenance

When editing `BUNDLE_SPEC.md` headings, update this table and keep the
`(a)`…`(l)` markers in the heading lines so `xtask spec-check` stays green.
