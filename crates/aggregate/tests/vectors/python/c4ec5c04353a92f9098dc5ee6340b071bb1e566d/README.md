# Python BASE characterization vectors (D16)

**Upstream BASE commit:** `c4ec5c04353a92f9098dc5ee6340b071bb1e566d`  
**Source:** `/root/prism-compute-plane/base` → `base.master.aggregation` (imports `aggregate_challenge_weights` from `base.master.aggregator`)  
**Captured:** throwaway `/tmp` script (not committed)

## Authority

These vectors characterize the **Python float/JSON aggregation path**.  
They are **not** consensus authority.

Where they disagree with [`docs/BUNDLE_SPEC.md`](../../../../../../docs/BUNDLE_SPEC.md) **§6 Aggregation formula (e)**, **the spec wins** (plan D16 / BUNDLE_SPEC line 10).

## Known divergences vs BUNDLE_SPEC §6

- Python path: float normalize per challenge, absolute emission_percent/100, burn remainder to uid 0.
- BUNDLE_SPEC §6: u128 FIXED=1e12, share bps, Hamilton largest-remainder to HOUSE=65535, no burn-to-uid0, empty Vec on all-zero (no-submit).
- Python zero-miner: build_zero_miner_weights self-burn; §6.5: empty final_vector.
- Python over-alloc: renormalize shares to 1.0; §6 requires sum bps == 10000 well-formed.
- Characterization only (D16). Spec wins on disagreement.

### §6.8 worked example

| Miner | UID | raw |
|-------|-----|-----|
| A | 1 | 50 |
| B | 2 | 30 |
| C | 3 | 20 |

Spec Hamilton result: `[(1, 32768), (2, 19660), (3, 13107)]` sum=65535.

Case `03_spec_like_scores_50_30_20_one_challenge.json` feeds the same ratios through Python floats and records whether `round(w*65535)` matches (it generally will for exact dyadic ratios, but the *algorithm* still differs: float normalize vs u128 FIXED + Hamilton).

## Encoding

Each `*.json`:

- `header`: SHA, matches_spec bool, divergence notes
- `inputs`: challenge_results + hotkey_to_uid (snapshot set)
- `python_float_output`: raw floats from Python
- `expected_vector`: `[[uid, weight_u16], ...]` integers with `weight_u16 = round(float * 65535)`

## Files

- `01_single_challenge_two_miners_equal.json`
- `02_two_challenges_absolute_shares_with_burn.json`
- `03_spec_like_scores_50_30_20_one_challenge.json`
- `04_zero_miner_fallback_burn.json`
- `05_unknown_hotkey_share_burns.json`
