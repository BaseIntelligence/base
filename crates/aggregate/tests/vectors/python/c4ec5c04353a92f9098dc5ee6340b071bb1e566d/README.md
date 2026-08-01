# Python BASE vectors — upstream `c4ec5c04353a92f9098dc5ee6340b071bb1e566d`

**Python is the authority.** These vectors were originally captured as "characterization
only" under plan D16, when `BUNDLE_SPEC` §6 was treated as the source of truth. That was
overridden: the Rust gateway must be a drop-in replacement for the live Python service at
<https://chain.joinbase.ai>, so `base.master.aggregator.aggregate_challenge_weights` now
defines the served vector and `crates/aggregate::python` is a bit-for-bit port of it.

What used to be listed as "divergence vs BUNDLE_SPEC §6" is now the
`served_algorithm_specification` block in each header. `BUNDLE_SPEC` §6 (u128 + Hamilton,
no burn, empty vector on all-zero) describes the *other* algorithm in this crate
([`aggregate::aggregate`]), which the bundle/validator path still uses.

The aggregator source is byte-identical between `c4ec5c04…` and the current
`origin/main` `8249563774ee2e71c41ae2cfac182ff32aa35dd1`; the newer directory next to this
one holds the wider edge-case suite. Both are replayed by
`tests/python_vectors.rs`, and both must reproduce **exactly** (uids, `f64` bit patterns,
`hotkey_weights` order, and the `round_ties_even` u16 vector).

## Files

- `01_single_challenge_two_miners_equal.json`
- `02_two_challenges_absolute_shares_with_burn.json`
- `03_spec_like_scores_50_30_20_one_challenge.json`
- `04_zero_miner_fallback_burn.json`
- `05_unknown_hotkey_share_burns.json`

## Encoding

- `header` — authority note, upstream sha, served-algorithm specification, case name
- `inputs` — `challenge_results`, `hotkey_to_uid`, `kwargs`; **JSON object key order is
  significant** (it is the Python `dict` insertion order the algorithm observes)
- `python_float_output` — `uids`, `weights`, `hotkey_weights` straight from `FinalWeights`
- `expected_vector` — `[[uid, weight_u16], …]`, `weight_u16 = round(w * 65535)` with
  Python's round-half-to-**even**
