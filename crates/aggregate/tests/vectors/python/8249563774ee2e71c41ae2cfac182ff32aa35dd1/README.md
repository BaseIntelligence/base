# Python BASE vectors — upstream `8249563774ee2e71c41ae2cfac182ff32aa35dd1`

**Python is the authority.** `base.master.aggregator.aggregate_challenge_weights` at BASE
`origin/main` `8249563774ee2e71c41ae2cfac182ff32aa35dd1` is the code serving
<https://chain.joinbase.ai>. `crates/aggregate::python` is a bit-for-bit port of it and
must reproduce every field here **exactly** — uids, `f64` bit patterns, `hotkey_weights`
order, and the `round_ties_even` u16 vector. On disagreement, Python wins; this is not
"characterization", it is the specification.

`BUNDLE_SPEC` §6 (u128 `FIXED` + Hamilton, no burn-to-uid-0, empty vector on all-zero)
describes the *other* algorithm in this crate ([`aggregate::aggregate`]), still used by
the bundle/validator path. The two are not expected to agree.

## Regenerating

```sh
cd /root/prism-compute-plane/base && \
  ./.venv/bin/python /root/gbase/crates/aggregate/tests/support/gen_vectors.py
```

`BASE_SRC` overrides the BASE package root and `VECTOR_OUT` the output directory. The
generator lives at `crates/aggregate/tests/support/gen_vectors.py` and is committed so the
vectors are reproducible. Ground truth interpreter: **CPython 3.12.3**.

## Coverage

| Case | What it pins |
|------|--------------|
| `06_zero_miner_min_allowed_{1,2,3}` | zero-miner fallback padding for each `min_allowed_weights` |
| `07_max_weight_limit_20000_pads_to_four` | `ceil(1 / (20000/65535)) = 4` forces extra padding uids |
| `08_max_weight_limit_9000_pads_to_eight` | same at a tighter cap (8 uids) |
| `09_zero_miner_error_not_enough_uids` | `ZeroMinerWeightError` — too few candidates |
| `10_zero_miner_error_max_weight_limit_too_low` | `ZeroMinerWeightError` — cap demands 66 uids |
| `11_over_allocation_scaled_back` | `sum(emission_percent) = 180 > 100` scales back, no burn |
| `12_over_allocation_three_way_uneven` | over-allocation with uneven intra-challenge weights |
| `13_hotkey_on_uid_zero_burns` | hotkey mapping to uid 0 is dropped and its mass burns |
| `14_unknown_hotkey_burns` | hotkey absent from `hotkey_to_uid` is dropped and its mass burns |
| `15_clean_weights_filters_bad_values` | negative / zero / NaN / ±inf weights filtered by `_clean_weights` |
| `16_all_weights_invalid_falls_back_to_zero_miner` | everything filtered → zero-miner burn |
| `17`–`20 order_fidelity_*` | the same data in four insertion orders |
| `21_duplicate_slug_last_emission_wins` | duplicate slugs collapse in `frac`, last value wins |
| `22_negative_and_zero_emission_percent` | `max(pct, 0.0)` clamp and the `share <= 0 → continue` skip |
| `23_not_ok_challenge_ignored` | `ok = false` challenges contribute nothing (their share burns) |
| `24_many_miners_uneven_shares` | 12 uids across 3 challenges with a shared hotkey |
| `25_half_even_rounding_pins` | `0.5 * 65535 = 32767.5 → 32768` (round-half-to-even) |
| `26_burn_below_eps_not_added` | residual burn at/below `EPS` is not added |

### Order fidelity

Cases `17`–`20` feed identical data in four different insertion orders. Under CPython
3.12 the *weights* come out identical in all four (builtin `sum()` is Neumaier-compensated,
so it is far more order-stable than a naive fold), but `hotkey_weights` **key order
differs per case** — it is `hotkey_scores` first-appearance order. Rust reproduces each
order exactly; a `BTreeMap`-based port would fail these.

### u16 sums

`chain_u16_sum` is recorded per case and is **not always 65535**: Python applies no
post-rounding renormalisation, so independent `round_ties_even` of each weight yields
65534, 65535 or 65536. Roughly 25–30 % of randomized cases land off 65535 (see
`tests/differential.rs`). This crate deliberately does **not** correct it — that would be
a behavioural divergence from the authority. Any chain-side normalisation belongs in the
caller.

## Encoding

- `header` — authority note, upstream sha, served-algorithm specification, case name
- `inputs` — `challenge_results`, `hotkey_to_uid`, `kwargs`; **JSON object key order is
  significant** (it is the Python `dict` insertion order the algorithm observes)
- `python_float_output` — `uids`, `weights`, `hotkey_weights` straight from `FinalWeights`
- `python_error` — verbatim `ZeroMinerWeightError` message when Python raised
- `expected_vector` — `[[uid, weight_u16], …]`, `weight_u16 = round(w * 65535)`
- `chain_u16_sum` — sum of the encoded vector (see above)
