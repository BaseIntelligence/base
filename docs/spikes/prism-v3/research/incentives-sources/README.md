# Incentive-design evidence reviews (sources for appendix 15)

> Non-normative research evidence. Per [`docs/AGENTS.md`](../../../../AGENTS.md),
> `docs/spikes/**` is research/evidence and **never** normative spec: when any
> of it conflicts with [`docs/PRISM.md`](../../../../PRISM.md),
> [`docs/BUNDLE_SPEC.md`](../../../../BUNDLE_SPEC.md), or a pre-registered
> anchor set, the normative doc wins.

These are the underlying evidence reviews cited by
[`../15-incentives-and-landscape.md`](../15-incentives-and-landscape.md). They
are preserved here because that appendix's conclusions — in particular the
copy-EV arithmetic and the live SN56 parameters that the significance-gated
emission mode is calibrated against — depend on primary sources with URLs, and
the appendix alone summarizes rather than reproduces them.

| File | Covers |
|------|--------|
| [`bittensor.md`](bittensor.md) | SN9 scoring pipeline and epsilon implementations, WTA parameters and documented failure modes (model hoarding, measured copying), IOTA's abandonment of WTA, SN37's competition framework, SN56/Gradients boss-round constants, Templar |
| [`nas-competition.md`](nas-competition.md) | NAS benchmarks and the reproducibility crisis, weight-sharing rank-correlation failures, zero-cost proxies, cross-fidelity rank correlation, the Ladder / Thresholdout mechanics, the Kaggle overfitting meta-analysis, best-arm identification, contest theory under noise |
| [`novelty-bounties.md`](novelty-bounties.md) | Novelty measurement and its measured evasion, the full Numerai originality→MMC→TC→MMC history, the Shapley cost wall, funding concentration vs dispersal, lottery funding, registered reports, tea.xyz / RetroPGF / Gitcoin / thanks.dev lineage failures |
| [`calc.py`](calc.py) | Reproducible arithmetic for the copy-EV and detection-threshold figures quoted in appendix 15 |

**Reading order:** start from
[`../15-incentives-and-landscape.md`](../15-incentives-and-landscape.md); come
here only to check a specific claim against its primary source.
