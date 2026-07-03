from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable

from base.schemas.weights import ChallengeWeightsResult, FinalWeights

CHAIN_U16_MAX = 65535
"""Bittensor weights are encoded as u16; 65535/65535 == 1.0 is the full share.

``max_weight_limit`` is supplied in this same u16 space (recon fact for netuid
100 = 65535), so the per-weight cap as a fraction is ``max_weight_limit /
CHAIN_U16_MAX``.
"""

ZERO_MINER_BURN_UID = 0
"""uid 0 == the validator itself, whose coldkey IS the SubnetOwner (recon-facts.md).

A single-entry ``{0: 1.0}`` vector is therefore a self/owner burn that is
on-chain acceptable while no miners have scored.
"""

EPS = 1e-12
"""Float tolerance: residual miner mass / burn at or below this is treated as 0."""


class ZeroMinerWeightError(RuntimeError):
    """No chain-valid weight vector can be built for the zero-miner case.

    Raised instead of returning a vector that would violate
    ``min_allowed_weights`` or ``max_weight_limit`` — the validator must abort
    rather than submit a vector the chain will reject.
    """


def build_zero_miner_weights(
    *,
    min_allowed_weights: int,
    max_weight_limit: int = CHAIN_U16_MAX,
    available_uids: Iterable[int] = (),
    burn_uid: int = ZERO_MINER_BURN_UID,
    allow_burn_self_vote: bool = True,
) -> dict[int, float]:
    """Build a chain-valid fallback weight vector when no miner has scored.

    Branches (derived from on-chain recon, not assumption):
      * ``min_allowed_weights <= 1`` and the burn uid is usable -> keep the
        single-entry self/owner burn ``{burn_uid: 1.0}``.
      * ``min_allowed_weights > 1`` (or a low ``max_weight_limit``) -> pad with
        the burn uid plus other metagraph uids and distribute equally so both
        the minimum-count and the per-weight cap hold.
      * otherwise -> raise :class:`ZeroMinerWeightError` (never submit invalid).
    """
    if max_weight_limit <= 0:
        raise ZeroMinerWeightError(
            f"max_weight_limit={max_weight_limit} admits no positive weight"
        )

    max_fraction = min(max_weight_limit / CHAIN_U16_MAX, 1.0)

    candidates: list[int] = []
    if allow_burn_self_vote:
        candidates.append(burn_uid)
    for uid in sorted(set(available_uids)):
        if uid == burn_uid:
            continue
        candidates.append(uid)

    required = max(int(min_allowed_weights), 1, math.ceil(1.0 / max_fraction))

    if len(candidates) < required:
        raise ZeroMinerWeightError(
            "cannot build a chain-valid zero-miner weight vector: need "
            f"{required} uids (min_allowed_weights={min_allowed_weights}, "
            f"max_weight_limit={max_weight_limit}) but only {len(candidates)} "
            "usable uid(s) available"
        )

    weight = 1.0 / required
    return {uid: weight for uid in candidates[:required]}


def _clean_weights(raw: dict[str, float]) -> dict[str, float]:
    cleaned: dict[str, float] = {}
    for hotkey, value in raw.items():
        weight = float(value)
        if not math.isfinite(weight):
            continue
        if weight <= 0:
            continue
        cleaned[str(hotkey)] = weight
    return cleaned


def normalize_weights(raw: dict[str, float]) -> dict[str, float]:
    cleaned = _clean_weights(raw)
    total = sum(cleaned.values())
    if total <= 0:
        return {}
    return {hotkey: value / total for hotkey, value in cleaned.items()}


def normalize_emissions(results: list[ChallengeWeightsResult]) -> dict[str, float]:
    active = {r.slug: max(float(r.emission_percent), 0.0) for r in results if r.ok}
    total = sum(active.values())
    if total <= 0:
        return {slug: 0.0 for slug in active}
    return {slug: value / total for slug, value in active.items()}


def aggregate_challenge_weights(
    challenge_results: list[ChallengeWeightsResult],
    hotkey_to_uid: dict[str, int],
    *,
    min_allowed_weights: int = 1,
    max_weight_limit: int = CHAIN_U16_MAX,
) -> FinalWeights:
    """Aggregate per-challenge weights with ABSOLUTE emission shares + burn.

    ``emission_percent`` is an absolute share of 100 (not a relative share
    normalized across challenges): an OK challenge with ``emission_percent=e``
    owns ``e/100`` of the whole vector and distributes it across its own miners.
    Any share NOT landing on a real miner burns to :data:`ZERO_MINER_BURN_UID`
    (the subnet owner): the unallocated remainder ``1 - sum(shares)``, challenges
    with no valid miners, and weights mapping to uid 0 / unknown uids. Operators
    who over-allocate (``sum(shares) > 1``) are scaled back to sum exactly 1.0 so
    there is no burn in that case.

    The zero-miner path (no real miner scored at all) is UNCHANGED: it defers to
    :func:`build_zero_miner_weights` for a chain-valid full-burn vector.
    """
    ok_results = [result for result in challenge_results if result.ok]

    frac = {
        result.slug: max(float(result.emission_percent), 0.0) / 100.0
        for result in ok_results
    }
    alloc_total = sum(frac.values())
    if alloc_total > 1.0:
        frac = {slug: value / alloc_total for slug, value in frac.items()}

    hotkey_scores: defaultdict[str, float] = defaultdict(float)
    for result in ok_results:
        share = frac.get(result.slug, 0.0)
        if share <= 0.0:
            continue
        for hotkey, weight in normalize_weights(result.weights).items():
            hotkey_scores[hotkey] += share * weight

    uid_scores: defaultdict[int, float] = defaultdict(float)
    kept_hotkeys: dict[str, float] = {}
    for hotkey, score in hotkey_scores.items():
        uid = hotkey_to_uid.get(hotkey)
        if uid is None or uid == ZERO_MINER_BURN_UID:
            continue
        uid_scores[uid] += score
        kept_hotkeys[hotkey] = score

    miner_total = sum(uid_scores.values())
    if miner_total <= EPS:
        normalized = build_zero_miner_weights(
            min_allowed_weights=min_allowed_weights,
            max_weight_limit=max_weight_limit,
            available_uids=hotkey_to_uid.values(),
        )
        kept_hotkeys = {}
    else:
        final_scores: dict[int, float] = dict(uid_scores)
        burn = 1.0 - miner_total
        if burn > EPS:
            final_scores[ZERO_MINER_BURN_UID] = (
                final_scores.get(ZERO_MINER_BURN_UID, 0.0) + burn
            )
        total = sum(final_scores.values())
        normalized = {uid: value / total for uid, value in final_scores.items()}

    ordered = sorted(normalized.items())
    return FinalWeights(
        uids=[uid for uid, _ in ordered],
        weights=[weight for _, weight in ordered],
        hotkey_weights=kept_hotkeys,
    )
