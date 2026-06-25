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
    emissions = normalize_emissions(challenge_results)
    hotkey_scores: defaultdict[str, float] = defaultdict(float)

    for result in challenge_results:
        if not result.ok:
            continue
        emission = emissions.get(result.slug, 0.0)
        for hotkey, weight in normalize_weights(result.weights).items():
            hotkey_scores[hotkey] += emission * weight

    uid_scores: defaultdict[int, float] = defaultdict(float)
    kept_hotkeys: dict[str, float] = {}
    for hotkey, weight in hotkey_scores.items():
        uid = hotkey_to_uid.get(hotkey)
        if uid is None:
            continue
        if uid == 0:
            continue
        uid_scores[uid] += weight
        kept_hotkeys[hotkey] = weight

    total = sum(uid_scores.values())
    if total > 0:
        normalized = {uid: value / total for uid, value in uid_scores.items()}
    else:
        normalized = build_zero_miner_weights(
            min_allowed_weights=min_allowed_weights,
            max_weight_limit=max_weight_limit,
            available_uids=hotkey_to_uid.values(),
        )
        kept_hotkeys = {}

    ordered = sorted(normalized.items())
    return FinalWeights(
        uids=[uid for uid, _ in ordered],
        weights=[weight for _, weight in ordered],
        hotkey_weights=kept_hotkeys,
    )
