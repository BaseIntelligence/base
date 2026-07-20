"""Dual parameter ladder for PRISM Research Lab emission admission (VAL-RESLAB-003+).

Two admitted stages replace the legacy single 150M emission cap:

| Stage      | Cap           | Role                                               |
| ---------- | ------------- | -------------------------------------------------- |
| ``explore``  | 124_000_000 | Cheap continuous exploration; may provisional-crown |
| ``promote``  | 350_000_000 | Same package pin re-eval; confirms or revokes       |

Stage labels ride on scores/manifests as ``param_ladder_stage`` (values ``explore`` /
``promote``). Provisional-crown eligibility is explore-stage only; promote-path hooks
(confirm/revoke state machine) live on top of these types — full weights wiring can
complete in the weights feature when split cleanly.

Official Comparison default pin follows the **promote** ceiling so dual-pin honesty is
documentable (explore runs may still pin 124M explicitly).
"""

from __future__ import annotations

from typing import Any, Literal

ParamLadderStage = Literal["explore", "promote"]

EXPLORE_MAX_PARAMETERS = 124_000_000
PROMOTE_MAX_PARAMETERS = 350_000_000

# Canonical stage label strings (scores / manifests / Official pin).
STAGE_EXPLORE: ParamLadderStage = "explore"
STAGE_PROMOTE: ParamLadderStage = "promote"

PARAM_LADDER_STAGE_FIELD = "param_ladder_stage"
PARAM_LADDER_CAP_FIELD = "param_ladder_cap"
PARAM_LADDER_PROVISIONAL_CROWN_ELIGIBLE_FIELD = "provisional_crown_eligible"

# All admitted stage labels in fixed order (explore first).
PARAM_LADDER_STAGES: tuple[ParamLadderStage, ...] = (STAGE_EXPLORE, STAGE_PROMOTE)

# Map stage → hard max_parameters.
PARAM_LADDER_CAPS: dict[ParamLadderStage, int] = {
    STAGE_EXPLORE: EXPLORE_MAX_PARAMETERS,
    STAGE_PROMOTE: PROMOTE_MAX_PARAMETERS,
}

# Official Comparison default pin uses the promote ceiling; explore pins remain explicit.
OFFICIAL_DEFAULT_PARAM_CAP = PROMOTE_MAX_PARAMETERS
OFFICIAL_DEFAULT_PARAM_STAGE: ParamLadderStage = STAGE_PROMOTE

# Backward-compat alias: single historical constant that once meant "the" emission cap.
# Paths still importing a singular name should prefer stage-aware helpers below.
LEGACY_SINGLE_PARAM_CAP = 150_000_000


def max_parameters_for_stage(stage: ParamLadderStage | str) -> int:
    """Return the hard param cap for an admitted ladder stage.

    Raises ``ValueError`` on unknown stage labels (fail closed).
    """
    key = str(stage).strip().lower()
    if key not in PARAM_LADDER_CAPS:
        raise ValueError(
            f"unknown param ladder stage {stage!r}; expected one of {list(PARAM_LADDER_STAGES)}"
        )
    return PARAM_LADDER_CAPS[key]  # type: ignore[index]


def normalize_param_ladder_stage(
    stage: ParamLadderStage | str | None,
    *,
    default: ParamLadderStage = STAGE_EXPLORE,
) -> ParamLadderStage:
    """Coerce a caller/config stage label to a canonical admitted stage.

    ``None`` / blank → ``default`` (explore for continuous emission admission).
    """
    if stage is None:
        return default
    key = str(stage).strip().lower()
    if not key:
        return default
    if key not in PARAM_LADDER_CAPS:
        raise ValueError(
            f"unknown param ladder stage {stage!r}; expected one of {list(PARAM_LADDER_STAGES)}"
        )
    return key  # type: ignore[return-value]


def resolve_max_parameters(
    *,
    stage: ParamLadderStage | str | None = None,
    max_parameters: int | None = None,
    default_stage: ParamLadderStage = STAGE_EXPLORE,
) -> tuple[ParamLadderStage, int]:
    """Resolve ``(stage, cap)`` from optional stage + optional numeric override.

    Rules:
    - Explicit ``max_parameters`` wins for the numeric cap; stage still defaults to
      ``default_stage`` unless provided (allows both lab overrides and dual labels).
    - When only stage is set, the stage's locked cap is used.
    - When neither is set, return ``(default_stage, cap(default_stage))``.
    """
    resolved_stage = normalize_param_ladder_stage(stage, default=default_stage)
    if max_parameters is not None:
        cap = int(max_parameters)
        if cap <= 0:
            raise ValueError(f"max_parameters must be positive, got {max_parameters!r}")
        return resolved_stage, cap
    return resolved_stage, max_parameters_for_stage(resolved_stage)


def stage_for_param_count(
    param_count: int,
    *,
    prefer: ParamLadderStage | str | None = None,
) -> ParamLadderStage:
    """Pick the smallest ladder stage whose cap admits ``param_count``.

    Counts above the promote ceiling raise ``ValueError`` (fail closed for admission).
    When ``prefer`` is set and its cap already admits the count, that stage is kept
    (so an explicit promote-stage pin stays labeled promote even under 124M).
    """
    count = int(param_count)
    if count < 0:
        raise ValueError(f"param_count must be non-negative, got {param_count!r}")
    if prefer is not None:
        preferred = normalize_param_ladder_stage(prefer)
        if count <= max_parameters_for_stage(preferred):
            return preferred
    if count <= EXPLORE_MAX_PARAMETERS:
        return STAGE_EXPLORE
    if count <= PROMOTE_MAX_PARAMETERS:
        return STAGE_PROMOTE
    raise ValueError(f"param_count {count:,} exceeds promote ladder cap {PROMOTE_MAX_PARAMETERS:,}")


def is_within_stage_cap(param_count: int, stage: ParamLadderStage | str) -> bool:
    """True when ``param_count`` is ≤ the stage's hard cap."""
    return int(param_count) <= max_parameters_for_stage(stage)


def provisional_crown_eligible(
    *,
    stage: ParamLadderStage | str,
    param_count: int | None = None,
    score_valid: bool = True,
) -> bool:
    """Explore-stage provisional crown eligibility gate (VAL-RESLAB-004 hook).

    A qualifying explore-stage run may provisionally occupy architecture/training crowns.
    Promote-stage scores are confirmed/revoked on the promote path (not provisional).
    Full revoke/confirm state machine may complete in the weights feature; this helper is
    the admission/type surface tests lock here.
    """
    if not score_valid:
        return False
    resolved = normalize_param_ladder_stage(stage)
    if resolved != STAGE_EXPLORE:
        return False
    if param_count is not None and not is_within_stage_cap(param_count, STAGE_EXPLORE):
        return False
    return True


def promote_path_decision(
    *,
    provisional_stage: ParamLadderStage | str,
    promote_stage: ParamLadderStage | str,
    promote_valid: bool,
    promote_beats_provisional: bool | None = None,
) -> Literal["confirm", "revoke", "ineligible"]:
    """Promote-path state-machine hook (VAL-RESLAB-005 surface).

    - ``confirm``: promote-stage valid and (if compared) beats or ties provisional
    - ``revoke``: promote-stage invalid or loses fair compare vs provisional
    - ``ineligible``: stages are not the expected explore→promote pair

    Durable crown transitions use :func:`resolve_crown_status_transition` so weights /
    ``get_weights`` never keep a revoked provisional winner.
    """
    provisional = normalize_param_ladder_stage(provisional_stage)
    promote = normalize_param_ladder_stage(promote_stage)
    if provisional != STAGE_EXPLORE or promote != STAGE_PROMOTE:
        return "ineligible"
    if not promote_valid:
        return "revoke"
    if promote_beats_provisional is False:
        return "revoke"
    return "confirm"


# Durable crown_status values stored on architecture_families / training_variants.
CrownStatus = Literal["none", "provisional", "confirmed", "revoked"]
CROWN_STATUS_NONE: CrownStatus = "none"
CROWN_STATUS_PROVISIONAL: CrownStatus = "provisional"
CROWN_STATUS_CONFIRMED: CrownStatus = "confirmed"
CROWN_STATUS_REVOKED: CrownStatus = "revoked"


def resolve_package_pin(
    *,
    family_hash: str,
    package_pin: str | None = None,
    architecture_source_hash: str | None = None,
) -> str:
    """Stable package/family pin used to bind explore provisional → promote confirm/revoke.

    Prefer an explicit package pin (source/bundle digest). Fall back to family_hash so same
    architecture family promotes under the locked pin rules even when a separate package
    digest is unavailable.
    """
    for candidate in (package_pin, architecture_source_hash, family_hash):
        if candidate is None:
            continue
        text = str(candidate).strip()
        if text:
            return text
    return str(family_hash)


def resolve_crown_status_transition(
    *,
    previous_status: str | None,
    previous_stage: ParamLadderStage | str | None,
    previous_pin: str | None,
    incoming_stage: ParamLadderStage | str,
    incoming_pin: str,
    score_valid: bool,
    score_beats_previous: bool | None = None,
) -> CrownStatus:
    """Durable crown_status transition for one family/variant row (VAL-RESLAB-004/005).

    Rules:
    - Ineligible / non-positive scores never install a crown (stay previous, or ``none``).
    - Explore + valid → ``provisional`` (may populate weights map).
    - Promote + valid + same package/family pin as a provisional crown:
        - confirm when promote wins/ties fair compare (default when compare is None or True)
        - revoke when promote loses fair compare
    - Promote invalid (or crown-score 0) against matching provisional pin → ``revoked``.
    - Promote with no prior provisional crown may still reverse a dead provisional via pin
      match; without a matching pin, a valid promote installs ``confirmed`` for durable
      promote-first packages (no dead provisional left behind).
    - Confirmed crowns stay confirmed when a later eligible score arrives on the same pin;
      a failed promote on a confirmed/pinned tooth revokes.
    """
    prev_raw = str(previous_status or CROWN_STATUS_NONE).strip().lower() or CROWN_STATUS_NONE
    if prev_raw == CROWN_STATUS_PROVISIONAL:
        prev = CROWN_STATUS_PROVISIONAL
    elif prev_raw == CROWN_STATUS_CONFIRMED:
        prev = CROWN_STATUS_CONFIRMED
    elif prev_raw == CROWN_STATUS_REVOKED:
        prev = CROWN_STATUS_REVOKED
    else:
        prev = CROWN_STATUS_NONE
    stage = normalize_param_ladder_stage(incoming_stage)
    pin = str(incoming_pin or "").strip()
    prev_pin = str(previous_pin or "").strip()
    same_pin = bool(pin) and bool(prev_pin) and pin == prev_pin

    if stage == STAGE_EXPLORE:
        if not score_valid:
            if prev in (CROWN_STATUS_PROVISIONAL, CROWN_STATUS_CONFIRMED):
                return prev
            return CROWN_STATUS_NONE
        return CROWN_STATUS_PROVISIONAL

    # promote stage
    if not score_valid:
        if same_pin and prev in (
            CROWN_STATUS_PROVISIONAL,
            CROWN_STATUS_CONFIRMED,
            CROWN_STATUS_NONE,
        ):
            return CROWN_STATUS_REVOKED
        if prev == CROWN_STATUS_CONFIRMED:
            return CROWN_STATUS_CONFIRMED
        if same_pin:
            return CROWN_STATUS_REVOKED
        return prev

    # valid promote
    if same_pin and prev == CROWN_STATUS_PROVISIONAL:
        decision = promote_path_decision(
            provisional_stage=STAGE_EXPLORE,
            promote_stage=STAGE_PROMOTE,
            promote_valid=True,
            promote_beats_provisional=score_beats_previous,
        )
        if decision == "confirm":
            return CROWN_STATUS_CONFIRMED
        return CROWN_STATUS_REVOKED
    if same_pin and prev == CROWN_STATUS_CONFIRMED:
        # Later promote on same pin keeps confirmed when still competitive.
        if score_beats_previous is False:
            return CROWN_STATUS_REVOKED
        return CROWN_STATUS_CONFIRMED
    # No prior provisional, or different pin: durable promote installs confirmed when valid.
    return CROWN_STATUS_CONFIRMED


def crown_status_is_weight_eligible(status: str | None) -> bool:
    """True when crown_status may populate the emission weight map."""
    key = str(status or CROWN_STATUS_NONE).strip().lower() or CROWN_STATUS_NONE
    return key in {CROWN_STATUS_NONE, CROWN_STATUS_PROVISIONAL, CROWN_STATUS_CONFIRMED}


def ladder_labels(
    stage: ParamLadderStage | str,
    *,
    param_count: int | None = None,
    score_valid: bool = True,
    max_parameters: int | None = None,
) -> dict[str, Any]:
    """Stage labels for scores / manifests / Official pin honesty.

    Always includes ``param_ladder_stage`` and ``param_ladder_cap``. When
    ``param_count`` is known, also records provisional-crown eligibility.
    """
    resolved = normalize_param_ladder_stage(stage)
    cap = int(max_parameters) if max_parameters is not None else max_parameters_for_stage(resolved)
    payload: dict[str, Any] = {
        PARAM_LADDER_STAGE_FIELD: resolved,
        PARAM_LADDER_CAP_FIELD: cap,
    }
    if param_count is not None:
        payload["model_params"] = int(param_count)
        payload[PARAM_LADDER_PROVISIONAL_CROWN_ELIGIBLE_FIELD] = provisional_crown_eligible(
            stage=resolved,
            param_count=int(param_count),
            score_valid=score_valid,
        )
    else:
        payload[PARAM_LADDER_PROVISIONAL_CROWN_ELIGIBLE_FIELD] = provisional_crown_eligible(
            stage=resolved,
            score_valid=score_valid,
        )
    return payload


def dual_ladder_summary() -> dict[str, Any]:
    """Machine-readable dual-ladder constants for pin / docs / tests."""
    return {
        "stages": list(PARAM_LADDER_STAGES),
        "explore_max_parameters": EXPLORE_MAX_PARAMETERS,
        "promote_max_parameters": PROMOTE_MAX_PARAMETERS,
        "official_default_param_cap": OFFICIAL_DEFAULT_PARAM_CAP,
        "official_default_param_stage": OFFICIAL_DEFAULT_PARAM_STAGE,
        "legacy_single_param_cap_removed": LEGACY_SINGLE_PARAM_CAP,
        "provisional_crown_stage": STAGE_EXPLORE,
        "promote_path_stage": STAGE_PROMOTE,
    }
