"""AGATE measured LLM rules residual gate (package verify before TEE auth).

Product spine (mission AGATE / VAL-AGATE-004..007 + 012/013):

1. Production dual-flag path requires a **measured** agent-driven residual under
   ``.rules`` (review CVM path). Host static analyzer alone is **not** enough
   for eval prepare / TEE authorization.
2. Residual verdict + rules digests must be bound into review materials used for
   re-verify; residual fail/missing → no eval authorizable.
3. Agent model rule: **no closed model catalog**; **ban personal finetunes**
   only. Review judge pin (``REVIEW_MODEL``) stays separate.

Does not loosen ACLOCK env URL locks. Does not restore Base LLM gateway.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from agent_challenge.review.canonical import canonical_json_v1, canonical_sha256

# ---------------------------------------------------------------------------
# Stable refuse codes (wire)
# ---------------------------------------------------------------------------

REFUSE_RESIDUAL_MISSING = "package_residual_missing"
REFUSE_RESIDUAL_FAIL = "package_residual_reject"
REFUSE_RESIDUAL_UNBOUND = "package_residual_unbound"
REFUSE_HOST_ONLY = "host_analyzer_insufficient_for_tee"
REFUSE_RULES_DIGEST_MISSING = "rules_digests_missing"
REFUSE_PACKAGE_TREE_MISSING = "package_tree_sha_missing_for_residual"
REFUSE_FINETUNE = "agent_model_personal_finetune_forbidden"
REFUSE_RESIDUAL_KIND = "package_residual_kind_invalid"

# Residual kinds
MEASURED_RESIDUAL_KIND = "measured_review_cvm_llm_rules"
HOST_ANALYZER_KIND = "host_analyzer_static"
UNMEASURED_HOST_KIND = "unmeasured_host_python"

RESIDUAL_VERDICTS = frozenset({"allow", "reject", "fail"})
_ALLOW_VERDICT = "allow"

# Top-level envelope / outcome keys for bound residual materials.
ENVELOPE_RESIDUAL_KEY = "package_residual"
OUTCOME_RESIDUAL_KEY = "package_residual"

# Personal-finetune markers (agent / eval model path only — not review judge pin).
_FINETUNE_PREFIX_RE = re.compile(r"^ft:", re.IGNORECASE)
_FINETUNE_SEGMENT_RE = re.compile(
    r"(?:^|[:/_.-])(?:ft|finetune|fine-tune|fine_tune)(?:$|[:/_.-])",
    re.IGNORECASE,
)
_PERSONAL_PROVIDER_RE = re.compile(
    r"(?:^|/)(?:personal|private-ft|user-ft|my-finetune)(?:/|:)",
    re.IGNORECASE,
)
_OPENROUTER_CUSTOM_FT_RE = re.compile(
    r"(?:openrouter/)?(?:customer|custom|org)/.+:ft-",
    re.IGNORECASE,
)


class PackageResidualError(PermissionError):
    """Fail-closed package residual refuse with a stable reason code."""

    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = code
        super().__init__(message or code)


@dataclass(frozen=True)
class PackageResidualMaterials:
    """Bound residual materials carried next to review envelope/outcome."""

    residual_kind: str
    residual_verdict: str
    rules_bundle_sha256: str
    rules_version: str
    rules_file_digests: Mapping[str, str]
    package_tree_sha: str | None
    residual_digest: str
    rules_policy_text_sha256: str | None = None
    harness_kind: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "residual_kind": self.residual_kind,
            "residual_verdict": self.residual_verdict,
            "rules_bundle_sha256": self.rules_bundle_sha256,
            "rules_version": self.rules_version,
            "rules_file_digests": dict(self.rules_file_digests),
            "residual_digest": self.residual_digest,
        }
        if self.package_tree_sha is not None:
            payload["package_tree_sha"] = self.package_tree_sha
        if self.rules_policy_text_sha256 is not None:
            payload["rules_policy_text_sha256"] = self.rules_policy_text_sha256
        if self.harness_kind is not None:
            payload["harness_kind"] = self.harness_kind
        return payload


@dataclass(frozen=True)
class PackageResidualAdmission:
    """Decision for whether residual allows TEE-authable eval."""

    admitted: bool
    reason_code: str
    residual_verdict: str | None = None
    residual_kind: str | None = None
    residual_digest: str | None = None
    rules_bundle_sha256: str | None = None
    package_tree_sha: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "admitted": self.admitted,
            "reason_code": self.reason_code,
            "residual_verdict": self.residual_verdict,
            "residual_kind": self.residual_kind,
            "residual_digest": self.residual_digest,
            "rules_bundle_sha256": self.rules_bundle_sha256,
            "package_tree_sha": self.package_tree_sha,
        }


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise PackageResidualError(
            REFUSE_RULES_DIGEST_MISSING,
            f"{name} must be 64-char hex",
        )
    try:
        int(value, 16)
    except ValueError as exc:
        raise PackageResidualError(
            REFUSE_RULES_DIGEST_MISSING,
            f"{name} must be hex",
        ) from exc
    return value.lower()


def compute_residual_digest(payload: Mapping[str, Any]) -> str:
    """Canonical digest of residual bind payload (excludes residual_digest itself)."""

    closed = {
        "residual_kind": payload.get("residual_kind"),
        "residual_verdict": payload.get("residual_verdict"),
        "rules_bundle_sha256": payload.get("rules_bundle_sha256"),
        "rules_version": payload.get("rules_version"),
        "rules_file_digests": dict(payload.get("rules_file_digests") or {}),
        "package_tree_sha": payload.get("package_tree_sha"),
        "rules_policy_text_sha256": payload.get("rules_policy_text_sha256"),
        "harness_kind": payload.get("harness_kind"),
    }
    return canonical_sha256(closed)


def build_package_residual_materials(
    *,
    residual_verdict: str,
    rules_bundle_sha256: str,
    rules_version: str,
    rules_file_digests: Mapping[str, str],
    package_tree_sha: str | None = None,
    residual_kind: str = MEASURED_RESIDUAL_KIND,
    rules_policy_text_sha256: str | None = None,
    harness_kind: str | None = None,
) -> PackageResidualMaterials:
    """Build bound residual materials for review envelope / outcome."""

    verdict = str(residual_verdict or "").strip().lower()
    if verdict not in RESIDUAL_VERDICTS:
        raise PackageResidualError(
            REFUSE_RESIDUAL_KIND,
            f"invalid residual_verdict {residual_verdict!r}",
        )
    kind = str(residual_kind or "").strip()
    if not kind:
        raise PackageResidualError(REFUSE_RESIDUAL_KIND, "residual_kind required")

    bundle = _require_sha256(rules_bundle_sha256, "rules_bundle_sha256")
    version = str(rules_version or "").strip()
    if not version:
        raise PackageResidualError(REFUSE_RULES_DIGEST_MISSING, "rules_version required")

    digests: dict[str, str] = {}
    if not isinstance(rules_file_digests, Mapping) or not rules_file_digests:
        raise PackageResidualError(
            REFUSE_RULES_DIGEST_MISSING,
            "rules_file_digests required",
        )
    for path, digest in sorted(rules_file_digests.items(), key=lambda item: str(item[0])):
        rel = str(path).strip()
        if not rel:
            raise PackageResidualError(REFUSE_RULES_DIGEST_MISSING, "empty rules path")
        digests[rel] = _require_sha256(digest, f"rules_file_digests[{rel}]")

    tree: str | None = None
    if package_tree_sha is not None and str(package_tree_sha).strip():
        tree = _require_sha256(str(package_tree_sha).strip(), "package_tree_sha")

    policy_sha: str | None = None
    if rules_policy_text_sha256 is not None and str(rules_policy_text_sha256).strip():
        policy_sha = _require_sha256(
            str(rules_policy_text_sha256).strip(),
            "rules_policy_text_sha256",
        )

    raw = {
        "residual_kind": kind,
        "residual_verdict": verdict,
        "rules_bundle_sha256": bundle,
        "rules_version": version,
        "rules_file_digests": digests,
        "package_tree_sha": tree,
        "rules_policy_text_sha256": policy_sha,
        "harness_kind": harness_kind,
    }
    digest = compute_residual_digest(raw)
    return PackageResidualMaterials(
        residual_kind=kind,
        residual_verdict=verdict,
        rules_bundle_sha256=bundle,
        rules_version=version,
        rules_file_digests=digests,
        package_tree_sha=tree,
        residual_digest=digest,
        rules_policy_text_sha256=policy_sha,
        harness_kind=harness_kind,
    )


def materials_from_mapping(raw: Mapping[str, Any] | None) -> PackageResidualMaterials | None:
    """Parse residual materials from envelope/outcome; None if absent."""

    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise PackageResidualError(REFUSE_RESIDUAL_UNBOUND, "package_residual must be object")
    if not raw:
        return None
    try:
        materials = build_package_residual_materials(
            residual_verdict=str(raw.get("residual_verdict") or ""),
            rules_bundle_sha256=str(raw.get("rules_bundle_sha256") or ""),
            rules_version=str(raw.get("rules_version") or ""),
            rules_file_digests=dict(raw.get("rules_file_digests") or {}),
            package_tree_sha=(
                str(raw["package_tree_sha"]) if raw.get("package_tree_sha") is not None else None
            ),
            residual_kind=str(raw.get("residual_kind") or ""),
            rules_policy_text_sha256=(
                str(raw["rules_policy_text_sha256"])
                if raw.get("rules_policy_text_sha256") is not None
                else None
            ),
            harness_kind=(
                str(raw["harness_kind"]) if raw.get("harness_kind") is not None else None
            ),
        )
    except PackageResidualError:
        raise
    except (TypeError, ValueError) as exc:
        raise PackageResidualError(REFUSE_RESIDUAL_UNBOUND, str(exc)) from exc

    claimed = raw.get("residual_digest")
    if claimed is not None and str(claimed).lower() != materials.residual_digest:
        raise PackageResidualError(
            REFUSE_RESIDUAL_UNBOUND,
            "residual_digest does not match recomputed materials",
        )
    return materials


def bind_package_residual_into_review_materials(
    *,
    envelope: Mapping[str, Any] | None = None,
    outcome: Mapping[str, Any] | None = None,
    materials: PackageResidualMaterials,
) -> dict[str, Any]:
    """Bind residual materials into review envelope and/or outcome bags.

    Returns a dict with optionally updated ``envelope`` and ``outcome`` copies.
    """

    result: dict[str, Any] = {}
    bag = materials.as_dict()
    if envelope is not None:
        env = dict(envelope)
        env[ENVELOPE_RESIDUAL_KEY] = bag
        result["envelope"] = env
    if outcome is not None:
        out = dict(outcome)
        out[OUTCOME_RESIDUAL_KEY] = bag
        result["outcome"] = out
    if not result:
        result["package_residual"] = bag
    return result


def extract_package_residual(
    *,
    envelope: Mapping[str, Any] | str | bytes | None = None,
    outcome: Mapping[str, Any] | None = None,
    residual: Mapping[str, Any] | None = None,
    review_core: Mapping[str, Any] | None = None,
) -> PackageResidualMaterials | None:
    """Locate residual materials from envelope, outcome, explicit bag, or core side.

    Prefer explicit residual / envelope top-level; fall back to outcome; then a
    non-schema ``package_residual`` nested under review_core (when callers stashed
    materials without mutating closed report keys).
    """

    if residual is not None:
        return materials_from_mapping(residual)

    env: Mapping[str, Any] | None = None
    if isinstance(envelope, Mapping):
        env = envelope
    elif isinstance(envelope, (str, bytes)):
        import json

        try:
            raw = envelope if isinstance(envelope, str) else envelope.decode("utf-8")
            parsed = json.loads(raw)
        except (TypeError, ValueError, UnicodeDecodeError):
            parsed = None
        if isinstance(parsed, Mapping):
            env = parsed

    if env is not None:
        if ENVELOPE_RESIDUAL_KEY in env:
            return materials_from_mapping(env.get(ENVELOPE_RESIDUAL_KEY))  # type: ignore[arg-type]
        nested = env.get("envelope")
        if isinstance(nested, Mapping) and ENVELOPE_RESIDUAL_KEY in nested:
            return materials_from_mapping(nested.get(ENVELOPE_RESIDUAL_KEY))  # type: ignore[arg-type]

    if isinstance(outcome, Mapping) and OUTCOME_RESIDUAL_KEY in outcome:
        return materials_from_mapping(outcome.get(OUTCOME_RESIDUAL_KEY))  # type: ignore[arg-type]

    if isinstance(review_core, Mapping):
        side = review_core.get(ENVELOPE_RESIDUAL_KEY)
        if isinstance(side, Mapping):
            return materials_from_mapping(side)
        # Also accept rules digests already observed in harness identity fields
        # when residual_kind/verdict are present as siblings (session bind path).
        if review_core.get("residual_verdict") is not None:
            return materials_from_mapping(review_core)

    return None


def is_host_only_residual(kind: str | None) -> bool:
    """True when residual claims host static analyzer / unmeasured host path."""

    k = (kind or "").strip()
    return k in {HOST_ANALYZER_KIND, UNMEASURED_HOST_KIND, "host_analyzer", "offline_ast"}


def is_measured_residual_kind(kind: str | None) -> bool:
    k = (kind or "").strip()
    return k == MEASURED_RESIDUAL_KIND or k in {
        "measured_review_cvm",
        "measured_review_cvm_script_zip",
        "selfdeploy_review",
    }


def admit_package_residual_for_eval(
    *,
    envelope: Mapping[str, Any] | str | bytes | None = None,
    outcome: Mapping[str, Any] | None = None,
    residual: Mapping[str, Any] | None = None,
    review_core: Mapping[str, Any] | None = None,
    dual_flags_on: bool = True,
    require_package_tree_sha: bool = True,
    expected_package_tree_sha: str | None = None,
    host_analyzer_allow: bool | None = None,
) -> PackageResidualAdmission:
    """Admit eval/TEE only when measured residual allow is bound.

    Production dual-flag path:
    - Missing residual → refuse
    - Host-only residual / host analyzer allow alone → refuse
    - residual_verdict != allow → refuse
    - Missing rules digests → refuse (build path)
    - package_tree_sha required/match when configured
    """

    if not dual_flags_on:
        # Residual gate is a production dual-flag requirement; flag-off is not
        # TEE-authable production emit (other gates already refuse). Treat as
        # residual missing for prepare-style callers that still invoke this.
        return PackageResidualAdmission(
            admitted=False,
            reason_code=REFUSE_RESIDUAL_MISSING,
        )

    try:
        materials = extract_package_residual(
            envelope=envelope,
            outcome=outcome,
            residual=residual,
            review_core=review_core,
        )
    except PackageResidualError as exc:
        return PackageResidualAdmission(admitted=False, reason_code=exc.code)

    if materials is None:
        # Host analyzer allow alone never unlocks TEE auth.
        if host_analyzer_allow is True:
            return PackageResidualAdmission(
                admitted=False,
                reason_code=REFUSE_HOST_ONLY,
            )
        return PackageResidualAdmission(
            admitted=False,
            reason_code=REFUSE_RESIDUAL_MISSING,
        )

    if is_host_only_residual(materials.residual_kind):
        return PackageResidualAdmission(
            admitted=False,
            reason_code=REFUSE_HOST_ONLY,
            residual_verdict=materials.residual_verdict,
            residual_kind=materials.residual_kind,
            residual_digest=materials.residual_digest,
            rules_bundle_sha256=materials.rules_bundle_sha256,
            package_tree_sha=materials.package_tree_sha,
        )

    if not is_measured_residual_kind(materials.residual_kind):
        return PackageResidualAdmission(
            admitted=False,
            reason_code=REFUSE_RESIDUAL_KIND,
            residual_verdict=materials.residual_verdict,
            residual_kind=materials.residual_kind,
            residual_digest=materials.residual_digest,
            rules_bundle_sha256=materials.rules_bundle_sha256,
            package_tree_sha=materials.package_tree_sha,
        )

    if materials.residual_verdict != _ALLOW_VERDICT:
        return PackageResidualAdmission(
            admitted=False,
            reason_code=REFUSE_RESIDUAL_FAIL,
            residual_verdict=materials.residual_verdict,
            residual_kind=materials.residual_kind,
            residual_digest=materials.residual_digest,
            rules_bundle_sha256=materials.rules_bundle_sha256,
            package_tree_sha=materials.package_tree_sha,
        )

    if require_package_tree_sha:
        tree = materials.package_tree_sha
        if not tree:
            return PackageResidualAdmission(
                admitted=False,
                reason_code=REFUSE_PACKAGE_TREE_MISSING,
                residual_verdict=materials.residual_verdict,
                residual_kind=materials.residual_kind,
                residual_digest=materials.residual_digest,
                rules_bundle_sha256=materials.rules_bundle_sha256,
            )
        if expected_package_tree_sha is not None:
            expected = str(expected_package_tree_sha).strip().lower()
            if expected and expected != tree:
                return PackageResidualAdmission(
                    admitted=False,
                    reason_code=REFUSE_RESIDUAL_UNBOUND,
                    residual_verdict=materials.residual_verdict,
                    residual_kind=materials.residual_kind,
                    residual_digest=materials.residual_digest,
                    rules_bundle_sha256=materials.rules_bundle_sha256,
                    package_tree_sha=tree,
                )

    return PackageResidualAdmission(
        admitted=True,
        reason_code="package_residual_allow",
        residual_verdict=materials.residual_verdict,
        residual_kind=materials.residual_kind,
        residual_digest=materials.residual_digest,
        rules_bundle_sha256=materials.rules_bundle_sha256,
        package_tree_sha=materials.package_tree_sha,
    )


def require_package_residual_for_eval(**kwargs: Any) -> PackageResidualAdmission:
    """Fail-closed wrapper for prepare / TEE-auth call sites."""

    decision = admit_package_residual_for_eval(**kwargs)
    if not decision.admitted:
        raise PackageResidualError(decision.reason_code)
    return decision


def host_analyzer_alone_insufficient(
    *,
    host_analyzer_allow: bool,
    residual: Mapping[str, Any] | PackageResidualMaterials | None = None,
    dual_flags_on: bool = True,
) -> bool:
    """VAL-AGATE-007: host allow without measured residual is insufficient."""

    if not dual_flags_on:
        return True
    if not host_analyzer_allow:
        return True
    bag: Mapping[str, Any] | None
    if isinstance(residual, PackageResidualMaterials):
        bag = residual.as_dict()
    else:
        bag = residual
    decision = admit_package_residual_for_eval(
        residual=bag,
        dual_flags_on=True,
        host_analyzer_allow=host_analyzer_allow,
        require_package_tree_sha=False,
    )
    return not decision.admitted


# ---------------------------------------------------------------------------
# Agent model rule: no closed catalog; ban personal finetunes only
# ---------------------------------------------------------------------------


def is_personal_finetune_model(model_id: str | None) -> bool:
    """Detect personally fine-tuned / custom finetune model identifiers."""

    if model_id is None:
        return False
    text = str(model_id).strip()
    if not text:
        return False
    if _FINETUNE_PREFIX_RE.search(text):
        return True
    if _OPENROUTER_CUSTOM_FT_RE.search(text):
        return True
    if _PERSONAL_PROVIDER_RE.search(text):
        return True
    # Segment form: openai/gpt-4o:ft-org-name-... or .../fine_tune/...
    if _FINETUNE_SEGMENT_RE.search(text):
        return True
    lowered = text.lower()
    # Explicit personal markers without catalog.
    if "personal-finetune" in lowered or "private-finetune" in lowered:
        return True
    return False


def refuse_personal_finetune_model(model_id: str | None) -> None:
    """Raise when model_id is a personal finetune (stable reason code)."""

    if is_personal_finetune_model(model_id):
        raise PackageResidualError(
            REFUSE_FINETUNE,
            f"personal finetune model refused: {model_id!r}",
        )


def agent_model_requires_closed_catalog() -> bool:
    """VAL-AGATE-012: agent/eval path does not require a fixed model catalog."""

    return False


def assert_no_closed_agent_model_catalog(
    catalog: Sequence[str] | Mapping[str, Any] | None,
) -> None:
    """Product assert: empty/None catalog is OK; non-empty closed catalog is not required.

    Callers must not treat a fixed short allowlist as a production gate for miners.
    A present catalog for documentation is ignored; enforcement must only use the
    finetune ban. This helper fails if code attempts to *require* membership.
    """

    # No exception — closed catalogs are simply not required. Keep for tests/rg.
    _ = catalog
    if agent_model_requires_closed_catalog():  # pragma: no cover - constant False
        raise PackageResidualError("agent_model_closed_catalog_required")


def filter_agent_model_or_refuse(model_id: str | None) -> str | None:
    """Pass-through model id after finetune refuse; no catalog membership check."""

    if model_id is None or str(model_id).strip() == "":
        return None
    refuse_personal_finetune_model(model_id)
    return str(model_id).strip()


def residual_materials_from_rules_pack(
    *,
    rules_version: str,
    rules_bundle_sha256: str,
    rules_file_digests: Mapping[str, str],
    residual_verdict: str = "allow",
    package_tree_sha: str | None = None,
    rules_policy_text_sha256: str | None = None,
    harness_kind: str | None = None,
    residual_kind: str = MEASURED_RESIDUAL_KIND,
) -> PackageResidualMaterials:
    """Convenience builder from harness rules pack digests."""

    return build_package_residual_materials(
        residual_verdict=residual_verdict,
        rules_bundle_sha256=rules_bundle_sha256,
        rules_version=rules_version,
        rules_file_digests=rules_file_digests,
        package_tree_sha=package_tree_sha,
        residual_kind=residual_kind,
        rules_policy_text_sha256=rules_policy_text_sha256,
        harness_kind=harness_kind,
    )


def sha256_hex(data: bytes) -> str:
    return sha256(data).hexdigest()


def inventory_residual_gate() -> dict[str, Any]:
    """Static inventory for black-box / docs checks."""

    return {
        "measured_residual_kind": MEASURED_RESIDUAL_KIND,
        "host_insufficient_kinds": sorted({HOST_ANALYZER_KIND, UNMEASURED_HOST_KIND}),
        "envelope_key": ENVELOPE_RESIDUAL_KEY,
        "require_before_eval": True,
        "host_analyzer_alone_sufficient": False,
        "agent_model_closed_catalog": False,
        "personal_finetune_refuse_code": REFUSE_FINETUNE,
        "residual_fail_code": REFUSE_RESIDUAL_FAIL,
        "residual_missing_code": REFUSE_RESIDUAL_MISSING,
        "binds": [
            "residual_verdict",
            "rules_bundle_sha256",
            "rules_version",
            "rules_file_digests",
            "package_tree_sha",
            "residual_digest",
        ],
        "canonical_bind": True,
        "canonical_json_v1_digest": True,
        "digest_helper": canonical_json_v1.__name__,
    }


__all__ = [
    "ENVELOPE_RESIDUAL_KEY",
    "HOST_ANALYZER_KIND",
    "MEASURED_RESIDUAL_KIND",
    "OUTCOME_RESIDUAL_KEY",
    "PackageResidualAdmission",
    "PackageResidualError",
    "PackageResidualMaterials",
    "REFUSE_FINETUNE",
    "REFUSE_HOST_ONLY",
    "REFUSE_PACKAGE_TREE_MISSING",
    "REFUSE_RESIDUAL_FAIL",
    "REFUSE_RESIDUAL_KIND",
    "REFUSE_RESIDUAL_MISSING",
    "REFUSE_RESIDUAL_UNBOUND",
    "REFUSE_RULES_DIGEST_MISSING",
    "RESIDUAL_VERDICTS",
    "UNMEASURED_HOST_KIND",
    "admit_package_residual_for_eval",
    "agent_model_requires_closed_catalog",
    "assert_no_closed_agent_model_catalog",
    "bind_package_residual_into_review_materials",
    "build_package_residual_materials",
    "compute_residual_digest",
    "extract_package_residual",
    "filter_agent_model_or_refuse",
    "host_analyzer_alone_insufficient",
    "inventory_residual_gate",
    "is_host_only_residual",
    "is_measured_residual_kind",
    "is_personal_finetune_model",
    "materials_from_mapping",
    "refuse_personal_finetune_model",
    "require_package_residual_for_eval",
    "residual_materials_from_rules_pack",
    "sha256_hex",
]
