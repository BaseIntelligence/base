"""Product miner harness entry: ZIP + shipping script + .rules before OpenRouter.

Sole production admission path into the measured **review** harness is:

1. Non-empty agent ZIP bytes (``zip_sha256``)
2. Shipping entry-script identity (selfdeploy / review runtime)
3. Authoritative ``.rules`` pack digests (``rules_version`` + bundle)

``tools/agent_parity_harness.py`` is **never** accepted as product review.
OpenRouter judgment is allowed only after rules digests are bound.

Refuse codes are stable wire strings (library/ac-attestation.md Mode A freeze).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any

from .canonical import canonical_json_v1, canonical_sha256

# ---------------------------------------------------------------------------
# Stable refuse / entry codes (wire) — mirror library/ac-attestation.md
# ---------------------------------------------------------------------------

REFUSE_MISSING_ZIP = "product_review_missing_zip"
REFUSE_EMPTY_ZIP = "product_review_empty_zip"
REFUSE_MISSING_ENTRY_SCRIPT = "product_review_missing_entry_script"
REFUSE_MISSING_RULES = "product_review_missing_rules"
REFUSE_EMPTY_RULES = "product_review_empty_rules"
REFUSE_PARITY_HARNESS = "product_review_parity_harness_forbidden"
REFUSE_UNMEASURED_HOST = "product_review_unmeasured_host_forbidden"
REFUSE_OR_BEFORE_RULES = "product_review_openrouter_before_rules_forbidden"

PRODUCT_HARNESS_KIND = "measured_review_cvm_script_zip"
PARITY_HARNESS_KIND = "agent_parity_harness"
SESSION_IDENTITY_SCHEMA_V1 = 1

# Known non-product tools (match inventory / research locks).
_PARITY_HARNESS_MARKERS = frozenset(
    {
        "agent_parity_harness",
        "agent_parity_harness.py",
        "tools/agent_parity_harness.py",
        "tools.agent_parity_harness",
    }
)

# Shipping entry identities that *are* product (self-deploy / review CLI).
PRODUCT_ENTRY_SCRIPT_MARKERS = frozenset(
    {
        "agent_challenge.selfdeploy",
        "python -m agent_challenge.selfdeploy",
        "agent-challenge-selfdeploy",
        "selfdeploy/cli.py",
        "src/agent_challenge/selfdeploy/cli.py",
        "selfdeploy/review.py",
        "src/agent_challenge/selfdeploy/review.py",
        "docker/review/review_runtime.py",
    }
)

# Acceptable product harness_kind aliases for measured review.
_PRODUCT_HARNESS_KINDS = frozenset(
    {
        PRODUCT_HARNESS_KIND,
        "measured_review_cvm",
        "selfdeploy_review",
    }
)

_UNMEASURED_KINDS = frozenset(
    {
        "host_python",
        "offline_ast",
        "base_master_llm",
        "unmeasured",
        PARITY_HARNESS_KIND,
    }
)


def sha256_hex(data: bytes) -> str:
    return sha256(data).hexdigest()


def _normalize_entry_marker(entry_script: str | Path | None) -> str:
    if entry_script is None:
        return ""
    text = str(entry_script).strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def is_parity_harness_entry(entry_script: str | Path | None) -> bool:
    """True when the proposed entry is the internal parity tool (not product)."""

    marker = _normalize_entry_marker(entry_script)
    if not marker:
        return False
    lower = marker.lower()
    base = PurePosixPath(lower).name
    if base in _PARITY_HARNESS_MARKERS or lower in _PARITY_HARNESS_MARKERS:
        return True
    if "agent_parity_harness" in lower:
        return True
    return False


def is_product_entry_script(entry_script: str | Path | None) -> bool:
    """Heuristic inventory check: known shipping self-deploy / review entry."""

    marker = _normalize_entry_marker(entry_script)
    if not marker or is_parity_harness_entry(marker):
        return False
    lower = marker.lower()
    if lower in {m.lower() for m in PRODUCT_ENTRY_SCRIPT_MARKERS}:
        return True
    for product in PRODUCT_ENTRY_SCRIPT_MARKERS:
        p = product.lower()
        if lower.endswith("/" + p) or lower.endswith(p):
            return True
    return False


def digest_entry_script(
    *,
    entry_script_path: str | Path | None = None,
    entry_script_bytes: bytes | None = None,
    entry_script_identity: str | None = None,
) -> dict[str, str]:
    """Build entry-script digests for session identity."""

    identity = _normalize_entry_marker(
        entry_script_identity or (str(entry_script_path) if entry_script_path is not None else "")
    )
    if not identity and entry_script_bytes is None:
        raise ValueError(REFUSE_MISSING_ENTRY_SCRIPT)

    path_digest = sha256_hex(identity.encode("utf-8")) if identity else ""
    if entry_script_bytes is not None:
        content_digest = sha256_hex(entry_script_bytes)
    elif entry_script_path is not None:
        path = Path(entry_script_path)
        if path.is_file():
            content_digest = sha256_hex(path.read_bytes())
        else:
            content_digest = path_digest
    else:
        content_digest = path_digest

    return {
        "entry_script_identity": identity or "bytes-only",
        "entry_script_path_sha256": path_digest or content_digest,
        "entry_script_content_sha256": content_digest,
    }


def digest_agent_zip(zip_bytes: bytes | None) -> str:
    """SHA-256 of exact agent ZIP bytes (production artifact identity)."""

    if zip_bytes is None:
        raise ValueError(REFUSE_MISSING_ZIP)
    if not isinstance(zip_bytes, (bytes, bytearray)):
        raise TypeError("zip_bytes must be bytes")
    if len(zip_bytes) == 0:
        raise ValueError(REFUSE_EMPTY_ZIP)
    return sha256_hex(bytes(zip_bytes))


@dataclass(frozen=True)
class RulesPackDigest:
    """Authoritative .rules pack digests for pre-OpenRouter binding."""

    rules_version: str
    bundle_sha256: str
    files: tuple[str, ...]
    file_digests: Mapping[str, str]
    policy_text_sha256: str

    def as_session_fields(self) -> dict[str, Any]:
        return {
            "rules_version": self.rules_version,
            "rules_bundle_sha256": self.bundle_sha256,
            "rules_files": list(self.files),
            "rules_file_digests": dict(self.file_digests),
            "rules_policy_text_sha256": self.policy_text_sha256,
        }


def load_rules_pack_digests(
    rules_dir: Path | str | None = None,
    *,
    files: Mapping[str, bytes] | None = None,
) -> RulesPackDigest:
    """Load .rules Markdown pack and compute rules_version + bundle digests.

    Encoding matches :func:`agent_challenge.rules.load_rules`: sorted relative
    paths, each ``path || NUL || content || NUL`` into the rules_version stream;
    bundle is the canonical JSON of path→sha256 map.
    """

    if files is None:
        if rules_dir is None:
            raise ValueError(REFUSE_MISSING_RULES)
        root = Path(rules_dir)
        if not root.is_dir():
            raise ValueError(REFUSE_MISSING_RULES)
        md_paths = sorted(p for p in root.glob("*.md") if p.is_file())
        if not md_paths:
            raise ValueError(REFUSE_EMPTY_RULES)
        loaded: dict[str, bytes] = {}
        for path in md_paths:
            try:
                rel = path.relative_to(root).as_posix()
            except ValueError:
                rel = path.name
            if root.name == ".rules":
                rel = f".rules/{path.name}"
            loaded[rel] = path.read_bytes()
        files = loaded
    else:
        if not files:
            raise ValueError(REFUSE_EMPTY_RULES)

    ordered = sorted(files.items(), key=lambda item: item[0])
    version_stream = sha256()
    file_digests: dict[str, str] = {}
    policy_sections: list[str] = []
    for relative, content in ordered:
        if not isinstance(content, (bytes, bytearray)):
            raise TypeError(f"rules content for {relative!r} must be bytes")
        version_stream.update(relative.encode("utf-8"))
        version_stream.update(b"\0")
        version_stream.update(content)
        version_stream.update(b"\0")
        file_digests[relative] = sha256_hex(bytes(content))
        policy_sections.append(bytes(content).decode("utf-8"))

    rules_version = version_stream.hexdigest()
    bundle_sha256 = sha256_hex(canonical_json_v1(file_digests))
    policy_text = "\n\n".join(section.rstrip() for section in policy_sections) + "\n"
    policy_text_sha256 = sha256_hex(policy_text.encode("utf-8"))
    return RulesPackDigest(
        rules_version=rules_version,
        bundle_sha256=bundle_sha256,
        files=tuple(k for k, _ in ordered),
        file_digests=file_digests,
        policy_text_sha256=policy_text_sha256,
    )


@dataclass(frozen=True)
class ReviewSessionIdentity:
    """Review session identity materials bound before measured OpenRouter judgment."""

    schema_version: int
    harness_kind: str
    zip_sha256: str
    entry_script_identity: str
    entry_script_path_sha256: str
    entry_script_content_sha256: str
    rules_version: str
    rules_bundle_sha256: str
    rules_files: tuple[str, ...]
    rules_file_digests: Mapping[str, str]
    rules_policy_text_sha256: str
    openrouter_allowed: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "harness_kind": self.harness_kind,
            "zip_sha256": self.zip_sha256,
            "entry_script_identity": self.entry_script_identity,
            "entry_script_path_sha256": self.entry_script_path_sha256,
            "entry_script_content_sha256": self.entry_script_content_sha256,
            "rules_version": self.rules_version,
            "rules_bundle_sha256": self.rules_bundle_sha256,
            "rules_files": list(self.rules_files),
            "rules_file_digests": dict(self.rules_file_digests),
            "rules_policy_text_sha256": self.rules_policy_text_sha256,
            "openrouter_allowed": self.openrouter_allowed,
        }

    def session_identity_sha256(self) -> str:
        return canonical_sha256(self.as_dict())


class ProductHarnessAdmissionError(ValueError):
    """Production review path refused at harness entry."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(code if not detail else f"{code}: {detail}")


def admit_product_review_entry(
    *,
    agent_zip_bytes: bytes | None,
    entry_script: str | Path | None = None,
    entry_script_bytes: bytes | None = None,
    entry_script_identity: str | None = None,
    rules_dir: Path | str | None = None,
    rules_files: Mapping[str, bytes] | None = None,
    harness_kind: str | None = None,
    openrouter_call_attempted: bool = False,
) -> ReviewSessionIdentity:
    """Sole product gate into measured review harness.

    Steps (fail closed, in order):
    1. Reject parity harness / unmeasured host markers.
    2. Require non-empty agent ZIP bytes → ``zip_sha256``.
    3. Require entry script identity (path and/or content) → script digests.
    4. Load rules pack digests; refuse missing/empty rules.
    5. Mark OpenRouter allowed only AFTER rules digests are bound.
    6. If OpenRouter was attempted before rules (caller flag), refuse.
    """

    kind = (harness_kind or PRODUCT_HARNESS_KIND).strip()
    identity_hint = entry_script_identity or (
        str(entry_script) if entry_script is not None else None
    )

    if (
        kind == PARITY_HARNESS_KIND
        or is_parity_harness_entry(identity_hint)
        or is_parity_harness_entry(entry_script)
    ):
        raise ProductHarnessAdmissionError(
            REFUSE_PARITY_HARNESS,
            "agent_parity_harness is internal evaluation parity, not product review",
        )
    if kind not in _PRODUCT_HARNESS_KINDS:
        if kind in _UNMEASURED_KINDS:
            raise ProductHarnessAdmissionError(
                REFUSE_UNMEASURED_HOST,
                f"harness_kind={kind!r} is not the measured review product path",
            )
        raise ProductHarnessAdmissionError(
            REFUSE_UNMEASURED_HOST,
            f"harness_kind={kind!r} is not the measured review product path",
        )

    if openrouter_call_attempted and rules_files is None and rules_dir is None:
        raise ProductHarnessAdmissionError(REFUSE_OR_BEFORE_RULES)

    try:
        zip_sha = digest_agent_zip(agent_zip_bytes)
    except ValueError as exc:
        raise ProductHarnessAdmissionError(str(exc)) from exc

    try:
        script = digest_entry_script(
            entry_script_path=entry_script,
            entry_script_bytes=entry_script_bytes,
            entry_script_identity=entry_script_identity or identity_hint,
        )
    except ValueError as exc:
        raise ProductHarnessAdmissionError(str(exc) or REFUSE_MISSING_ENTRY_SCRIPT) from exc

    if not script["entry_script_identity"] or script["entry_script_identity"] == "bytes-only":
        if entry_script_bytes is None and not identity_hint:
            raise ProductHarnessAdmissionError(REFUSE_MISSING_ENTRY_SCRIPT)

    try:
        rules = load_rules_pack_digests(rules_dir, files=rules_files)
    except ValueError as exc:
        code = str(exc)
        if code not in {REFUSE_MISSING_RULES, REFUSE_EMPTY_RULES}:
            code = REFUSE_MISSING_RULES
        raise ProductHarnessAdmissionError(code) from exc

    if openrouter_call_attempted:
        raise ProductHarnessAdmissionError(REFUSE_OR_BEFORE_RULES)

    return ReviewSessionIdentity(
        schema_version=SESSION_IDENTITY_SCHEMA_V1,
        harness_kind=PRODUCT_HARNESS_KIND,
        zip_sha256=zip_sha,
        entry_script_identity=script["entry_script_identity"],
        entry_script_path_sha256=script["entry_script_path_sha256"],
        entry_script_content_sha256=script["entry_script_content_sha256"],
        rules_version=rules.rules_version,
        rules_bundle_sha256=rules.bundle_sha256,
        rules_files=rules.files,
        rules_file_digests=rules.file_digests,
        rules_policy_text_sha256=rules.policy_text_sha256,
        openrouter_allowed=True,
    )


def require_rules_before_openrouter(identity: ReviewSessionIdentity | None) -> str | None:
    """Return refuse code if OpenRouter must not run yet; None if OK."""

    if identity is None:
        return REFUSE_OR_BEFORE_RULES
    if not identity.rules_version or not identity.rules_bundle_sha256:
        return REFUSE_MISSING_RULES
    if not identity.openrouter_allowed:
        return REFUSE_OR_BEFORE_RULES
    return None


def inventory_product_vs_parity() -> dict[str, Any]:
    """Static inventory statement for VAL-ACAT-001 black-box checks."""

    return {
        "product_harness": {
            "kind": PRODUCT_HARNESS_KIND,
            "entry_examples": sorted(PRODUCT_ENTRY_SCRIPT_MARKERS),
            "requires": ["agent_zip_bytes", "entry_script_identity", "rules_pack"],
            "binds_into_session": [
                "zip_sha256",
                "entry_script_path_sha256",
                "entry_script_content_sha256",
                "rules_version",
                "rules_bundle_sha256",
            ],
        },
        "not_product": {
            "agent_parity_harness": {
                "path": "tools/agent_parity_harness.py",
                "role": "internal harbor vs own_runner parity (Task 23)",
                "refuse_code": REFUSE_PARITY_HARNESS,
            },
            "offline_ast": {
                "role": "static green without attestation",
                "refuse_code": REFUSE_UNMEASURED_HOST,
            },
            "base_master_llm": {
                "role": "Base LLM gateway / master-side review",
                "refuse_code": REFUSE_UNMEASURED_HOST,
            },
        },
        "rules_before_openrouter": True,
        "session_identity_schema_version": SESSION_IDENTITY_SCHEMA_V1,
    }


def refuse_parity_harness_as_review(entry: str | Path | None = None) -> None:
    """Explicit product ban: parity harness is never production review."""

    if entry is None or is_parity_harness_entry(entry):
        raise ProductHarnessAdmissionError(
            REFUSE_PARITY_HARNESS,
            "agent_parity_harness is internal evaluation parity, not product review",
        )


# ---------------------------------------------------------------------------
# AGATE residual producer (measured LLM rules residual → durable materials)
# ---------------------------------------------------------------------------


def map_decision_verdict_to_residual_verdict(decision_verdict: str | None) -> str:
    """Map review policy/decision verdict onto residual_verdict {allow,reject,fail}.

    - allow → allow (prep may authorize when residual+tree bound)
    - reject → reject (prepare stays refuse)
    - escalate / other / missing → fail (prepare stays refuse)
    """

    text = str(decision_verdict or "").strip().lower()
    if text in {"allow", "allowed", "accept", "accepted"}:
        return "allow"
    if text in {"reject", "rejected", "deny", "denied"}:
        return "reject"
    return "fail"


def produce_package_residual_from_identity(
    identity: ReviewSessionIdentity | Mapping[str, Any],
    *,
    residual_verdict: str,
    package_tree_sha: str | None = None,
    residual_kind: str | None = None,
) -> Any:
    """Build measured package residual materials from harness session identity.

    Live harness_entry producer path: after rules residual (measured review
    decision) completes, bind verdict + rules digests + package_tree_sha into
    materials used by eval prepare / TEE auth.
    """

    from agent_challenge.evaluation.llm_rules_residual import (
        MEASURED_RESIDUAL_KIND,
        residual_materials_from_rules_pack,
    )

    if isinstance(identity, ReviewSessionIdentity):
        rules_version = identity.rules_version
        rules_bundle = identity.rules_bundle_sha256
        file_digests = dict(identity.rules_file_digests)
        policy_sha = identity.rules_policy_text_sha256
        harness_kind = identity.harness_kind or PRODUCT_HARNESS_KIND
    elif isinstance(identity, Mapping):
        rules_version = str(identity.get("rules_version") or "").strip()
        rules_bundle = str(identity.get("rules_bundle_sha256") or "").strip()
        raw_digests = identity.get("rules_file_digests") or {}
        if not isinstance(raw_digests, Mapping) or not raw_digests:
            raise ProductHarnessAdmissionError(
                REFUSE_MISSING_RULES,
                "rules_file_digests required for residual producer",
            )
        file_digests = {str(k): str(v) for k, v in raw_digests.items()}
        policy_raw = identity.get("rules_policy_text_sha256")
        policy_sha = str(policy_raw).strip() if policy_raw is not None else None
        harness_kind = str(identity.get("harness_kind") or PRODUCT_HARNESS_KIND)
    else:
        raise TypeError("identity must be ReviewSessionIdentity or mapping")

    if not rules_version or not rules_bundle or not file_digests:
        raise ProductHarnessAdmissionError(
            REFUSE_MISSING_RULES,
            "rules digests required for residual producer",
        )

    kind = residual_kind or MEASURED_RESIDUAL_KIND
    return residual_materials_from_rules_pack(
        rules_version=rules_version,
        rules_bundle_sha256=rules_bundle,
        rules_file_digests=file_digests,
        residual_verdict=residual_verdict,
        package_tree_sha=package_tree_sha,
        rules_policy_text_sha256=policy_sha,
        harness_kind=harness_kind,
        residual_kind=kind,
    )


def bind_measured_residual_into_review_materials(
    *,
    identity: ReviewSessionIdentity | Mapping[str, Any],
    residual_verdict: str,
    package_tree_sha: str | None = None,
    envelope: Mapping[str, Any] | None = None,
    outcome: Mapping[str, Any] | None = None,
    residual_kind: str | None = None,
) -> dict[str, Any]:
    """Produce residual from harness identity and bind into envelope and/or outcome.

    Preference for durable host storage: bind onto **outcome** (verification
    outcome JSON is not exact-key locked). Envelope may also carry residual when
    callers want a full bound bag for unit/fixture paths; closed guest envelope
    schema still validates exact keys without residual (host merge prefers
    outcome).
    """

    from agent_challenge.evaluation.llm_rules_residual import (
        bind_package_residual_into_review_materials,
    )

    materials = produce_package_residual_from_identity(
        identity,
        residual_verdict=residual_verdict,
        package_tree_sha=package_tree_sha,
        residual_kind=residual_kind,
    )
    return bind_package_residual_into_review_materials(
        envelope=envelope,
        outcome=outcome,
        materials=materials,
    )


def produce_package_residual_from_identity_json(
    identity_json: str | bytes | Mapping[str, Any] | None,
    *,
    residual_verdict: str,
    package_tree_sha: str | None = None,
) -> Any | None:
    """Parse harness_identity_json and produce residual materials; None if absent."""

    if identity_json is None:
        return None
    bag: Mapping[str, Any] | None
    if isinstance(identity_json, Mapping):
        bag = identity_json
    elif isinstance(identity_json, (str, bytes)):
        import json

        try:
            raw = identity_json if isinstance(identity_json, str) else identity_json.decode("utf-8")
            parsed = json.loads(raw)
        except (TypeError, ValueError, UnicodeDecodeError):
            return None
        if not isinstance(parsed, dict):
            return None
        bag = parsed
    else:
        return None
    try:
        return produce_package_residual_from_identity(
            bag,
            residual_verdict=residual_verdict,
            package_tree_sha=package_tree_sha,
        )
    except (ProductHarnessAdmissionError, TypeError, ValueError):
        return None


__all__ = [
    "PARITY_HARNESS_KIND",
    "PRODUCT_ENTRY_SCRIPT_MARKERS",
    "PRODUCT_HARNESS_KIND",
    "ProductHarnessAdmissionError",
    "REFUSE_EMPTY_RULES",
    "REFUSE_EMPTY_ZIP",
    "REFUSE_MISSING_ENTRY_SCRIPT",
    "REFUSE_MISSING_RULES",
    "REFUSE_MISSING_ZIP",
    "REFUSE_OR_BEFORE_RULES",
    "REFUSE_PARITY_HARNESS",
    "REFUSE_UNMEASURED_HOST",
    "ReviewSessionIdentity",
    "RulesPackDigest",
    "SESSION_IDENTITY_SCHEMA_V1",
    "admit_product_review_entry",
    "bind_measured_residual_into_review_materials",
    "digest_agent_zip",
    "digest_entry_script",
    "inventory_product_vs_parity",
    "is_parity_harness_entry",
    "is_product_entry_script",
    "load_rules_pack_digests",
    "map_decision_verdict_to_residual_verdict",
    "produce_package_residual_from_identity",
    "produce_package_residual_from_identity_json",
    "refuse_parity_harness_as_review",
    "require_rules_before_openrouter",
    "sha256_hex",
]
