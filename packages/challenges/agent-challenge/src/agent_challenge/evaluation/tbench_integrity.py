"""Terminal-Bench 2.1 integrity + allow_internet product policy (VAL-ACLOCK-011..016).

Frozen task identity is **content-addressed**. Eval never network-fetches task
trees; digests come from ``golden/dataset-digest.json``; plans bind
``selected_tasks`` from the **validator** prepare path only. Miners cannot
supply an alternate task URL/git.

## allow_internet product policy (locked choice)

Inventory of the frozen Terminal-Bench 2.1 bake shows **every** task tree sets
``[environment].allow_internet = true`` (harbor parity for package installs and
public data pulls). Rewriting all task.toml files is explicitly out of scope.

**Chosen product default:** ``retain_task_authored_with_review_risk``.

Rationale (least-breaking that still blocks obvious cheat):

1. Forcing ``--network none`` on scored runs would break legitimate TB tasks that
   require egress under the frozen defs (apt/pip/curl/data pulls).
2. Obvious cheat surfaces are already closed elsewhere:
   - no miner task URL / git override (this module + eval plan schema)
   - local cache + digest fail-closed (``own_runner.taskdefs``)
   - validator-authored ``selected_tasks`` only (eval prepare / plan wire)
   - agent env allowlist = OpenRouter key + cost limit only
   - miner env rejects PROXY/URL/HOST injection
   - review ``.rules`` hardcoding/anti-cheat: answer hardcoding remains cheat
3. Residual risk: agent process in a task container with egress may call the
   public internet (and OpenRouter). That is **retained and documented** as
   review-class residual, not silent. Operators may still opt into
   ``CHALLENGE_SCORED_TASK_NETWORK_RESTRICT=1`` to force ``network none`` for
   lab experiments (non-default; breaks TB parity).

Harness pins (dataset digest, OpenRouter origin, review joinbase URL,
DOCKER_HOST unix, gateway forbid) remain **required anti-cheat**, distinct from
miner answer hardcoding which remains **cheat**.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from agent_challenge.evaluation.benchmarks import (
    TERMINAL_BENCH_2_1_DIGEST_PATH,
    TERMINAL_BENCH_2_1_FALLBACK_TASK_IDS,
    load_canonical_terminal_bench_2_1_task_ids,
    validate_fallback_task_ids,
)
from agent_challenge.evaluation.own_runner.taskdefs import (
    ResourceLimits,
    bare_task_name,
    load_dataset_digest,
)

# --------------------------------------------------------------------------- #
# Policy constants (locked product language)
# --------------------------------------------------------------------------- #
ALLOW_INTERNET_POLICY_ID: Final[str] = "retain_task_authored_with_review_risk"
ALLOW_INTERNET_POLICY_LABEL: Final[str] = (
    "Retain frozen task-authored allow_internet; document residual egress risk"
)

#: Keys that must never appear as miner- or plan-supplied alternate task sources.
FORBIDDEN_TASK_SOURCE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "task_url",
        "task_git",
        "task_git_url",
        "git_url",
        "dataset_url",
        "tasks_url",
        "task_source_url",
        "harbor_url",
        "terminal_bench_url",
        "task_repo",
        "task_repository",
        "custom_task_url",
        "override_task_url",
    }
)

#: Harness pins that remain required anti-cheat (not "hardcoding cheat").
REQUIRED_HARNESS_PINS: Final[tuple[str, ...]] = (
    "golden/dataset-digest.json content-addressed TB 2.1 digests",
    "local baked task-cache (no network fetch at eval)",
    "OpenRouter origin/TLS pin for measured LLM",
    "REVIEW_API_BASE_URL hard-pin to chain.joinbase.ai AC path",
    "DOCKER_HOST unix-only",
    "Base LLM gateway forbidden on scored path",
    "miner env keys/tokens only (no URL/proxy/host)",
    "validator-authored selected_tasks + KR RA-TLS authority",
)

ANSWER_HARDCODING_IS_CHEAT: Final[str] = (
    "Miner branching on task id / hardcoding answers / reading hidden tests is "
    "cheat under .rules/hardcoding.md and .rules/anti-cheat.md. Harness pins "
    "are required anti-cheat and must not be loosened."
)

# Env opt-in to force network isolation on scored task containers (non-default).
_SCORED_NETWORK_RESTRICT_ENV: Final[str] = "CHALLENGE_SCORED_TASK_NETWORK_RESTRICT"


class TbenchIntegrityError(ValueError):
    """Fail-closed integrity violation for TB 2.1 contracts."""


# --------------------------------------------------------------------------- #
# Digest / fallback set
# --------------------------------------------------------------------------- #
def frozen_digest_path() -> Path:
    return TERMINAL_BENCH_2_1_DIGEST_PATH


def load_frozen_task_ids() -> frozenset[str]:
    """Bare + prefixed ids present in the frozen digest manifest."""

    manifest = load_dataset_digest(frozen_digest_path())
    bare = frozenset(str(name) for name in manifest["tasks"])
    prefixed = frozenset(f"terminal-bench/{name}" for name in bare)
    return bare | prefixed


def assert_fallback_ids_subset_of_frozen(
    fallback_ids: Sequence[str] = TERMINAL_BENCH_2_1_FALLBACK_TASK_IDS,
) -> None:
    """VAL-ACLOCK-013: fallback selection IDs ⊆ frozen digest task set."""

    canonical = load_canonical_terminal_bench_2_1_task_ids()
    validate_fallback_task_ids(fallback_ids, canonical=canonical)


def assert_selected_task_ids_in_frozen(task_ids: Iterable[str]) -> None:
    """Every selected task id must resolve into the frozen digest set."""

    frozen = load_frozen_task_ids()
    unknown = [
        task_id
        for task_id in task_ids
        if task_id not in frozen and bare_task_name(task_id) not in frozen
    ]
    if unknown:
        raise TbenchIntegrityError(
            f"selected task ids not in frozen digest set: {unknown}"
        )


# --------------------------------------------------------------------------- #
# Miner cannot supply alternate task URL/git
# --------------------------------------------------------------------------- #
def forbidden_task_source_keys_present(mapping: Mapping[str, Any] | None) -> list[str]:
    """Return forbidden task-source keys present in a mapping (case-insensitive)."""

    if not mapping:
        return []
    lowered = {str(key).lower(): str(key) for key in mapping}
    hits: list[str] = []
    for forbidden in sorted(FORBIDDEN_TASK_SOURCE_KEYS):
        if forbidden in lowered:
            hits.append(lowered[forbidden])
    return hits


def assert_no_miner_task_source_fields(mapping: Mapping[str, Any] | None) -> None:
    """VAL-ACLOCK-012: reject miner/plan fields that supply alternate task URL/git."""

    hits = forbidden_task_source_keys_present(mapping)
    if hits:
        raise TbenchIntegrityError(
            f"miner/plan must not supply alternate task sources: {hits}"
        )


def selected_task_item_allowed_keys() -> frozenset[str]:
    """Eval plan ``selected_tasks[]`` is schema-closed (validator fields only)."""

    return frozenset({"task_id", "image_ref", "task_config_sha256"})


# --------------------------------------------------------------------------- #
# No network fetch at eval (module / loader contract)
# --------------------------------------------------------------------------- #
_TASKDEFS_MODULE_PATH = (
    Path(__file__).resolve().parent / "own_runner" / "taskdefs.py"
)

# Tokens that would indicate a network client in the task-def loader.
_NETWORK_CLIENT_MARKERS: Final[tuple[str, ...]] = (
    "import urllib",
    "from urllib",
    "import requests",
    "from requests",
    "import httpx",
    "from httpx",
    "urlopen(",
    "aiohttp",
)


def assert_taskdefs_loader_is_local_only(source_text: str | None = None) -> None:
    """VAL-ACLOCK-011: taskdefs module must not perform network fetches."""

    text = source_text
    if text is None:
        text = _TASKDEFS_MODULE_PATH.read_text(encoding="utf-8")
    lowered = text.lower()
    # Positive contract language must remain.
    if "no network" not in lowered and "never network" not in lowered:
        raise TbenchIntegrityError("taskdefs module lost no-network contract language")
    for marker in _NETWORK_CLIENT_MARKERS:
        if marker.lower() in lowered:
            raise TbenchIntegrityError(
                f"taskdefs loader must not use network client marker {marker!r}"
            )


# --------------------------------------------------------------------------- #
# allow_internet inventory + network gate
# --------------------------------------------------------------------------- #
def inventory_allow_internet(
    cache_root: Path,
) -> dict[str, Any]:
    """Scan a baked task-cache tree for ``allow_internet`` declarations."""

    if not cache_root.is_dir():
        raise TbenchIntegrityError(f"task cache root missing: {cache_root}")
    allow_true: list[str] = []
    allow_false: list[str] = []
    unset: list[str] = []
    for child in sorted(cache_root.iterdir()):
        if not child.is_dir():
            continue
        toml_path = child / "task.toml"
        if not toml_path.is_file():
            # harbor layout: <name>/<content_hash>/task.toml
            nested = list(child.glob("*/task.toml"))
            if len(nested) != 1:
                continue
            toml_path = nested[0]
        try:
            with open(toml_path, "rb") as handle:
                meta = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise TbenchIntegrityError(f"unreadable task.toml {toml_path}: {exc}") from exc
        env = meta.get("environment") if isinstance(meta, dict) else None
        value = env.get("allow_internet") if isinstance(env, dict) else None
        name = child.name
        if value is True:
            allow_true.append(name)
        elif value is False:
            allow_false.append(name)
        else:
            unset.append(name)
    return {
        "policy_id": ALLOW_INTERNET_POLICY_ID,
        "policy_label": ALLOW_INTERNET_POLICY_LABEL,
        "cache_root": str(cache_root),
        "allow_internet_true": allow_true,
        "allow_internet_false": allow_false,
        "allow_internet_unset": unset,
        "counts": {
            "true": len(allow_true),
            "false": len(allow_false),
            "unset": len(unset),
            "scanned": len(allow_true) + len(allow_false) + len(unset),
        },
    }


def scored_task_network_restrict_enabled(
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Non-default lab opt-in: force ``--network none`` on scored task containers."""

    env = environ if environ is not None else os.environ
    raw = str(env.get(_SCORED_NETWORK_RESTRICT_ENV, "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def effective_network_arg(
    resources: ResourceLimits,
    *,
    scored_run: bool = True,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Map task resources → docker ``--network`` under product policy.

    Default production: honor frozen task ``allow_internet`` (harbor parity).
    When ``CHALLENGE_SCORED_TASK_NETWORK_RESTRICT`` is enabled and ``scored_run``,
    force ``\"none\"`` regardless of task.toml (lab fail-closed; breaks TB parity).

    Miners cannot flip this via plan/env task_url fields — only ops env + task
    authored resources reach this function.
    """

    if scored_run and scored_task_network_restrict_enabled(environ):
        return "none"
    if resources.allow_internet:
        return None
    return "none"


def allow_internet_policy_snapshot() -> dict[str, Any]:
    """Machine-readable policy object for docs/tests/evidence."""

    return {
        "policy_id": ALLOW_INTERNET_POLICY_ID,
        "policy_label": ALLOW_INTERNET_POLICY_LABEL,
        "default_scored_behavior": "honor_task_toml_allow_internet",
        "opt_in_restrict_env": _SCORED_NETWORK_RESTRICT_ENV,
        "opt_in_restrict_default": False,
        "forbidden_miner_task_source_keys": sorted(FORBIDDEN_TASK_SOURCE_KEYS),
        "required_harness_pins": list(REQUIRED_HARNESS_PINS),
        "answer_hardcoding": ANSWER_HARDCODING_IS_CHEAT,
        "selected_tasks_author": "validator_prepare_only",
        "task_def_network_at_eval": False,
    }
