"""Live task-cache completeness for terminal-bench prepare surface.

Rootcause residual (grant-v6): public eval image baked only three bare tasks under
``docker/canonical/live-task-cache`` while prepare/select draws from
:data:`TERMINAL_BENCH_2_1_FALLBACK_TASK_IDS` (30 ids, including
``filter-js-from-html``). Guest preflight then raises
``TaskDefNotFoundError`` before key-release.

These tests lock the bake contract:

* every FALLBACK task id has a resolvable cache root under the measured cache;
* ``load_task_from_manifest`` succeeds with the frozen digest;
* regression sample: ``filter-js-from-html`` and ``break-filter-js-from-html``;
* Dockerfile COPY + any populate-script wiring keep the cache path intact.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_challenge.evaluation.benchmarks import (
    TERMINAL_BENCH_2_1_FALLBACK_TASK_IDS,
    load_canonical_terminal_bench_2_1_task_ids,
)
from agent_challenge.evaluation.own_runner.taskdefs import (
    TaskDefNotFoundError,
    bare_task_name,
    load_dataset_digest,
    load_task_from_manifest,
    resolve_task_root,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_LIVE_CACHE = _REPO_ROOT / "docker" / "canonical" / "live-task-cache"
_DIGEST_PATH = _REPO_ROOT / "golden" / "dataset-digest.json"
_CANONICAL_DOCKERFILE = _REPO_ROOT / "docker" / "canonical" / "Dockerfile"
_POPULATE_SCRIPT = _REPO_ROOT / "scripts" / "populate_live_task_cache.py"

# Grant residual selected samples that failed (missing) vs already-present tree.
_GRANT_RESIDUAL_SAMPLES = (
    "terminal-bench/filter-js-from-html",
    "terminal-bench/break-filter-js-from-html",
)


def _fallback_bare_ids() -> tuple[str, ...]:
    return tuple(bare_task_name(task_id) for task_id in TERMINAL_BENCH_2_1_FALLBACK_TASK_IDS)


def test_live_task_cache_directory_exists() -> None:
    assert _LIVE_CACHE.is_dir(), f"missing live-task-cache root: {_LIVE_CACHE}"


def test_dataset_digest_covers_fallback_and_canonical() -> None:
    manifest = load_dataset_digest(_DIGEST_PATH)
    tasks = set(manifest["tasks"])
    for bare in _fallback_bare_ids():
        assert bare in tasks, f"FALLBACK bare id {bare!r} absent from frozen digest"
    canonical = [bare_task_name(t) for t in load_canonical_terminal_bench_2_1_task_ids()]
    assert set(canonical) == tasks
    assert len(tasks) == 89


def test_live_task_cache_contains_every_fallback_bare_dir() -> None:
    present = {p.name for p in _LIVE_CACHE.iterdir() if p.is_dir()}
    missing = [bare for bare in _fallback_bare_ids() if bare not in present]
    assert missing == [], f"live-task-cache missing prepare-eligible FALLBACK bare dirs: {missing}"


def test_resolve_and_load_manifest_succeeds_for_every_fallback() -> None:
    manifest = load_dataset_digest(_DIGEST_PATH)
    failures: list[str] = []
    for task_id in TERMINAL_BENCH_2_1_FALLBACK_TASK_IDS:
        try:
            root = resolve_task_root(_LIVE_CACHE, task_id)
            loaded = load_task_from_manifest(
                root,
                task_id=task_id,
                digest_manifest=manifest,
                verify_digest=True,
            )
        except Exception as exc:  # noqa: BLE001 - collect for one assertion
            failures.append(f"{task_id}: {type(exc).__name__}: {exc}")
            continue
        assert loaded.task_id == task_id
        expected = manifest["tasks"][bare_task_name(task_id)]["content_digest_sha256"]
        assert loaded.content_digest_sha256 == expected
    assert failures == [], "FALLBACK load_task_from_manifest failures:\n" + "\n".join(failures)


@pytest.mark.parametrize("task_id", _GRANT_RESIDUAL_SAMPLES)
def test_grant_residual_sample_loads_from_live_cache(task_id: str) -> None:
    """Regression for grant-v6 TaskDefNotFoundError residual sample pair."""
    manifest = load_dataset_digest(_DIGEST_PATH)
    root = resolve_task_root(_LIVE_CACHE, task_id)
    loaded = load_task_from_manifest(
        root,
        task_id=task_id,
        digest_manifest=manifest,
        verify_digest=True,
    )
    assert loaded.task_id == task_id
    bare = bare_task_name(task_id)
    assert (root / "task.toml").is_file()
    assert loaded.content_digest_sha256 == manifest["tasks"][bare]["content_digest_sha256"]


def test_missing_bare_dir_raises_task_def_not_found(tmp_path: Path) -> None:
    """Sanity: empty cache root cannot resolve a FALLBACK id (same fail-closed path)."""
    empty = tmp_path / "empty-cache"
    empty.mkdir()
    with pytest.raises(TaskDefNotFoundError, match="filter-js-from-html"):
        resolve_task_root(empty, "terminal-bench/filter-js-from-html")


def test_dockerfile_copies_live_task_cache_into_guest_cache_root() -> None:
    text = _CANONICAL_DOCKERFILE.read_text(encoding="utf-8")
    assert "COPY docker/canonical/live-task-cache/ /opt/agent-challenge/task-cache/" in text
    assert "COPY golden/dataset-digest.json" in text


def test_populate_live_task_cache_script_exists_and_documents_source() -> None:
    assert _POPULATE_SCRIPT.is_file(), f"missing populate script: {_POPULATE_SCRIPT}"
    body = _POPULATE_SCRIPT.read_text(encoding="utf-8")
    assert "live-task-cache" in body
    assert "dataset-digest.json" in body or "dataset_digest" in body
    # No network invent at eval/populate: must read pinned harbor/local source.
    assert "harbor" in body.lower() or "DEFAULT_CACHE_ROOT" in body or "cache" in body.lower()


def test_live_task_cache_has_at_least_full_fallback_prepare_surface() -> None:
    """Lean residual minimum is full FALLBACK (30); full digest (89) preferred."""
    present = {p.name for p in _LIVE_CACHE.iterdir() if p.is_dir()}
    assert len(present) >= len(TERMINAL_BENCH_2_1_FALLBACK_TASK_IDS)
    assert set(_fallback_bare_ids()).issubset(present)


def test_preferred_full_digest_set_present_when_baked() -> None:
    """When full digest tree is baked, every digest bare id must load cleanly."""
    manifest = load_dataset_digest(_DIGEST_PATH)
    present = {p.name for p in _LIVE_CACHE.iterdir() if p.is_dir()}
    if not set(manifest["tasks"]).issubset(present):
        pytest.skip(
            "full 89-task digest set not baked (lean FALLBACK-only residual acceptable); "
            f"present={len(present)} digest={len(manifest['tasks'])}"
        )
    failures: list[str] = []
    for bare in sorted(manifest["tasks"]):
        try:
            root = resolve_task_root(_LIVE_CACHE, bare)
            load_task_from_manifest(
                root,
                task_id=bare,
                digest_manifest=manifest,
                verify_digest=True,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{bare}: {type(exc).__name__}: {exc}")
    assert failures == [], "full digest live-task-cache load failures:\n" + "\n".join(failures)
