"""Behavioral tests for the live-subset registry-ref side manifest + resolver.

The frozen golden manifest (``golden/dataset-digest.json``) pins each
Terminal-Bench task by a *bare content digest* (``harbor_registry_ref =
"sha256:<64hex>"``) with NO repository, so an in-CVM DooD orchestrator cannot
``docker pull`` it. For a live smoke E2E a small deterministic subset of task
images is published to a pullable, digest-pinned registry ref and recorded in a
SEPARATE side manifest (``golden/live-registry-refs.json``) so the frozen
content digests / canonical measurement stay byte-identical.

These tests pin:
  * the resolver is OPT-IN and fail-closed (no manifest configured -> no refs, so
    offline/flag-off behavior is byte-identical);
  * every published ref is repository-qualified AND digest-pinned (a bare
    ``sha256:...`` content digest or a floating tag is rejected);
  * the shipped side manifest is a strict subset of the frozen golden tasks and
    never mutates ``dataset-digest.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_challenge.canonical import compose as c
from agent_challenge.canonical import live_registry as lr

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = REPO_ROOT / "golden"
GOLDEN_MANIFEST = GOLDEN_DIR / "dataset-digest.json"
LIVE_MANIFEST = GOLDEN_DIR / lr.LIVE_REGISTRY_FILENAME

# The frozen golden dataset invariants this feature must NOT perturb.
FROZEN_CANONICAL_CONTENT_DIGEST = "8da006d76bcf59c2af3f36ed4420192d3930bda43683f32d80013a6ee5e7e02d"
FROZEN_TASK_COUNT = 89

_GOOD_REF = "docker.io/mathiiss/agent-challenge-tb21-x@sha256:" + ("a" * 64)


# --------------------------------------------------------------------------- #
# ref validation: repository-qualified AND digest-pinned
# --------------------------------------------------------------------------- #
def test_pullable_ref_accepts_repo_digest():
    assert lr.is_pullable_ref(_GOOD_REF)
    assert lr.assert_pullable_ref(_GOOD_REF) == _GOOD_REF


@pytest.mark.parametrize(
    "bad",
    [
        "sha256:" + ("a" * 64),  # bare content digest (the golden behavior) - not pullable
        "docker.io/mathiiss/x:latest",  # floating tag, not digest-pinned
        "docker.io/mathiiss/x@sha256:" + ("a" * 63),  # short digest
        "plainname@sha256:" + ("a" * 64),  # no repository/namespace ('/')
        "",
        123,
    ],
)
def test_pullable_ref_rejects_non_pullable(bad):
    assert not lr.is_pullable_ref(bad)
    with pytest.raises(lr.LiveRegistryError):
        lr.assert_pullable_ref(bad)


# --------------------------------------------------------------------------- #
# opt-in, fail-closed resolution (byte-identical offline)
# --------------------------------------------------------------------------- #
def test_resolve_refs_is_empty_when_unconfigured():
    # No explicit path and no env => no live refs (legacy behavior preserved).
    assert lr.resolve_live_registry_refs() == {}
    assert lr.resolve_live_registry_refs(env={}) == {}
    assert lr.resolve_live_registry_refs(env={"UNRELATED": "1"}) == {}


def test_resolve_refs_from_explicit_path(tmp_path):
    manifest = {
        "schema": lr.LIVE_REGISTRY_SCHEMA,
        "tasks": {"foo": {"registry_ref": _GOOD_REF}},
    }
    p = tmp_path / "live.json"
    p.write_text(json.dumps(manifest), encoding="utf-8")
    refs = lr.resolve_live_registry_refs(path=p)
    assert refs == {"foo": _GOOD_REF}


def test_resolve_refs_from_env(tmp_path):
    manifest = {"tasks": {"foo": {"registry_ref": _GOOD_REF}}}
    p = tmp_path / "live.json"
    p.write_text(json.dumps(manifest), encoding="utf-8")
    refs = lr.resolve_live_registry_refs(env={lr.LIVE_REGISTRY_ENV: str(p)})
    assert refs == {"foo": _GOOD_REF}


def test_configured_but_broken_manifest_raises(tmp_path):
    p = tmp_path / "missing.json"
    with pytest.raises(lr.LiveRegistryError):
        lr.resolve_live_registry_refs(path=p)


def test_parse_rejects_bare_digest_ref():
    with pytest.raises(lr.LiveRegistryError):
        lr.parse_live_registry({"tasks": {"foo": {"registry_ref": "sha256:" + "a" * 64}}})


# --------------------------------------------------------------------------- #
# task_id resolution (bare + dataset-prefixed)
# --------------------------------------------------------------------------- #
def test_resolve_task_image_bare_and_prefixed():
    reg = lr.parse_live_registry({"tasks": {"foo": {"registry_ref": _GOOD_REF}}})
    assert reg.resolve_task_image("foo") == _GOOD_REF
    # A dataset-prefixed id resolves to the same bare-keyed entry.
    assert reg.resolve_task_image("terminal-bench/foo") == _GOOD_REF
    # Unknown task -> None (fail-closed to legacy behavior).
    assert reg.resolve_task_image("does-not-exist") is None


def test_string_ref_shorthand_is_accepted():
    reg = lr.parse_live_registry({"tasks": {"foo": _GOOD_REF}})
    assert reg.resolve_task_image("foo") == _GOOD_REF


# --------------------------------------------------------------------------- #
# the SHIPPED side manifest
# --------------------------------------------------------------------------- #
def test_shipped_live_manifest_is_valid_and_pullable():
    reg = lr.load_live_registry(LIVE_MANIFEST)
    assert reg.task_refs, "shipped live manifest has no task refs"
    for task_id, ref in reg.task_refs.items():
        assert lr.is_pullable_ref(ref), (task_id, ref)


def test_shipped_orchestrator_image_is_digest_pinned():
    reg = lr.load_live_registry(LIVE_MANIFEST)
    assert reg.orchestrator_image is not None
    assert lr.is_pullable_ref(reg.orchestrator_image)
    # The deploy path's digest-pin guard accepts it (no bare tag).
    assert c.assert_digest_pinned(reg.orchestrator_image) == reg.orchestrator_image


def test_shipped_live_subset_is_subset_of_golden_and_small():
    golden = json.loads(GOLDEN_MANIFEST.read_text(encoding="utf-8"))
    golden_tasks = set(golden["tasks"])
    reg = lr.load_live_registry(LIVE_MANIFEST)
    assert set(reg.task_refs) <= golden_tasks
    assert 1 <= len(reg.task_refs) <= 5  # a small deterministic subset


def test_shipped_live_manifest_content_digests_match_golden():
    golden = json.loads(GOLDEN_MANIFEST.read_text(encoding="utf-8"))
    live = json.loads(LIVE_MANIFEST.read_text(encoding="utf-8"))
    for task_id, entry in live["tasks"].items():
        recorded = entry.get("content_digest_sha256")
        assert recorded is not None, task_id
        assert recorded == golden["tasks"][task_id]["content_digest_sha256"], task_id


def test_golden_dataset_digest_is_unperturbed():
    # Discriminator: this feature must never mutate the frozen golden manifest.
    golden = json.loads(GOLDEN_MANIFEST.read_text(encoding="utf-8"))
    assert golden["canonical_content_digest_sha256"] == FROZEN_CANONICAL_CONTENT_DIGEST
    assert golden["task_count"] == FROZEN_TASK_COUNT
    assert len(golden["tasks"]) == FROZEN_TASK_COUNT
