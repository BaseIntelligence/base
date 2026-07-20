"""Backend wiring for the live-registry DooD task-image resolution.

The own-runner backend threads the opt-in live-subset registry refs into the
default preparer's :class:`TaskContainerBuilder`, and ``main`` resolves them from
the environment. With no manifest configured the resolution is empty, so the
in-CVM / offline path is byte-identical to legacy behavior.
"""

from __future__ import annotations

import json

from agent_challenge.canonical.live_registry import LIVE_REGISTRY_ENV
from agent_challenge.evaluation import own_runner_backend as backend


def test_build_default_preparer_threads_live_refs_to_builder(monkeypatch):
    captured = {}

    class _SpyBuilder:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(backend, "TaskContainerBuilder", _SpyBuilder)

    ref = "docker.io/mathiiss/agent-challenge-tb21-foo@sha256:" + ("a" * 64)
    backend._build_default_preparer(
        task_ids=[],
        cache_root=backend.DEFAULT_CACHE_ROOT,
        digest_manifest={"tasks": {}},
        digest_manifest_path=None,
        builder=None,
        agent_env=None,
        parsed_by_id={},
        live_registry_refs={"foo": ref},
    )
    assert captured.get("live_registry_refs") == {"foo": ref}


def test_build_default_preparer_defaults_to_no_live_refs(monkeypatch):
    captured = {}

    class _SpyBuilder:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(backend, "TaskContainerBuilder", _SpyBuilder)

    backend._build_default_preparer(
        task_ids=[],
        cache_root=backend.DEFAULT_CACHE_ROOT,
        digest_manifest={"tasks": {}},
        digest_manifest_path=None,
        builder=None,
        agent_env=None,
        parsed_by_id={},
    )
    # Byte-identical legacy default: no live refs configured.
    assert captured.get("live_registry_refs") in (None, {})


def test_main_resolves_live_refs_from_env(monkeypatch, tmp_path):
    ref = "docker.io/mathiiss/agent-challenge-tb21-foo@sha256:" + ("b" * 64)
    manifest = tmp_path / "live-registry-refs.json"
    manifest.write_text(json.dumps({"tasks": {"foo": {"registry_ref": ref}}}), encoding="utf-8")

    captured = {}

    async def _fake_run(**kwargs):
        captured.update(kwargs)

        class _R:  # minimal stand-in; _emit_job_result is stubbed out below
            pass

        return _R()

    monkeypatch.setattr(backend, "run_own_runner_job", _fake_run)
    monkeypatch.setattr(backend, "_acquire_golden_key_if_required", lambda: None)
    monkeypatch.setattr(backend, "_emit_job_result", lambda result, task_ids: 0)
    monkeypatch.setenv(LIVE_REGISTRY_ENV, str(manifest))

    rc = backend.main(["run", "--job-dir", str(tmp_path / "job"), "--task", "foo"])
    assert rc == 0
    assert captured.get("live_registry_refs") == {"foo": ref}


def test_main_no_live_refs_when_env_unset(monkeypatch, tmp_path):
    captured = {}

    async def _fake_run(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(backend, "run_own_runner_job", _fake_run)
    monkeypatch.setattr(backend, "_acquire_golden_key_if_required", lambda: None)
    monkeypatch.setattr(backend, "_emit_job_result", lambda result, task_ids: 0)
    monkeypatch.delenv(LIVE_REGISTRY_ENV, raising=False)

    rc = backend.main(["run", "--job-dir", str(tmp_path / "job"), "--task", "foo"])
    assert rc == 0
    assert captured.get("live_registry_refs") == {}
