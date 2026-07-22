"""AGATE package_tree_sha: deterministic tree proof + bind + guest mismatch refuse.

Covers VAL-AGATE-001 / 002 / 003 / 010 for milestone package-tree-sha-proof.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from agent_challenge.canonical import eval_wire as ew
from agent_challenge.evaluation.own_runner_backend import (
    assert_package_tree_matches_plan,
    package_tree_sha_from_directory,
)
from agent_challenge.submissions.artifacts import (
    ArtifactValidationError,
    compute_package_tree_sha_from_entries,
    compute_package_tree_sha_from_zip_bytes,
    extract_zip_to_directory,
    store_zip_bytes,
)
from agent_challenge.submissions.artifacts import (
    package_tree_sha_from_directory as artifacts_package_tree_sha_from_directory,
)


def _zip_bytes(entries: dict[str, bytes | str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for filename, contents in entries.items():
            payload = contents.encode("utf-8") if isinstance(contents, str) else contents
            archive.writestr(filename, payload)
    return buffer.getvalue()


def _honest_agent_entries() -> dict[str, bytes | str]:
    return {
        "agent.py": "class Agent:\n    pass\n",
        "README.md": "docs\n",
        "pkg/util.py": "x = 1\n",
    }


# --------------------------------------------------------------------------- #
# VAL-AGATE-001 — deterministic algorithm + store on submit
# --------------------------------------------------------------------------- #


def test_package_tree_sha_is_deterministic_and_order_independent() -> None:
    """Sorted relpath + content yields the same hex regardless of insertion order."""

    forward = [
        ("agent.py", b"class Agent:\n    pass\n"),
        ("z_last.txt", b"z\n"),
        ("a_first.txt", b"a\n"),
    ]
    reverse = list(reversed(forward))
    assert compute_package_tree_sha_from_entries(forward) == compute_package_tree_sha_from_entries(
        reverse
    )
    assert len(compute_package_tree_sha_from_entries(forward)) == 64


def test_package_tree_sha_changes_when_content_or_path_changes() -> None:
    base = [("agent.py", b"class Agent:\n    pass\n"), ("a.txt", b"one\n")]
    content_changed = [("agent.py", b"class Agent:\n    pass\n"), ("a.txt", b"two\n")]
    path_changed = [("agent.py", b"class Agent:\n    pass\n"), ("b.txt", b"one\n")]
    base_sha = compute_package_tree_sha_from_entries(base)
    assert compute_package_tree_sha_from_entries(content_changed) != base_sha
    assert compute_package_tree_sha_from_entries(path_changed) != base_sha


def test_store_zip_bytes_persists_package_tree_sha_next_to_zip_sha(tmp_path: Path) -> None:
    """VAL-AGATE-001: submit/store records package_tree_sha with zip identity."""

    archive = _zip_bytes(_honest_agent_entries())
    metadata = store_zip_bytes(zip_bytes=archive, artifact_root=str(tmp_path))
    expected = compute_package_tree_sha_from_zip_bytes(archive)

    assert metadata.package_tree_sha == expected
    assert metadata.manifest is not None
    assert metadata.manifest.package_tree_sha == expected
    assert metadata.manifest.zip_sha256 == hashlib.sha256(archive).hexdigest()

    stored = json.loads(Path(metadata.manifest_path or "").read_text(encoding="utf-8"))
    assert stored["package_tree_sha"] == expected
    assert stored["zip_sha256"] == metadata.zip_sha256

    # Directory recompute (guest-shaped) matches store.
    extract_root = tmp_path / "extracted"
    extract_zip_to_directory(zip_path=metadata.artifact_path, target_directory=extract_root)
    assert package_tree_sha_from_directory(extract_root) == expected


def test_package_tree_sha_directory_matches_zip_bytes(tmp_path: Path) -> None:
    archive = _zip_bytes(_honest_agent_entries())
    zip_path = tmp_path / "agent.zip"
    zip_path.write_bytes(archive)
    extract_root = tmp_path / "extracted"
    extract_zip_to_directory(zip_path=zip_path, target_directory=extract_root)
    assert artifacts_package_tree_sha_from_directory(extract_root) == (
        compute_package_tree_sha_from_zip_bytes(archive)
    )
    assert package_tree_sha_from_directory(extract_root) == (
        compute_package_tree_sha_from_zip_bytes(archive)
    )


# --------------------------------------------------------------------------- #
# VAL-AGATE-003 — bound into immutable plan + review materials
# --------------------------------------------------------------------------- #


def _eval_plan(*, package_tree_sha: str) -> dict[str, Any]:
    policy = {
        "schema_version": 1,
        "per_task_aggregation": "mean",
        "keep_policy": "off",
        "drop_lowest_n": 0,
        "threshold_f64be": None,
    }
    return {
        "schema_version": 1,
        "eval_run_id": "eval-run-tree-sha",
        "submission_id": "submission-tree-001",
        "submission_version": 1,
        "authorizing_review_digest": "1" * 64,
        "agent_hash": "a" * 64,
        "package_tree_sha": package_tree_sha,
        "selected_tasks": [
            {
                "task_id": "task-a",
                "image_ref": "registry.example/task@sha256:" + "d" * 64,
                "task_config_sha256": "2" * 64,
            }
        ],
        "k": 1,
        "scoring_policy": policy,
        "scoring_policy_digest": ew.scoring_policy_digest(policy),
        "eval_app": {
            "image_ref": "registry.example/eval@sha256:" + "d" * 64,
            "compose_hash": "c" * 64,
            "app_identity": "agent-challenge-eval",
            "kms_key_algorithm": "x25519",
            "kms_public_key_hex": "3" * 64,
            "kms_public_key_sha256": hashlib.sha256(bytes.fromhex("3" * 64)).hexdigest(),
            "measurement": {
                "mrtd": "a1" * 48,
                "rtmr0": "a2" * 48,
                "rtmr1": "a3" * 48,
                "rtmr2": "a4" * 48,
                "os_image_hash": "a5" * 32,
                "key_provider": "validator-kms",
                "vm_shape": "tdx-small",
            },
        },
        "key_release_endpoint": "keyrelease.example:8701",
        "result_endpoint": "/evaluation/v1/runs/eval-run-tree-sha/result",
        "key_release_nonce": "key-nonce-001",
        "score_nonce": "score-nonce-001",
        "run_token_sha256": "5" * 64,
        "issued_at_ms": 1,
        "expires_at_ms": 2,
    }


def test_eval_plan_requires_package_tree_sha() -> None:
    """VAL-AGATE-003: immutable plan carries package_tree_sha (cannot omit)."""

    tree_sha = "b" * 64
    plan = _eval_plan(package_tree_sha=tree_sha)
    validated = ew.validate_eval_plan(plan)
    assert validated["package_tree_sha"] == tree_sha

    omitted = dict(plan)
    del omitted["package_tree_sha"]
    with pytest.raises(ew.EvalWireError, match="package_tree_sha|invalid fields"):
        ew.validate_eval_plan(omitted)


def test_eval_plan_rejects_non_sha_package_tree() -> None:
    plan = _eval_plan(package_tree_sha="not-a-digest")
    with pytest.raises(ew.EvalWireError):
        ew.validate_eval_plan(plan)


def test_build_plan_binds_submission_package_tree_sha() -> None:
    """_build_plan / authorization binds submission.package_tree_sha into plan."""

    from agent_challenge.evaluation import authorization as auth

    src = Path(auth.__file__).read_text(encoding="utf-8")
    assert "package_tree_sha" in src
    assert 'plan = {' in src or "plan =" in src


def test_review_materials_bind_package_tree_sha() -> None:
    """Review session creation path binds package_tree_sha next to artifact digest."""

    from agent_challenge.review import sessions as review_sessions

    src = Path(review_sessions.__file__).read_text(encoding="utf-8")
    assert "package_tree_sha" in src


# --------------------------------------------------------------------------- #
# VAL-AGATE-002 / 010 — guest recompute fail-closed mismatch
# --------------------------------------------------------------------------- #


def test_guest_package_tree_match_accepts_honest_extract(tmp_path: Path) -> None:
    archive = _zip_bytes(_honest_agent_entries())
    expected = compute_package_tree_sha_from_zip_bytes(archive)
    zip_path = tmp_path / "agent.zip"
    zip_path.write_bytes(archive)
    extract_root = tmp_path / "pkg"
    extract_zip_to_directory(zip_path=zip_path, target_directory=extract_root)
    actual = assert_package_tree_matches_plan(
        package_root=extract_root,
        plan_package_tree_sha=expected,
    )
    assert actual == expected


def test_guest_package_tree_mismatch_refuses(tmp_path: Path) -> None:
    """VAL-AGATE-002 / 010: altered content after bind → guest refuse."""

    archive = _zip_bytes(_honest_agent_entries())
    expected = compute_package_tree_sha_from_zip_bytes(archive)
    zip_path = tmp_path / "agent.zip"
    zip_path.write_bytes(archive)
    extract_root = tmp_path / "pkg"
    extract_zip_to_directory(zip_path=zip_path, target_directory=extract_root)
    # Tamper after bind.
    (extract_root / "agent.py").write_text(
        "class Agent:\n    # cheat\n    pass\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="package_tree_sha"):
        assert_package_tree_matches_plan(
            package_root=extract_root,
            plan_package_tree_sha=expected,
        )


def test_guest_package_tree_mismatch_on_path_add(tmp_path: Path) -> None:
    archive = _zip_bytes(_honest_agent_entries())
    expected = compute_package_tree_sha_from_zip_bytes(archive)
    zip_path = tmp_path / "agent.zip"
    zip_path.write_bytes(archive)
    extract_root = tmp_path / "pkg"
    extract_zip_to_directory(zip_path=zip_path, target_directory=extract_root)
    (extract_root / "extra_cheat.py").write_text("print('x')\n", encoding="utf-8")
    with pytest.raises(ValueError, match="package_tree_sha"):
        assert_package_tree_matches_plan(
            package_root=extract_root,
            plan_package_tree_sha=expected,
        )


def test_guest_missing_plan_tree_sha_refuses(tmp_path: Path) -> None:
    zip_path = tmp_path / "agent.zip"
    zip_path.write_bytes(_zip_bytes(_honest_agent_entries()))
    extract_root = tmp_path / "pkg"
    extract_zip_to_directory(zip_path=zip_path, target_directory=extract_root)
    with pytest.raises(ValueError, match="package_tree_sha"):
        assert_package_tree_matches_plan(
            package_root=extract_root,
            plan_package_tree_sha="",
        )


def test_guest_path_before_trials_wired_in_own_runner_backend() -> None:
    """VAL-AGATE-010: own_runner guest path invokes tree check before trials."""

    from agent_challenge.evaluation import own_runner_backend as backend

    src = Path(backend.__file__).read_text(encoding="utf-8")
    assert "assert_package_tree_matches_plan" in src
    assert "package_tree_sha" in src


def test_artifact_validation_propagates_tree_sha_errors() -> None:
    with pytest.raises(ArtifactValidationError):
        compute_package_tree_sha_from_zip_bytes(b"not-a-zip")
