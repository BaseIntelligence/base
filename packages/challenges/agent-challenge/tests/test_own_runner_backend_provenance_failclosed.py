"""Guest identity checks must hash real bytes — never env/declared digest echo.

The eval CVM previously accepted ``CHALLENGE_PHALA_AGENT_HASH`` (and the
package_tree_sha env twins) as proof of identity when no ZIP/package was on
disk. That is a tautology: the host injects the plan digest, the guest echoes
it back. These tests lock fail-closed behavior: missing bytes always raise.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from agent_challenge.evaluation import own_runner_backend as backend
from agent_challenge.evaluation.own_runner_backend import (
    PHALA_AGENT_HASH_ENV,
    assert_agent_artifact_matches_plan,
    assert_package_tree_matches_plan,
)
from agent_challenge.submissions.artifacts import compute_package_tree_sha_from_zip_bytes


def _zip_bytes(entries: dict[str, bytes | str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for filename, contents in entries.items():
            payload = contents.encode("utf-8") if isinstance(contents, str) else contents
            archive.writestr(filename, payload)
    return buffer.getvalue()


def _honest_zip() -> bytes:
    return _zip_bytes(
        {
            "agent.py": "class Agent:\n    pass\n",
            "README.md": "docs\n",
        }
    )


# --------------------------------------------------------------------------- #
# agent_hash — real bytes required
# --------------------------------------------------------------------------- #


def test_agent_hash_env_echo_matching_plan_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Host-supplied CHALLENGE_PHALA_AGENT_HASH equal to the plan is not proof."""

    plan_hash = "a" * 64
    monkeypatch.setenv(PHALA_AGENT_HASH_ENV, plan_hash)
    with pytest.raises(ValueError, match=r"cannot verify|unavailable|artifact"):
        assert_agent_artifact_matches_plan(
            artifact_path=None,
            plan_agent_hash=plan_hash,
        )


def test_agent_hash_declared_param_matching_plan_is_rejected() -> None:
    """A declared_agent_hash kwarg equal to the plan is not proof either."""

    plan_hash = "b" * 64
    with pytest.raises(TypeError):
        # Parameter removed: callers must supply on-disk bytes.
        assert_agent_artifact_matches_plan(  # type: ignore[call-arg]
            artifact_path=None,
            plan_agent_hash=plan_hash,
            declared_agent_hash=plan_hash,
        )


def test_agent_hash_missing_bytes_raises_without_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(PHALA_AGENT_HASH_ENV, raising=False)
    with pytest.raises(ValueError, match=r"cannot verify|unavailable|artifact"):
        assert_agent_artifact_matches_plan(
            artifact_path=None,
            plan_agent_hash="c" * 64,
        )


def test_agent_hash_real_zip_bytes_match(tmp_path: Path) -> None:
    payload = b"PK\x03\x04real-agent-bytes"
    zip_path = tmp_path / "agent.zip"
    zip_path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    assert (
        assert_agent_artifact_matches_plan(
            artifact_path=zip_path,
            plan_agent_hash=digest,
        )
        == digest
    )


def test_agent_hash_real_zip_bytes_mismatch(tmp_path: Path) -> None:
    zip_path = tmp_path / "agent.zip"
    zip_path.write_bytes(b"agent-a")
    with pytest.raises(ValueError, match="agent artifact"):
        assert_agent_artifact_matches_plan(
            artifact_path=zip_path,
            plan_agent_hash=hashlib.sha256(b"agent-b").hexdigest(),
        )


# --------------------------------------------------------------------------- #
# package_tree_sha — real bytes required
# --------------------------------------------------------------------------- #


def test_package_tree_env_echo_matching_plan_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Host-supplied package_tree_sha env equal to the plan is not proof."""

    archive = _honest_zip()
    expected = compute_package_tree_sha_from_zip_bytes(archive)
    monkeypatch.setenv("CHALLENGE_PHALA_PACKAGE_TREE_SHA", expected)
    monkeypatch.setenv("CHALLENGE_AGENT_PACKAGE_TREE_SHA", expected)
    with pytest.raises(ValueError, match=r"cannot verify|unavailable|package"):
        assert_package_tree_matches_plan(
            package_root=None,
            plan_package_tree_sha=expected,
            zip_path=None,
        )


def test_package_tree_missing_bytes_raises_without_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CHALLENGE_PHALA_PACKAGE_TREE_SHA", raising=False)
    monkeypatch.delenv("CHALLENGE_AGENT_PACKAGE_TREE_SHA", raising=False)
    with pytest.raises(ValueError, match=r"cannot verify|unavailable|package"):
        assert_package_tree_matches_plan(
            package_root=None,
            plan_package_tree_sha="d" * 64,
            zip_path=None,
        )


def test_package_tree_zip_path_recompute_accepts(tmp_path: Path) -> None:
    archive = _honest_zip()
    expected = compute_package_tree_sha_from_zip_bytes(archive)
    zip_path = tmp_path / "agent.zip"
    zip_path.write_bytes(archive)
    actual = assert_package_tree_matches_plan(
        package_root=None,
        plan_package_tree_sha=expected,
        zip_path=zip_path,
    )
    assert actual == expected


def test_source_has_no_env_echo_provenance_fallback() -> None:
    """Static guard: the two assert helpers must not read digest env as proof."""

    src = Path(backend.__file__).read_text(encoding="utf-8")
    # Locate the two function bodies by slicing between defs.
    agent_fn = src.split("def assert_agent_artifact_matches_plan", 1)[1].split("\ndef ", 1)[0]
    tree_fn = src.split("def assert_package_tree_matches_plan", 1)[1].split("\ndef ", 1)[0]
    assert "os.environ.get(PHALA_AGENT_HASH_ENV)" not in agent_fn
    assert "declared_agent_hash" not in agent_fn
    assert "CHALLENGE_PHALA_PACKAGE_TREE_SHA" not in tree_fn
    assert "CHALLENGE_AGENT_PACKAGE_TREE_SHA" not in tree_fn
    assert "declared package_tree_sha" not in tree_fn
