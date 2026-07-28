"""Pod boot contract pure-function unit tests (Lium IMAGE ENTRYPOINT path).

Scenarios (contract):
- S1 happy: full 40-char commit SHA accepted and normalized
- S2 happy: unambiguous 7+ hex short SHA accepted
- S3 edge: path traversal / shell metachar / spaces in SHA rejected
- S4 happy: https git URL accepted
- S5 edge: file://, javascript:, shell metachar URLs rejected
- S6 happy: build_install_plan returns uv pip install argv for local project
- S7 edge: pyproject path with .. / escape rejected
- S8 edge: assert_no_forbidden_env raises on HF/LIUM/secret keys
- S9 happy: build_boot_env emits only non-secret PRISM_* keys
- S10 adjacent: boot env never includes secrets even via kwargs
"""

from __future__ import annotations

from pathlib import Path

import pytest

from prism_challenge.evaluator.pod_boot import (
    FORBIDDEN_ENV_KEYS,
    PodBootError,
    assert_no_forbidden_env,
    build_boot_env,
    build_install_plan,
    validate_commit_sha,
    validate_repo_url,
)

FULL_SHA = "a" * 40
SHORT_SHA = "abc1234"
HTTPS_REPO = "https://github.com/miner-org/miner-repo.git"


# --- S1 / S2: SHA format happy -------------------------------------------------


def test_validate_commit_sha_accepts_full_40_hex() -> None:
    assert validate_commit_sha(FULL_SHA) == FULL_SHA


def test_validate_commit_sha_accepts_short_7_plus_hex() -> None:
    assert validate_commit_sha(SHORT_SHA) == SHORT_SHA
    assert validate_commit_sha("deadbeef") == "deadbeef"
    assert validate_commit_sha("ABCDEF1") == "abcdef1"


# --- S3: SHA reject path traversal / metachar / short -------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "abc",  # too short
        "g" * 7,  # non-hex
        "../etc/passwd",
        "abc1234;rm -rf /",
        "abc1234 && true",
        "abc 1234",
        "abc\n1234",
        "abc`id`def",
        "$(whoami)",
        "a" * 41,  # longer than full SHA
        "a" * 39,  # 39 hex valid under 7+ rule
    ],
)
def test_validate_commit_sha_rejects_malformed_and_injection(bad: str) -> None:
    # 39-char pure hex is valid under "7+ hex" rule — skip that case
    if bad == "a" * 39:
        assert validate_commit_sha(bad) == bad
        return
    with pytest.raises(PodBootError):
        validate_commit_sha(bad)


def test_validate_commit_sha_rejects_path_traversal() -> None:
    with pytest.raises(PodBootError, match="(?i)sha|commit|invalid"):
        validate_commit_sha("../../evil")


def test_validate_commit_sha_rejects_spaces_and_shell_metachar() -> None:
    for bad in ("ab cd ef", "abc;id", "abc|id", "abc&id", "abc$id", "abc`id`"):
        with pytest.raises(PodBootError):
            validate_commit_sha(bad)


# --- S4 / S5: repo URL --------------------------------------------------------


def test_validate_repo_url_accepts_https_git_shape() -> None:
    assert validate_repo_url(HTTPS_REPO) == HTTPS_REPO
    assert (
        validate_repo_url("https://gitlab.com/org/name")
        == "https://gitlab.com/org/name"
    )


@pytest.mark.parametrize(
    "bad",
    [
        "file:///etc/passwd",
        "FILE:///tmp/x",
        "javascript:alert(1)",
        "http://github.com/org/repo.git",  # http not https
        "git@github.com:org/repo.git",
        "https://github.com/org/repo.git; rm -rf /",
        "https://github.com/org/repo.git && true",
        "https://github.com/org/repo.git`id`",
        "https://github.com/org/repo.git$(id)",
        "https://github.com/org/repo with space.git",
        "",
        "ftp://example.com/repo.git",
        "https://",
    ],
)
def test_validate_repo_url_rejects_dangerous_shapes(bad: str) -> None:
    with pytest.raises(PodBootError):
        validate_repo_url(bad)


# --- S6 / S7: install plan ----------------------------------------------------


def test_build_install_plan_returns_uv_pip_install_argv(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("[project]\nname='miner'\n", encoding="utf-8")
    plan = build_install_plan(pyproject)
    assert plan[0] == "uv"
    assert "pip" in plan
    assert "install" in plan
    assert "--no-cache" in plan
    # install target is the project directory (parent of pyproject)
    assert str(tmp_path) in plan or str(tmp_path.resolve()) in plan


def test_build_install_plan_rejects_path_traversal(tmp_path: Path) -> None:
    # Construct a path that tries to escape via .. components in the given path
    evil = tmp_path / "proj" / ".." / ".." / "etc" / "passwd" / "pyproject.toml"
    with pytest.raises(PodBootError):
        build_install_plan(evil)


def test_build_install_plan_rejects_non_pyproject_name(tmp_path: Path) -> None:
    other = tmp_path / "setup.cfg"
    other.write_text("x=1\n", encoding="utf-8")
    with pytest.raises(PodBootError):
        build_install_plan(other)


# --- S8: forbidden env --------------------------------------------------------


def test_forbidden_env_keys_is_frozenset_with_core_secrets() -> None:
    assert isinstance(FORBIDDEN_ENV_KEYS, frozenset)
    for key in (
        "HF_TOKEN",
        "PRISM_HF_TOKEN",
        "LIUM_API_KEY",
        "LIUM_API_KEY_FILE",
    ):
        assert key in FORBIDDEN_ENV_KEYS


def test_assert_no_forbidden_env_passes_clean_env() -> None:
    assert_no_forbidden_env(
        {
            "PRISM_REPO_URL": HTTPS_REPO,
            "PRISM_COMMIT_SHA": FULL_SHA,
            "PATH": "/usr/bin",
        }
    )


@pytest.mark.parametrize(
    "key",
    [
        "HF_TOKEN",
        "PRISM_HF_TOKEN",
        "LIUM_API_KEY",
        "LIUM_API_KEY_FILE",
        "HUGGING_FACE_HUB_TOKEN",
        "AWS_SECRET_ACCESS_KEY",
        "OPENAI_API_KEY",
    ],
)
def test_assert_no_forbidden_env_raises_on_secret_keys(key: str) -> None:
    with pytest.raises(PodBootError, match="(?i)forbidden|secret"):
        assert_no_forbidden_env({key: "should-never-be-on-pod"})


def test_assert_no_forbidden_env_raises_on_obvious_secret_suffix() -> None:
    with pytest.raises(PodBootError):
        assert_no_forbidden_env({"MY_SERVICE_PASSWORD": "x"})
    with pytest.raises(PodBootError):
        assert_no_forbidden_env({"VENDOR_SECRET": "x"})


# --- S9 / S10: boot env builder -----------------------------------------------


def test_build_boot_env_emits_only_non_secret_prism_keys() -> None:
    env = build_boot_env(
        repo_url=HTTPS_REPO,
        commit_sha=FULL_SHA,
        master_checkpoint_url="https://chain.joinbase.ai/internal/v1/checkpoints",
        submission_id="sub-123",
        attempt=1,
    )
    assert env["PRISM_REPO_URL"] == HTTPS_REPO
    assert env["PRISM_COMMIT_SHA"] == FULL_SHA
    assert (
        env["PRISM_MASTER_CHECKPOINT_URL"]
        == "https://chain.joinbase.ai/internal/v1/checkpoints"
    )
    assert env["PRISM_SUBMISSION_ID"] == "sub-123"
    assert env["PRISM_ATTEMPT"] == "1"
    # No secrets
    for secret in FORBIDDEN_ENV_KEYS:
        assert secret not in env
    assert_no_forbidden_env(env)


def test_build_boot_env_never_includes_secrets_via_kwargs() -> None:
    with pytest.raises(PodBootError):
        build_boot_env(
            repo_url=HTTPS_REPO,
            commit_sha=FULL_SHA,
            master_checkpoint_url="https://chain.joinbase.ai/internal/v1/checkpoints",
            submission_id="sub-123",
            attempt=2,
            HF_TOKEN="leak",
        )
    with pytest.raises(PodBootError):
        build_boot_env(
            repo_url=HTTPS_REPO,
            commit_sha=FULL_SHA,
            master_checkpoint_url="https://chain.joinbase.ai/internal/v1/checkpoints",
            submission_id="sub-123",
            attempt=2,
            LIUM_API_KEY="leak",
        )


def test_build_boot_env_accepts_extra_non_secret_kwargs() -> None:
    env = build_boot_env(
        repo_url=HTTPS_REPO,
        commit_sha=SHORT_SHA,
        master_checkpoint_url="https://master.example/checkpoints",
        submission_id="sub-9",
        attempt=3,
        PRISM_WORK_DIR="/workspace/miner",
    )
    assert env["PRISM_WORK_DIR"] == "/workspace/miner"
    assert env["PRISM_COMMIT_SHA"] == SHORT_SHA
    assert "HF_TOKEN" not in env
    assert_no_forbidden_env(env)


def test_build_boot_env_validates_inputs() -> None:
    with pytest.raises(PodBootError):
        build_boot_env(
            repo_url="file:///tmp/x",
            commit_sha=FULL_SHA,
            master_checkpoint_url="https://master.example/checkpoints",
            submission_id="sub",
            attempt=1,
        )
    with pytest.raises(PodBootError):
        build_boot_env(
            repo_url=HTTPS_REPO,
            commit_sha="../evil",
            master_checkpoint_url="https://master.example/checkpoints",
            submission_id="sub",
            attempt=1,
        )


def test_build_boot_env_rejects_non_positive_attempt() -> None:
    with pytest.raises(PodBootError):
        build_boot_env(
            repo_url=HTTPS_REPO,
            commit_sha=FULL_SHA,
            master_checkpoint_url="https://master.example/checkpoints",
            submission_id="sub",
            attempt=0,
        )


def test_module_is_offline_importable() -> None:
    """Import must not touch the network (no side effects at import time)."""
    import prism_challenge.evaluator.pod_boot as mod

    assert hasattr(mod, "validate_commit_sha")
    assert hasattr(mod, "FORBIDDEN_ENV_KEYS")
