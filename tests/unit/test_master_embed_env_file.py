"""Embedded challenge env-file contract for the master supervisor.

``docker/master-entrypoint.sh`` launches each embedded challenge under
``env -i`` so Prism never inherits ``CHALLENGE_*`` and agent-challenge never
inherits ``PRISM_*``. That isolation is deliberate, but it also dropped every
operator-supplied setting because the allowlist was hardcoded with no
extension point.

These tests lock the supported extension point: a per-challenge env file whose
keys are merged into the isolated child environment, without breaking
cross-challenge isolation.

T40/T41: product path is host-trust / unattested only. Dual Phala flags must
not be true in templates; tests use host-trust keys (NO_PHALA / UNATTESTED /
DOCKER_BACKEND) to prove the merge mechanism.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = REPO_ROOT / "docker/master-entrypoint.sh"

FAKE_UVICORN = """#!/usr/bin/env bash
# Dump the isolated child environment so the test can assert on it.
# The dump path is baked in: `env -i` strips DUMP_DIR from the child env.
target="unknown"
for arg in "$@"; do
  case "${arg}" in
    prism_challenge.app:app) target="prism" ;;
    agent_challenge.app:app) target="ac" ;;
  esac
done
env > "__DUMP_DIR__/${target}.env"
"""

FAKE_PYTHON = """#!/usr/bin/env bash
exit 0
"""


def _write_exec(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _run_entrypoint(
    tmp_path: Path,
    ac_env_file_body: str | None,
    prism_env_file_body: str | None = None,
) -> dict[str, str]:
    """Run the entrypoint with stubbed uvicorn/python; return child env dumps."""

    bin_dir = tmp_path / "bin"
    dump_dir = tmp_path / "dump"
    ac_dir = tmp_path / "ac"
    prism_dir = tmp_path / "prism"
    for directory in (bin_dir, dump_dir, ac_dir, prism_dir):
        directory.mkdir(parents=True, exist_ok=True)

    _write_exec(
        bin_dir / "uvicorn", FAKE_UVICORN.replace("__DUMP_DIR__", str(dump_dir))
    )
    _write_exec(bin_dir / "python", FAKE_PYTHON)

    token_file = tmp_path / "shared_token"
    token_file.write_text("test-token", encoding="utf-8")

    if ac_env_file_body is not None:
        (ac_dir / "embed.env").write_text(ac_env_file_body, encoding="utf-8")
    if prism_env_file_body is not None:
        (prism_dir / "embed.env").write_text(prism_env_file_body, encoding="utf-8")

    env = {
        "PATH": f"{bin_dir}:{os.environ.get('PATH', '/usr/bin:/bin')}",
        "HOME": str(tmp_path),
        "DUMP_DIR": str(dump_dir),
        "BASE_MASTER_AC_DATA_DIR": str(ac_dir),
        "BASE_MASTER_PRISM_DATA_DIR": str(prism_dir),
        "PRISM_SHARED_TOKEN_FILE": str(token_file),
        "CHALLENGE_SHARED_TOKEN_FILE": str(token_file),
    }

    subprocess.run(
        ["bash", str(ENTRYPOINT), "/bin/true"],
        env=env,
        check=True,
        capture_output=True,
        timeout=60,
    )

    dumps: dict[str, str] = {}
    for name in ("ac", "prism"):
        dump = dump_dir / f"{name}.env"
        dumps[name] = dump.read_text(encoding="utf-8") if dump.is_file() else ""
    return dumps


def test_ac_env_file_supplies_host_trust_settings(tmp_path: Path) -> None:
    """Operator embed.env must reach the agent-challenge child under env -i."""

    dumps = _run_entrypoint(
        tmp_path,
        "\n".join(
            [
                "# durable AC embed overrides (host-trust / T40)",
                "CHALLENGE_NO_PHALA=true",
                "CHALLENGE_UNATTESTED_EXECUTION=true",
                "CHALLENGE_DOCKER_BACKEND=broker",
                "BASE_CHALLENGE_SLUG=agent-challenge",
                "",
            ]
        ),
    )

    ac_env = dumps["ac"]
    assert "CHALLENGE_NO_PHALA=true" in ac_env
    assert "CHALLENGE_UNATTESTED_EXECUTION=true" in ac_env
    assert "CHALLENGE_DOCKER_BACKEND=broker" in ac_env
    assert "BASE_CHALLENGE_SLUG=agent-challenge" in ac_env


def test_ac_env_file_does_not_leak_into_prism_child(tmp_path: Path) -> None:
    """Cross-challenge isolation must survive the env-file merge."""

    dumps = _run_entrypoint(
        tmp_path,
        "CHALLENGE_NO_PHALA=true\nCHALLENGE_DOCKER_BACKEND=broker\n",
    )

    prism_env = dumps["prism"]
    assert "CHALLENGE_NO_PHALA" not in prism_env
    assert "CHALLENGE_DOCKER_BACKEND=broker" not in prism_env


def test_ac_env_file_overrides_builtin_default(tmp_path: Path) -> None:
    """File-provided values win over the hardcoded defaults."""

    dumps = _run_entrypoint(tmp_path, "CHALLENGE_DOCKER_ENABLED=true\n")

    assert "CHALLENGE_DOCKER_ENABLED=true" in dumps["ac"]
    assert "CHALLENGE_DOCKER_ENABLED=false" not in dumps["ac"]


def test_missing_env_file_is_not_fatal(tmp_path: Path) -> None:
    """Absent embed.env keeps the previous behaviour (defaults only)."""

    dumps = _run_entrypoint(tmp_path, None)

    assert "CHALLENGE_DOCKER_ENABLED=false" in dumps["ac"]
    assert "CHALLENGE_NO_PHALA" not in dumps["ac"]


def test_env_file_ignores_comments_blanks_and_malformed_keys(tmp_path: Path) -> None:
    """Only well-formed KEY=VALUE lines with allowed prefixes are exported."""

    dumps = _run_entrypoint(
        tmp_path,
        "\n".join(
            [
                "# comment",
                "",
                "   ",
                "export CHALLENGE_EVAL_MAX_ATTEMPTS=3",
                "not-a-valid-key=nope",
                "RANDOM_UNRELATED=leak",
                "CHALLENGE_EVAL_APP_IDENTITY=app-id",
                "",
            ]
        ),
    )

    ac_env = dumps["ac"]
    assert "CHALLENGE_EVAL_MAX_ATTEMPTS=3" in ac_env
    assert "CHALLENGE_EVAL_APP_IDENTITY=app-id" in ac_env
    assert "RANDOM_UNRELATED" not in ac_env
    assert "not-a-valid-key" not in ac_env


def test_prism_env_file_forwards_unattested_flags(tmp_path: Path) -> None:
    """Prism embed.env must forward unattested / NO_PHALA flags under env -i."""

    dumps = _run_entrypoint(
        tmp_path,
        ac_env_file_body=None,
        prism_env_file_body="\n".join(
            [
                "# prism host-trust / unattested (T19-T22)",
                "CHALLENGE_UNATTESTED_EXECUTION=true",
                "CHALLENGE_NO_PHALA=true",
                "NO_PHALA=true",
                "PRISM_RAW_WEIGHT_PUSH_ENABLED=true",
                "RANDOM_LEAK=nope",
                "",
            ]
        ),
    )

    prism_env = dumps["prism"]
    assert "CHALLENGE_UNATTESTED_EXECUTION=true" in prism_env
    assert "CHALLENGE_NO_PHALA=true" in prism_env
    assert "NO_PHALA=true" in prism_env
    assert "PRISM_RAW_WEIGHT_PUSH_ENABLED=true" in prism_env
    assert "RANDOM_LEAK" not in prism_env


def test_prism_admission_requires_worker_defaults_false(tmp_path: Path) -> None:
    """Built-in default for admission_requires_worker is false (unattested path)."""

    dumps = _run_entrypoint(tmp_path, ac_env_file_body=None)
    assert (
        "PRISM_WORKER_PLANE__ADMISSION_REQUIRES_WORKER=false" in dumps["prism"]
    )


def test_prism_unattested_flags_do_not_leak_into_ac(tmp_path: Path) -> None:
    """Prism-only unattested keys must not appear in the AC child env."""

    dumps = _run_entrypoint(
        tmp_path,
        ac_env_file_body=None,
        prism_env_file_body=(
            "CHALLENGE_UNATTESTED_EXECUTION=true\n"
            "CHALLENGE_NO_PHALA=true\n"
            "NO_PHALA=true\n"
        ),
    )
    ac_env = dumps["ac"]
    assert "CHALLENGE_UNATTESTED_EXECUTION" not in ac_env
    # AC may still get CHALLENGE_* from its own defaults/file — but not from prism file.
    # NO_PHALA from prism file must not leak:
    assert "NO_PHALA=true" not in ac_env

