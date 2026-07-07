"""Phase H turnkey installer — validator wallet + mock-metagraph auto-seed.

Under ``--auto-secrets`` the installer idempotently generates the validator
wallet+hotkey inside the (public) master image (``ensure_validator_wallet``, STEP
5c) and, when the operator left ``MOCK_METAGRAPH`` at its default ``[]``, seeds the
derived hotkey into ``network.mock_metagraph`` with ``validator_permit=true``
(``_auto_seed_mock_metagraph``, STEP 5d) BEFORE ``deploy_master`` renders the
master config. An operator-supplied ``MOCK_METAGRAPH`` is rendered verbatim (no
auto-seed).

Behavioral dry-run tests: run the real installer with a stub ``docker`` on PATH,
asserting the planned argv / rendered config (dry-run prints the master.yaml),
never the source text. The ss58 is a ``<derived-at-apply>`` placeholder in dry-run
(the real value needs the image at ``--apply``).
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SWARM_INSTALLER = ROOT / "deploy" / "swarm" / "install-swarm.sh"

IMAGE_MASTER_MARKER = "base-master"
DEPLOY_MASTER_STEP = "STEP 9/12 deploy_master"


def _docker_stub(bin_dir: Path) -> None:
    """`docker` stub: recent engine, INACTIVE swarm, every `inspect` MISSES."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    stub = bin_dir / "docker"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'case "$1" in\n'
        "  version) echo '29.1.3' ;;\n"
        "  info) echo 'inactive' ;;\n"
        "  *) exit 1 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _run(
    tmp_path: Path,
    *,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    _docker_stub(bin_dir)
    env = dict(os.environ)
    # --auto-secrets generates every secret; YUNWU stays external (set it so the
    # real-mode default completes cleanly).
    env["YUNWU_API_KEY"] = "external-key"
    if extra_env:
        env.update(extra_env)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["BROKER_WORKSPACE_DIR"] = str(tmp_path / "broker-ws")
    env["MASTER_CONFIG_PATH"] = str(tmp_path / "master.yaml")
    return subprocess.run(
        [
            "bash",
            str(SWARM_INSTALLER),
            "--backup-dir",
            str(tmp_path / "missing"),
            "--greenfield",
            "--static-challenges",
            "--skip-ghcr-login",
            "--auto-secrets",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_installer_is_bash_n_clean() -> None:
    proc = subprocess.run(
        ["bash", "-n", str(SWARM_INSTALLER)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"bash -n failed: {proc.stderr!r}"


def test_wallet_gen_and_chown_planned_before_deploy_master(tmp_path: Path) -> None:
    result = _run(tmp_path)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    out = result.stdout
    lines = out.splitlines()

    gen_idx = next(
        (
            i
            for i, line in enumerate(lines)
            if "docker run" in line
            and "--entrypoint python3" in line
            and IMAGE_MASTER_MARKER in line
            and "validator-1" in line
            and "/var/lib/base/wallets" in line
        ),
        None,
    )
    assert gen_idx is not None, f"no wallet-gen docker-run line; stdout={out!r}"

    chown_idx = next(
        (i for i, line in enumerate(lines) if "chown -R" in line),
        None,
    )
    assert chown_idx is not None, f"no wallet chown -R line; stdout={out!r}"

    master_idx = next(
        (i for i, line in enumerate(lines) if DEPLOY_MASTER_STEP in line),
        None,
    )
    assert master_idx is not None, "deploy_master step not reached"
    assert gen_idx < master_idx
    assert chown_idx < master_idx


def test_default_mock_metagraph_auto_seeded_with_permit(tmp_path: Path) -> None:
    # MOCK_METAGRAPH left default => the rendered master config carries the derived
    # validator hotkey with validator_permit (placeholder ss58 in dry-run).
    result = _run(tmp_path)
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    out = result.stdout

    assert "auto-seeded MOCK_METAGRAPH" in out
    seeded = [line for line in out.splitlines() if "mock_metagraph:" in line]
    assert seeded, "mock_metagraph not rendered"
    rendered = seeded[0]
    assert rendered.strip() != "mock_metagraph: []"
    assert "validator_permit" in rendered
    assert "<derived-at-apply>" in rendered


def test_operator_mock_metagraph_rendered_verbatim_no_autoseed(
    tmp_path: Path,
) -> None:
    hotkey = "5OperatorSuppliedValidatorHotkeyABCDEFGHIJKLMNO"
    operator_mmg = f'[{{"hotkey":"{hotkey}","validator_permit":true,"stake":1000}}]'
    result = _run(tmp_path, extra_env={"MOCK_METAGRAPH": operator_mmg})
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    out = result.stdout

    # Operator-supplied set is rendered verbatim; no auto-seed replaces it.
    assert "auto-seeded MOCK_METAGRAPH" not in out
    assert hotkey in out
    assert "<derived-at-apply>" not in out
