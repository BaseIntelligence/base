"""Shipping AC runtime image must bake dcap-qvl onto PATH.

Product residual after ops interim (ac-dcap-pcs-ready):
  live dual-flag AC used a host bind ``/var/lib/base/tools/dcap-qvl`` ->
  ``/usr/local/bin/dcap-qvl``. Recreates without that bind surface
  ``review_verifier_unavailable``. Runtime Dockerfile now multi-stage builds
  Phala ``dcap-qvl-cli`` and installs the binary at ``/usr/local/bin/dcap-qvl``.

No invent trust roots / offline collateral in the image. PCS egress is still
ops/network and is not asserted here.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_challenge.sdk.config import ChallengeSettings

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = REPO_ROOT / "Dockerfile"
EXPECTED_BINARY_PATH = "/usr/local/bin/dcap-qvl"
PINNED_CRATE_VERSION = "0.5.2"


def _runtime_stage_text(dockerfile: str) -> str:
    """Extract the ``runtime`` stage (before ``terminal-bench-runner``)."""

    m = re.search(
        r"FROM\s+\S+\s+AS\s+runtime\b(.*?)(?=\nFROM\s|\Z)",
        dockerfile,
        flags=re.DOTALL | re.IGNORECASE,
    )
    assert m is not None, "Dockerfile missing runtime stage"
    return m.group(1)


def test_dockerfile_has_dcap_qvl_builder_stage() -> None:
    assert DOCKERFILE.is_file()
    body = DOCKERFILE.read_text(encoding="utf-8")
    assert re.search(
        r"FROM\s+\S+\s+AS\s+dcap-qvl-builder\b",
        body,
        flags=re.IGNORECASE,
    ), "Dockerfile must include a dcap-qvl-builder multi-stage"
    assert re.search(
        rf"cargo\s+install\s+dcap-qvl-cli[^\n]*{re.escape(PINNED_CRATE_VERSION)}",
        body,
    ) or re.search(
        rf"DCAP_QVL_CLI_VERSION\s*=\s*{re.escape(PINNED_CRATE_VERSION)}",
        body,
    ), f"dcap-qvl-cli must be pinned to {PINNED_CRATE_VERSION}"
    assert "cargo install dcap-qvl-cli" in body


def test_runtime_dockerfile_copies_dcap_qvl_to_path() -> None:
    """Fail closed if the published runtime image omits dcap-qvl."""

    body = DOCKERFILE.read_text(encoding="utf-8")
    runtime = _runtime_stage_text(body)

    copy_match = re.search(
        r"^\s*COPY\s+--from=dcap-qvl-builder\s+\S+\s+/usr/local/bin/dcap-qvl\s*$",
        runtime,
        flags=re.MULTILINE,
    )
    assert copy_match is not None, (
        f"runtime Dockerfile must COPY dcap-qvl from builder stage into {EXPECTED_BINARY_PATH}"
    )
    assert re.search(
        r"chmod\s+0?755\s+/usr/local/bin/dcap-qvl",
        runtime,
    ), "dcap-qvl must be world-executable for uid 10001"
    # Ensure install happens before USER drops privileges is handled in stage
    # (COPY may be root-owned absolute path; chmod + test as root).
    user_idx = re.search(r"^\s*USER\s+10001", runtime, flags=re.MULTILINE)
    assert user_idx is not None, "runtime stage must run as challenge uid 10001"
    assert copy_match.start() < user_idx.start(), (
        "COPY dcap-qvl must precede USER 10001 so root can place /usr/local/bin"
    )


def test_require_dcap_qvl_binary_fail_closed_when_attested_and_missing() -> None:
    settings = ChallengeSettings(
        database_url="sqlite+aiosqlite:///:memory:",
        shared_token="test-shared-token-not-evidence",
        attested_review_enabled=True,
        phala_attestation_enabled=True,
        review_evidence_encryption_key="dedicated-evidence-key-material",
        eval_result_signer_mnemonic="test mnemonic phrase for offline only",
    )
    with patch("shutil.which", return_value=None):
        with pytest.raises(ValueError, match="dcap-qvl on PATH"):
            settings.require_dcap_qvl_binary_for_production()


def test_require_dcap_qvl_binary_ok_when_present() -> None:
    settings = ChallengeSettings(
        database_url="sqlite+aiosqlite:///:memory:",
        shared_token="test-shared-token-not-evidence",
        attested_review_enabled=True,
        phala_attestation_enabled=True,
        review_evidence_encryption_key="dedicated-evidence-key-material",
        eval_result_signer_mnemonic="test mnemonic phrase for offline only",
    )
    host_bin = shutil.which("dcap-qvl")
    if host_bin is None:
        # Local workers without cargo install still pass via mock; image smoke
        # is covered by Dockerfile assertions above.
        with (
            patch("shutil.which", return_value="/usr/local/bin/dcap-qvl"),
            patch("pathlib.Path.is_file", return_value=True),
            patch("os.access", return_value=True),
        ):
            settings.require_dcap_qvl_binary_for_production()
        return
    settings.require_dcap_qvl_binary_for_production()  # no raise
    assert Path(host_bin).is_file()
    assert os.access(host_bin, os.X_OK)


def test_require_dcap_qvl_binary_skipped_when_legacy_flags_off() -> None:
    legacy = ChallengeSettings(
        database_url="sqlite+aiosqlite:///:memory:",
        shared_token="test-shared-token-not-evidence",
        attested_review_enabled=False,
        phala_attestation_enabled=False,
    )
    with patch("shutil.which", return_value=None):
        legacy.require_dcap_qvl_binary_for_production()  # no raise when flags off


def test_require_dcap_qvl_error_leaks_no_secrets() -> None:
    settings = ChallengeSettings(
        database_url="sqlite+aiosqlite:///:memory:",
        shared_token="secret-token-must-not-leak",
        attested_review_enabled=True,
        phala_attestation_enabled=True,
        review_evidence_encryption_key="dedicated-evidence-key-material",
        eval_result_signer_mnemonic="test mnemonic phrase for offline only",
    )
    with patch("shutil.which", return_value=None):
        try:
            settings.require_dcap_qvl_binary_for_production()
        except ValueError as exc:
            msg = str(exc).lower()
            assert "secret-token" not in msg
            assert "dedicated-evidence" not in msg
            assert "mnemonic" not in msg
        else:  # pragma: no cover
            raise AssertionError("expected ValueError when dcap-qvl missing")
