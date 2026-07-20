"""VAL-DEPLOY-001: the miner self-deploy CLI surface exists and self-documents.

The top-level CLI and every subcommand expose ``--help``; the CLI's subcommands
cover the full documented flow (fetch/prepare, publish/reproduce measurements,
deploy, run/eval, show attested result, teardown) and match the miner docs, with
no undocumented spend-capable subcommand.
"""

from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path

import pytest

from agent_challenge.selfdeploy import cli

DOC = Path(__file__).resolve().parents[1] / "docs" / "miner" / "self-deploy.md"

# The full-flow operations the CLI must cover (architecture §4 C7 / VAL-DEPLOY-001).
REQUIRED_SUBCOMMANDS = {
    "prepare",  # fetch/prepare canonical image + generated compose
    "measurements",  # publish/reproduce measurements
    "deploy",  # deploy the CVM
    "run",  # run/eval
    "result",  # show attested result
    "teardown",  # teardown
}


def _documented_subcommands() -> set[str]:
    text = DOC.read_text(encoding="utf-8")
    return set(re.findall(r"^###\s+`([a-z-]+)`", text, flags=re.MULTILINE))


def _help_text(argv: list[str]) -> tuple[int, str]:
    buffer = io.StringIO()
    code = 0
    with contextlib.redirect_stdout(buffer):
        with pytest.raises(SystemExit) as exc:
            cli.main(argv)
        code = int(exc.value.code or 0)
    return code, buffer.getvalue()


def test_top_level_help_lists_every_subcommand():
    code, text = _help_text(["--help"])
    assert code == 0
    for name in cli.SUBCOMMANDS:
        assert name in text, name


def test_cli_covers_the_full_documented_flow():
    assert REQUIRED_SUBCOMMANDS.issubset(set(cli.SUBCOMMANDS))


def test_every_subcommand_self_documents_via_help():
    for name in cli.SUBCOMMANDS:
        code, text = _help_text([name, "--help"])
        assert code == 0, name
        # Help is non-trivial: shows usage for the subcommand.
        assert name in text
        assert "usage:" in text.lower()


def test_cli_subcommands_match_the_miner_docs():
    documented = _documented_subcommands()
    cli_set = set(cli.SUBCOMMANDS)
    # Every documented subcommand exists in the CLI (docs describe only real ones).
    assert documented <= cli_set, documented - cli_set
    # Every CLI subcommand is documented (no undocumented surface at all).
    assert cli_set <= documented, cli_set - documented


def test_no_undocumented_spend_capable_subcommand():
    documented = _documented_subcommands()
    # Every spend-capable subcommand must be documented.
    assert cli.SPEND_CAPABLE_SUBCOMMANDS <= documented
    # And the spend-capable set is a subset of the real CLI surface.
    assert cli.SPEND_CAPABLE_SUBCOMMANDS <= set(cli.SUBCOMMANDS)


def test_missing_subcommand_is_rejected():
    # A bare invocation (no subcommand) is a usage error, not a silent no-op.
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert int(exc.value.code or 0) != 0
