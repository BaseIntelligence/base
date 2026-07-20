"""VAL-DEPLOY-022: self-deploy is opt-in and never runs automatically.

With the Phala backend feature flag OFF (the default), no code path auto-triggers
a CVM deploy: the default validator-run eval behavior is unchanged and a legacy
run creates zero CVMs. The self-deploy CLI touches Phala ONLY on an explicit
`deploy`/`teardown` invocation -- every other subcommand (and a dry-run deploy)
makes zero CVM-creating calls.

Offline discriminators: a SpyDeployer/SpyTeardowner assert zero calls for the
non-provisioning surface, the config default is off, and the evaluation path does
not import the self-deploy deployer (so a legacy run cannot reach it).
"""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from agent_challenge.sdk.config import ChallengeSettings
from agent_challenge.selfdeploy import cli
from agent_challenge.selfdeploy.plan import PHALA_API_KEY_ENV

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EVAL_PKG = _REPO_ROOT / "src" / "agent_challenge" / "evaluation"

DIGEST = "ghcr.io/baseintelligence/agent-challenge-canonical@sha256:" + ("a" * 64)
URL = "https://validator.example/keyrelease"


class _Spy:
    def __init__(self, result=None) -> None:
        self.calls: list[tuple] = []
        self._result = result if result is not None else {"ok": True}

    def __call__(self, *args):
        self.calls.append(args)
        return self._result


def _run_cli(argv, *, deployer=None, teardowner=None):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = cli.main(argv, deployer=deployer, teardowner=teardowner)
    return code, out.getvalue(), err.getvalue()


# --------------------------------------------------------------------------- #
# Config default: the Phala path is OFF unless explicitly enabled.
# --------------------------------------------------------------------------- #
def test_config_default_has_phala_path_off(monkeypatch):
    monkeypatch.delenv("CHALLENGE_PHALA_ATTESTATION_ENABLED", raising=False)
    monkeypatch.delenv("CHALLENGE_TERMINAL_BENCH_EXECUTION_BACKEND", raising=False)
    settings = ChallengeSettings()
    assert settings.phala_attestation_enabled is False
    assert settings.terminal_bench_execution_backend == "own_runner"


# --------------------------------------------------------------------------- #
# A legacy eval run cannot reach the self-deploy deployer at all.
# --------------------------------------------------------------------------- #
def test_evaluation_path_does_not_import_self_deploy():
    """Evaluation code must not import the self-deploy deployer surface.

    Fixture strings may mention the shipping miner CLI name; only real
    ``import`` / ``from ... import`` edges against ``agent_challenge.selfdeploy``
    (or a bare ``import selfdeploy``) are residualed as auto-trigger risk.
    """

    offenders = []
    import re

    # Match active import statements only; ignore comments and string literals that
    # mention the miner self-deploy CLI as documentation/fixtures.
    import_pat = re.compile(
        r"^\s*(?:from\s+agent_challenge\.selfdeploy(?:\.[\w.]+)?\s+import\b"
        r"|import\s+agent_challenge\.selfdeploy\b"
        r"|import\s+selfdeploy\b)",
        re.MULTILINE,
    )
    for path in _EVAL_PKG.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if import_pat.search(text):
            offenders.append(path.relative_to(_REPO_ROOT))
    assert offenders == [], offenders


# --------------------------------------------------------------------------- #
# Non-provisioning subcommands make ZERO CVM-creating / teardown calls.
# --------------------------------------------------------------------------- #
def test_prepare_makes_no_phala_calls(tmp_path):
    deployer, teardowner = _Spy(), _Spy()
    code, _out, _err = _run_cli(
        ["prepare", "--image", DIGEST, "--key-release-url", URL, "--out", str(tmp_path)],
        deployer=deployer,
        teardowner=teardowner,
    )
    assert code == 0
    assert deployer.calls == []
    assert teardowner.calls == []


def test_dry_run_deploy_makes_no_phala_calls():
    deployer, teardowner = _Spy(), _Spy()
    code, _out, _err = _run_cli(
        ["deploy", "--image", DIGEST, "--key-release-url", URL, "--dry-run"],
        deployer=deployer,
        teardowner=teardowner,
    )
    assert code == 0
    assert deployer.calls == []
    assert teardowner.calls == []


# --------------------------------------------------------------------------- #
# Only an EXPLICIT deploy / teardown reaches Phala.
# --------------------------------------------------------------------------- #
def test_explicit_deploy_is_the_only_thing_that_provisions(monkeypatch):
    monkeypatch.setenv(PHALA_API_KEY_ENV, "phak_" + "s" * 32)
    deployer, teardowner = _Spy(), _Spy()
    code, _out, _err = _run_cli(
        ["deploy", "--image", DIGEST, "--key-release-url", URL],
        deployer=deployer,
        teardowner=teardowner,
    )
    assert code == 0
    assert len(deployer.calls) == 1  # exactly one deploy, on explicit invocation
    assert teardowner.calls == []  # deploy never tears down


def test_explicit_teardown_is_the_only_thing_that_deletes():
    deployer, teardowner = _Spy(), _Spy()
    code, _out, _err = _run_cli(
        ["teardown", "--cvm-id", "cvm-123"],
        deployer=deployer,
        teardowner=teardowner,
    )
    assert code == 0
    assert teardowner.calls == [("cvm-123",)]
    assert deployer.calls == []  # teardown never provisions
