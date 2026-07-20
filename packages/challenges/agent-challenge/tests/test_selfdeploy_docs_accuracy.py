"""Docs-accuracy contract for the M6 miner + validator self-deploy docs.

Covers VAL-DEPLOY-017 (every documented command/flag/endpoint exists and behaves
as documented -- no documented-but-absent feature), VAL-DEPLOY-018 (zero
"watchtower" phrasing), VAL-DEPLOY-019 (cross-repo links to the base repo are
labeled "available after PR merge"), VAL-DEPLOY-020 (mandatory teardown +
money-cap guidance with valid `phala cvms ...` commands), and VAL-DEPLOY-021 (no
leaked secrets; credentials referenced only as the `PHALA_CLOUD_API_KEY` env var,
never written to a committed file).

The M6 self-deploy docs under test are the miner CLI doc and the validator
operations doc; both are walked for documented-command accuracy.
"""

from __future__ import annotations

import importlib
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
from _routing import iter_api_routes

from agent_challenge.app import app
from agent_challenge.selfdeploy import cli

_REPO_ROOT = Path(__file__).resolve().parents[1]
MINER_DOC = _REPO_ROOT / "docs" / "miner" / "self-deploy.md"
VALIDATOR_DOC = _REPO_ROOT / "docs" / "validator" / "self-deploy.md"
M6_DOCS = (MINER_DOC, VALIDATOR_DOC)

SELF_DEPLOY_PREFIX = "python -m agent_challenge.selfdeploy"


def _fenced_blocks(text: str) -> list[str]:
    return re.findall(r"```(?:[a-zA-Z0-9]*)\n(.*?)```", text, flags=re.DOTALL)


def _logical_commands(text: str) -> list[str]:
    """Return shell commands from fenced blocks, joining backslash continuations."""

    commands: list[str] = []
    for block in _fenced_blocks(text):
        joined = block.replace("\\\n", " ")
        for line in joined.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                commands.append(re.sub(r"\s+", " ", stripped))
    return commands


def _self_deploy_commands(text: str) -> list[list[str]]:
    out: list[list[str]] = []
    prefix_len = len(shlex.split(SELF_DEPLOY_PREFIX))
    for command in _logical_commands(text):
        if not command.startswith(SELF_DEPLOY_PREFIX):
            continue
        rest = shlex.split(command)[prefix_len:]
        # Skip meta/placeholder invocations (usage lines, `--help`).
        if not rest or rest[0].startswith(("<", "-")):
            continue
        out.append(rest)
    return out


# --------------------------------------------------------------------------- #
# The docs exist.
# --------------------------------------------------------------------------- #
def test_m6_docs_exist():
    for doc in M6_DOCS:
        assert doc.is_file(), doc


# --------------------------------------------------------------------------- #
# VAL-DEPLOY-017: every documented self-deploy command/flag is real.
# --------------------------------------------------------------------------- #
def test_documented_self_deploy_commands_parse_against_the_real_cli():
    parser = cli.build_parser()
    seen_subcommands: set[str] = set()
    total = 0
    for doc in M6_DOCS:
        for argv in _self_deploy_commands(doc.read_text(encoding="utf-8")):
            total += 1
            assert argv, "empty self-deploy invocation in the docs"
            seen_subcommands.add(argv[0])
            # parse_args validates the subcommand + every flag; an unknown flag
            # or subcommand raises SystemExit (documented-but-absent feature).
            namespace = parser.parse_args(argv)
            assert namespace.command == argv[0]
    assert total > 0, "no documented self-deploy commands found"
    # Every documented subcommand is a real CLI subcommand.
    assert seen_subcommands <= (set(cli.SUBCOMMANDS) | set(cli.ORDERED_SUBCOMMANDS)), (
        seen_subcommands - (set(cli.SUBCOMMANDS) | set(cli.ORDERED_SUBCOMMANDS))
    )


def test_ordered_review_and_eval_stages_are_documented_and_parseable():
    parser = cli.build_parser()
    expected = {
        ("review", "prepare"),
        ("review", "deploy"),
        ("review", "deployed"),
        ("review", "result"),
        ("review", "history"),
        ("review", "cancel"),
        ("review", "retry"),
        ("review", "teardown"),
        ("eval", "prepare"),
        ("eval", "deploy"),
        ("eval", "result"),
        ("eval", "status"),
        ("eval", "cancel"),
        ("eval", "retry"),
        ("eval", "failure"),
        ("eval", "teardown"),
    }
    observed: set[tuple[str, str]] = set()
    for doc in M6_DOCS:
        text = doc.read_text(encoding="utf-8")
        for argv in _self_deploy_commands(text):
            if len(argv) >= 2 and argv[0] in cli.ORDERED_SUBCOMMANDS:
                parser.parse_args(argv)
                observed.add((argv[0], argv[1]))
    assert observed == expected


def test_documented_module_entrypoints_exist():
    # The validator doc references module entrypoints; each must be importable
    # and expose a main() (a documented-but-absent module would fail here).
    referenced = set()
    for doc in M6_DOCS:
        text = doc.read_text(encoding="utf-8")
        referenced.update(re.findall(r"python -m ([a-zA-Z0-9_.]+)", text))
    # Drop the selfdeploy package invocation (its subcommands are validated above).
    module_targets = {m for m in referenced if m != "agent_challenge.selfdeploy"}
    for module_name in module_targets:
        module = importlib.import_module(module_name)
        assert hasattr(module, "main"), module_name


# --------------------------------------------------------------------------- #
# VAL-DEPLOY-018: no "watchtower" phrasing anywhere in the M6 docs.
# --------------------------------------------------------------------------- #
def test_no_watchtower_phrasing():
    for doc in M6_DOCS:
        assert "watchtower" not in doc.read_text(encoding="utf-8").lower(), doc


# --------------------------------------------------------------------------- #
# VAL-DEPLOY-019: cross-repo (base) links carry an "available after PR merge" tag.
# --------------------------------------------------------------------------- #
def test_cross_repo_base_links_are_labeled_available_after_pr_merge():
    # Any reference to the base validator/master repo (NOT the published
    # `baseagent` template) must be labeled as not-yet-merged.
    base_ref = re.compile(r"BaseIntelligence/base(?![a-zA-Z])")
    label = "available after pr merge"
    found_any = False
    for doc in M6_DOCS:
        for raw_line in doc.read_text(encoding="utf-8").splitlines():
            if base_ref.search(raw_line):
                found_any = True
                assert label in raw_line.lower(), raw_line
    assert found_any, "expected at least one labeled cross-repo base reference in the M6 docs"


# --------------------------------------------------------------------------- #
# VAL-DEPLOY-020: teardown + money-cap guidance with valid phala commands.
# --------------------------------------------------------------------------- #
def test_teardown_and_cap_guidance_present_with_valid_commands():
    miner = MINER_DOC.read_text(encoding="utf-8")
    validator = VALIDATOR_DOC.read_text(encoding="utf-8")
    for doc_text in (miner, validator):
        # Mandatory teardown commands are shown verbatim.
        assert "phala cvms list" in doc_text
        assert "phala cvms delete <id> -f" in doc_text
        # Money cap is stated.
        assert "$20" in doc_text
        # total: 0 teardown-confirmation guidance is present.
        assert "total: 0" in doc_text


def test_documented_teardown_command_matches_the_cli_implementation():
    # The documented teardown command form must match what the CLI actually runs.
    import inspect

    source = inspect.getsource(cli.default_phala_teardown)
    assert '"phala", "cvms", "delete"' in source
    assert '"-f"' in source


def test_documented_phala_commands_are_valid_when_cli_available():
    if shutil.which("phala") is None:
        pytest.skip("phala CLI not installed")
    help_text = subprocess.run(
        ["phala", "cvms", "--help"], capture_output=True, text=True, timeout=30
    ).stdout.lower()
    assert "delete" in help_text
    assert "list" in help_text
    delete_help = subprocess.run(
        ["phala", "cvms", "delete", "--help"], capture_output=True, text=True, timeout=30
    ).stdout.lower()
    assert "-f" in delete_help or "--force" in delete_help


# --------------------------------------------------------------------------- #
# VAL-DEPLOY-021: no leaked secrets; env-var-only credential handling.
# --------------------------------------------------------------------------- #
def test_docs_leak_no_secrets():
    secret_shapes = [
        re.compile(r"phak_[A-Za-z0-9]{16,}"),  # a real Phala key
        re.compile(r"\bsk-[A-Za-z0-9]{16,}"),  # a provider secret key
    ]
    for doc in M6_DOCS:
        text = doc.read_text(encoding="utf-8")
        for pattern in secret_shapes:
            assert pattern.search(text) is None, (doc, pattern.pattern)


def test_credentials_referenced_as_env_var_only():
    # The credential is referenced as the env-var name, never assigned a literal
    # secret value in the docs (env-var-only handling).
    assignment = re.compile(r"PHALA_CLOUD_API_KEY\s*=\s*(\S+)")
    referenced = False
    for doc in M6_DOCS:
        text = doc.read_text(encoding="utf-8")
        if "PHALA_CLOUD_API_KEY" in text:
            referenced = True
        for match in assignment.finditer(text):
            value = match.group(1).strip("\"'`")
            # An assignment, if shown at all, must not carry a real key value.
            assert not value.startswith(("phak_", "sk-")), value
    assert referenced, "the Phala credential env var must be referenced in the docs"


def test_ordered_routes_match_the_implemented_route_table_and_no_aliases_are_claimed():
    docs = "\n".join(doc.read_text(encoding="utf-8") for doc in M6_DOCS)
    expected = {
        ("POST", "/submissions/{submission_id}/review/prepare"),
        ("POST", "/submissions/{submission_id}/review/retry"),
        ("POST", "/submissions/{submission_id}/review/cancel"),
        ("POST", "/submissions/{submission_id}/review/deployed"),
        ("GET", "/submissions/{submission_id}/review/history"),
        ("GET", "/submissions/{submission_id}/review/report"),
        ("POST", "/submissions/{submission_id}/eval/prepare"),
        ("POST", "/submissions/{submission_id}/eval/retry"),
        ("POST", "/submissions/{submission_id}/eval/cancel"),
        ("POST", "/submissions/{submission_id}/eval/failure"),
        ("GET", "/submissions/{submission_id}/eval/status"),
        ("POST", "/evaluation/v1/runs/{eval_run_id}/result"),
    }
    available = {
        (method, route.path)
        for route in iter_api_routes(app)
        for method in (route.methods or set())
        if method not in {"HEAD", "OPTIONS"}
    }
    for method, path in expected:
        assert f"{method} {path}" in docs
        assert (method, path) in available
    for forbidden in (
        "/internal/v1/reviews/{session_id}/approvals",
        "/internal/v1/reviews/{session_id}/evidence/{object_ref}",
        "/review/v1/assignments/{assignment_id}/report",
        "/evaluation/v1/runs/{eval_run_id}/result",
    ):
        if forbidden == "/evaluation/v1/runs/{eval_run_id}/result":
            continue
        assert f"BASE public alias: {forbidden}" not in docs
    assert "/internal/v1/" in docs
    assert "BASE-blocked" in docs


def test_docs_pin_safe_ordered_observability_and_rejection_contract():
    docs = "\n".join(doc.read_text(encoding="utf-8") for doc in M6_DOCS)
    required = (
        "review_queued",
        "review_cvm_running",
        "review_provider_standby",
        "review_verifying",
        "review_allowed",
        "review_rejected",
        "review_escalated",
        "review_expired",
        "review_cancelled",
        "review_error",
        "eval_prepared",
        "eval_running",
        "eval_verifying",
        "eval_expired",
        "eval_cancelled",
        "eval_error",
        "eval_rejected",
        "eval_accepted",
        "attempt",
        "retryable",
        "reason_code",
        "report_available",
        "key_grant_state",
        "key_release_nonce_state",
        "score_nonce_state",
        "receipt_id",
        "body_sha256",
        "result_available",
        "next_cursor",
        "total_count",
        "review_allow_required",
        "eval_deploy_failed",
        "eval_tunnel_failed",
        "eval_key_release_unavailable",
        "eval_no_result",
        "no score",
        "no benchmark work",
    )
    for term in required:
        assert term in docs, term
    forbidden_claims = (
        "exposes unrestricted source",
        "returns raw OpenRouter response",
        "sends plaintext OPENROUTER_API_KEY",
        "returns the OPENROUTER_API_KEY value",
        "returns a raw quote",
        "exposes the full task list before allow",
    )
    for term in forbidden_claims:
        assert term.lower() not in docs.lower(), term
    assert "only digests" in docs.lower()
    assert "redacted" in docs.lower()


def test_docs_pin_encrypted_env_boundaries_and_raw_tcp_ratls_contract():
    miner = MINER_DOC.read_text(encoding="utf-8")
    validator = VALIDATOR_DOC.read_text(encoding="utf-8")
    docs = miner + "\n" + validator
    for term in (
        "OPENROUTER_API_KEY",
        "REVIEW_SESSION_TOKEN",
        "EVAL_RUN_TOKEN",
        # Named as forbidden residual (not required eval secrets).
        "BASE_GATEWAY_TOKEN",
        "BASE_LLM_GATEWAY_URL",
        "encrypted_env",
        "env_keys",
        "POST /cvms/provision",
        "POST /cvms",
        "KEY_RELEASE_RA_TLS_HOST",
        "KEY_RELEASE_RA_TLS_PORT",
        "KEY_RELEASE_RA_TLS_CERT_FILE",
        "KEY_RELEASE_RA_TLS_KEY_FILE",
        "KEY_RELEASE_RA_TLS_CA_FILE",
        "8701",
        "TLS 1.3",
        "4-byte big-endian",
        "schema_version",
        "quote_hex",
        "event_log",
        "no HTTP status framing",
    ):
        assert term in docs, term
    # Must not still document gateway flags as required eval deploy options.
    assert "--gateway-token-env BASE_GATEWAY_TOKEN" not in docs
    assert "--gateway-url-env BASE_LLM_GATEWAY_URL" not in docs
    assert "OPENROUTER_API_KEY" in miner
    assert "REVIEW_SESSION_TOKEN" in miner
    assert "EVAL_RUN_TOKEN" in miner
    assert "OPENROUTER_API_KEY" in validator
    assert "eval CVM" in validator
    assert "encrypted_env" in validator


def test_docs_secret_scan_rejects_values_and_unrestricted_evidence():
    secret_shapes = (
        re.compile(r"(?:PHALA_CLOUD_API_KEY|OPENROUTER_API_KEY)\s*=\s*(?!<)[^`\s]+"),
        re.compile(r"\b(?:phak_|sk-)[A-Za-z0-9_-]{16,}"),
        re.compile(r"Bearer\s+(?!<)[A-Za-z0-9._~+/=-]{20,}"),
    )
    for doc in M6_DOCS:
        text = doc.read_text(encoding="utf-8")
        for pattern in secret_shapes:
            assert pattern.search(text) is None, (doc, pattern.pattern)
        assert "never" in text.lower()
        assert "secret" in text.lower()


# --------------------------------------------------------------------------- #
# VAL-DEPLOY-017 residual: approval retry, explicit signing, cursor signing,
# RA-TLS setup, teardown, and failure behavior match the hardened CLI.
# These modules are discriminators: a docs revision that omitted any of the
# post-harden flags or behavior would fail even if CLI-parse tests still pass.
# --------------------------------------------------------------------------- #
def _first_matching_selfdeploy(doc_text: str, *must_contain: str) -> list[str]:
    """Return argv for the first documented command containing every token."""

    parser = cli.build_parser()
    for argv in _self_deploy_commands(doc_text):
        cont = set(argv)
        if all(token in cont for token in must_contain):
            namespace = parser.parse_args(argv)
            assert namespace.command == argv[0]
            return argv
    raise AssertionError(f"no documented command containing {must_contain!r}")


def test_docs_include_approval_backed_retry_example_with_required_args():
    """Approval-backed review retry docs must show --approval-id and full identity."""

    miner = MINER_DOC.read_text(encoding="utf-8")
    assert "--approval-id" in miner
    assert "approval_id" in miner
    # The documented approval-backed form must be a complete, parseable example.
    argv = _first_matching_selfdeploy(
        miner,
        "review",
        "retry",
        "--approval-id",
        "--assignment-id",
        "--base-url",
        "--submission-id",
        "--hotkey",
    )
    # Signing path is either --auto-sign or the explicit header pair.
    has_auto = "--auto-sign" in argv
    has_explicit = "--signature" in argv and "--nonce" in argv
    assert has_auto or has_explicit, argv
    # surroundings also describe ordinary retries (omit approval_id) so readers
    # do not believe the flag is always required.
    assert "omit" in miner.lower() or "optional" in miner.lower()
    assert "reject" in miner.lower() and "escalate" in miner.lower()


def test_docs_include_explicit_signature_examples_for_deploy_and_signed_routes():
    """Non-auto-sign deploy / prepare examples must include every required arg."""

    miner = MINER_DOC.read_text(encoding="utf-8")
    parser = cli.build_parser()

    # review prepare already uses explicit signature headers; they must remain
    # complete (signature + nonce; timestamp optional but documented).
    prepare_argv = _first_matching_selfdeploy(
        miner,
        "review",
        "prepare",
        "--signature",
        "--nonce",
        "--timestamp",
        "--hotkey",
        "--base-url",
        "--submission-id",
    )
    assert prepare_argv  # for type checkers / for clarity

    # At least one live review deploy example must use the explicit-header path
    # (no --auto-sign) so the docs no longer look like auto-sign is forced.
    explicit_deploys = [
        argv
        for argv in _self_deploy_commands(miner)
        if argv[:2] == ["review", "deploy"]
        and "--signature" in argv
        and "--nonce" in argv
        and "--auto-sign" not in argv
    ]
    assert explicit_deploys, "docs must show non-auto-sign review deploy with explicit headers"
    for argv in explicit_deploys:
        ns = parser.parse_args(argv)
        assert ns.command == "review"
        assert ns.review_command == "deploy"
        assert ns.signature is not None and ns.nonce is not None
        assert not ns.auto_sign

    # Wording must document that either path is accepted.
    assert "--signature" in miner and "--nonce" in miner
    assert "auto-sign" in miner.lower()
    assert "explicit" in miner.lower()


def test_docs_document_cursor_query_signing_behavior():
    """Cursor-bearing commands must show --cursor and note exact query signing."""

    miner = MINER_DOC.read_text(encoding="utf-8")
    parser = cli.build_parser()

    cursor_commands = {
        ("review", "history"): False,
        ("review", "result"): False,
        ("eval", "status"): False,
    }
    for argv in _self_deploy_commands(miner):
        key = (argv[0], argv[1]) if len(argv) >= 2 else None
        if key in cursor_commands and "--cursor" in argv:
            parser.parse_args(argv)
            cursor_commands[key] = True
    missing = [k for k, ok in cursor_commands.items() if not ok]
    assert not missing, f"cursor examples missing for {missing}"

    # Exact query-string signing (history/report/status) is the hardened rule.
    lowered = miner.lower()
    assert "query string" in lowered or "exact query" in lowered
    assert "cursor" in lowered
    # Server seal wording must not claim body-only (pre-hardening) signing.
    assert "canonical_request_string" in miner or "canonical request" in lowered


def test_docs_describe_ratls_setup_failure_and_teardown_behavior():
    """Docs must pin RA-TLS client envs, post-create teardown, and failure exits."""

    miner = MINER_DOC.read_text(encoding="utf-8")
    validator = VALIDATOR_DOC.read_text(encoding="utf-8")
    docs = miner + "\n" + validator
    lowered = docs.lower()

    # Client-side mTLS credential paths used by the measured eval compose.
    for env_name in (
        "CHALLENGE_PHALA_RA_TLS_CERT_FILE",
        "CHALLENGE_PHALA_RA_TLS_KEY_FILE",
        "CHALLENGE_PHALA_RA_TLS_CA_FILE",
        "KEY_RELEASE_RA_TLS_HOST",
        "KEY_RELEASE_RA_TLS_PORT",
    ):
        assert env_name in docs, env_name

    # Production does not fall back to HTTP /release for the live path.
    assert "no http status framing" in lowered
    assert "http `/release` is not the production transport" in lowered or (
        "not the production transport" in lowered and "/release" in docs
    )

    # Post-create failure deletes the attributable CVM; teardown is fail-closed.
    assert "attributable" in lowered
    assert "post-create" in lowered or "after create" in lowered
    assert "non-zero" in lowered or "nonzero" in lowered or "non zero" in lowered
    assert "bounded" in lowered and "diagnostic" in lowered

    # Acceptance remains a conjunction (no "quoted-only" claim).
    for term in ("binding", "quote", "measurement", "nonce", "key-grant"):
        assert term in lowered, term

    # Dry-run never fabricates IN-LIST membership.
    assert "unknown" in lowered
    assert "in-list" in lowered or "allowlist" in lowered

    # Documented teardown commands continue to parse and use delete -f.
    tear_argv = _first_matching_selfdeploy(miner, "teardown", "--cvm-id")
    assert tear_argv[0] in {"teardown", "review", "eval"}
    assert "phala cvms delete <id> -f" in miner
    assert "phala cvms delete <id> -f" in validator
