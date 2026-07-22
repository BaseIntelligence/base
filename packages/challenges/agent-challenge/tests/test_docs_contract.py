from __future__ import annotations

import os
import re
from pathlib import Path

from _routing import iter_api_routes

from agent_challenge.app import app
from agent_challenge.sdk.config import ChallengeSettings

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
DOCS_INDEX_README = ROOT / "docs" / "README.md"
VALIDATOR_README = ROOT / "docs" / "validator" / "README.md"
MINER_README = ROOT / "docs" / "miner" / "README.md"
SUBMIT_AGENT_DOC = ROOT / "docs" / "miner" / "submit-agent.md"
FRONTEND_API_CONTRACT = ROOT / "docs" / "frontend-api-contract.md"
CONFIG_EXAMPLE = ROOT / "config.example.yaml"
PYPROJECT = ROOT / "pyproject.toml"
CI = ROOT / ".github" / "workflows" / "ci.yml"
SCRIPTS_DIR = ROOT / "scripts"


def platform_miner_readme() -> Path | None:
    for name in ("PLATFORM_CHECKOUT", "PLATFORM_ROOT"):
        value = os.environ.get(name)
        if not value:
            continue
        readme = Path(value) / "docs" / "miner" / "README.md"
        if readme.is_file():
            return readme
    return None


PLATFORM_MINER_README = platform_miner_readme()
PLATFORM_DOC_PATHS = (PLATFORM_MINER_README,) if PLATFORM_MINER_README is not None else ()

CANONICAL_STRING = (
    "{METHOD}\n{PATH_WITH_SORTED_QUERY}\n{X-TIMESTAMP}\n{X-NONCE}\n{SHA256_HEX_OF_RAW_BODY}"
)
OWNER_HOTKEY = "5GziQCcRpN8NCJktX343brnfuVe3w6gUYieeStXPD1Dag2At"
SIGNED_HEADERS = ("X-Hotkey", "X-Signature", "X-Nonce", "X-Timestamp")
DOC_PATHS = (README, VALIDATOR_README, MINER_README, FRONTEND_API_CONTRACT) + PLATFORM_DOC_PATHS
DOCUMENTED_ROUTE_CONTRACT = {
    ("GET", "/benchmarks"),
    ("GET", "/benchmarks/tasks"),
    ("POST", "/submissions"),
    ("GET", "/submissions"),
    ("GET", "/submissions/count"),
    ("GET", "/submissions/{submission_id}"),
    ("GET", "/submissions/{submission_id}/versions"),
    ("GET", "/submissions/{submission_id}/status"),
    ("GET", "/submissions/{submission_id}/events"),
    ("GET", "/submissions/{submission_id}/task-events"),
    ("GET", "/submissions/{submission_id}/task-events/stream"),
    ("GET", "/agents/{agent_hash}/evaluation"),
    ("GET", "/leaderboard"),
    ("POST", "/owner/submissions/{submission_id}/revalidate"),
    ("POST", "/owner/submissions/{submission_id}/override"),
    ("POST", "/owner/submissions/{submission_id}/suspicious"),
    ("POST", "/owner/submissions/{submission_id}/admin-escalation"),
    ("GET", "/owner/audit"),
}
FORBIDDEN_SECRET_PATTERNS = {
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "openrouter_key": re.compile(r"\bsk-or-[A-Za-z0-9_-]{20,}\b"),
    "generic_long_secret_key": re.compile(r"\bsk-(?!test-|example-|placeholder)[A-Za-z0-9]{20,}\b"),
    "live_bearer_token": re.compile(r"Bearer (?!<)[A-Za-z0-9._~+/=-]{20,}"),
    "private_key_block": re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
    "live_database_url": re.compile(
        r"(?:postgresql?|mysql|mongodb(?:\+srv)?|redis)://(?!<)[^\s'\")]+",
        re.IGNORECASE,
    ),
    "mnemonic_material": re.compile(
        r"\b(?:seed phrase|mnemonic phrase|wallet private key)\b",
        re.IGNORECASE,
    ),
}

# Non-measured provider names that must never appear as agent-facing defaults in
# shipped AC docs/scripts (VAL-ACAT-014 flipped Base gateway; measured OpenRouter
# under .rules is now the legal review/eval LLM path and may be named). NOTE:
# this guard deliberately does NOT scan src/ — analyzer patterns redact providers.
PROVIDER_FORBIDDEN_TOKENS = (
    "deepseek",
    "deepseek-v4-pro",
    "chutes",
    "yunwu",
    # Raw OpenRouter key placeholders still forbidden in agent docs (keys stay in
    # measured encrypted_env only); product prose saying "measured OpenRouter" is OK.
    "<openrouter-api-key>",
    "sk-or-",
)

# Product and config phrases that embed openrouter transport names legitimately.
PROVIDER_TOKEN_ALLOWLIST_SUBSTRINGS = (
    "review_max_openrouter_request_bytes",
    "review_max_openrouter_response_bytes",
    "review_max_openrouter_metadata_bytes",
    "measured openrouter",
    "openrouter under",
    "openrouter.ai",
    "openrouter_api_key",
    "OPENROUTER_API_KEY",
)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def all_docs_text() -> str:
    return "\n".join(read(path) for path in DOC_PATHS)


def script_paths() -> tuple[Path, ...]:
    if not SCRIPTS_DIR.is_dir():
        return ()
    return tuple(
        sorted(
            path
            for path in SCRIPTS_DIR.rglob("*")
            if path.is_file() and path.suffix in {".py", ".sh"}
        )
    )


def provider_policy_scan_paths() -> tuple[Path, ...]:
    return (
        DOC_PATHS
        + (DOCS_INDEX_README, SUBMIT_AGENT_DOC, CONFIG_EXAMPLE)
        + tuple(sorted((ROOT / ".rules").glob("*.md")))
        + script_paths()
    )


def forbidden_provider_tokens_in(text: str) -> list[str]:
    cleaned = text
    for allowed in PROVIDER_TOKEN_ALLOWLIST_SUBSTRINGS:
        cleaned = re.sub(re.escape(allowed), "", cleaned, flags=re.IGNORECASE)
    lowered = cleaned.lower()
    return [token for token in PROVIDER_FORBIDDEN_TOKENS if token in lowered]


def route_table() -> set[tuple[str, str]]:
    routes: set[tuple[str, str]] = set()
    for route in iter_api_routes(app):
        for method in route.methods or set():
            if method not in {"HEAD", "OPTIONS"}:
                routes.add((method, route.path))
    return routes


def test_documented_endpoint_paths_exist_in_fastapi_route_table() -> None:
    docs = all_docs_text()
    available_routes = route_table()

    for method, path in DOCUMENTED_ROUTE_CONTRACT:
        assert f"{method} {path}" in docs or path in docs
        assert (method, path) in available_routes


def test_task18_operational_lifecycle_contract_is_documented() -> None:
    docs = all_docs_text()

    required_terms = (
        "one submission per hotkey",
        "3 hours",
        "ZIP receipt",
        "AST features",
        "similarity",
        "allow",
        "reject",
        "escalate",
        # VAL-ACAT-013/014: document Base gateway as **forbidden**, not required.
        "BASE_LLM_GATEWAY_URL",
        "base_gateway_forbidden",
        "measured OpenRouter",
        "terminal-bench/terminal-bench-2-1",
        "GET /submissions/{submission_id}/status",
        "GET /submissions/{submission_id}/events",
        "Last-Event-ID",
        "unknown Last-Event-ID",
        "replay_from",
        "POST /owner/submissions/{submission_id}/admin-escalation",
        "admin_allow",
        "admin_reject",
        "admin_request_rerun",
        "run_reconciler_once",
        "stable job dir",
        "harbor jobs resume -p <job_dir>",
        "Do not start duplicate Terminal-Bench jobs",
        "Known production caveats",
    )
    for term in required_terms:
        assert term in docs


def test_agate_agent_driven_order_is_documented() -> None:
    """VAL-AGATE-015: miner/validator docs describe package+rules → tree SHA → TEE → eval."""
    attestation = read(ROOT / "docs" / "miner" / "attestation-tee.md")
    evaluation = read(ROOT / "docs" / "evaluation.md")
    validator = read(VALIDATOR_README)
    miner = read(MINER_README)
    combined = "\n".join((attestation, evaluation, validator, miner))

    required_terms = (
        "agent-driven",
        "package_tree_sha",
        "LLM rules residual",
        "tree SHA",
        "TEE authorization",
        "ONLY THEN",
        "no closed catalog",
        "personal finetunes",
        "Host-static analyzer alone",
        "no eval prepare",
    )
    for term in required_terms:
        assert term in combined, f"missing AGATE docs term: {term}"

    # Explicit order language on both miner and validator surfaces.
    assert "package verify" in attestation.lower() or "Verify the package" in combined
    assert "Only then" in combined or "ONLY THEN" in combined
    assert "package_tree_sha" in attestation and "package_tree_sha" in evaluation
    assert "package_tree_sha" in validator


def test_repaired_lifecycle_status_contract_is_documented() -> None:
    docs = all_docs_text()

    raw_flow = (
        "analysis_queued -> ast_running -> llm_running -> analysis_allowed -> "
        "waiting_miner_env -> tb_queued -> tb_running"
    )
    required_terms = (
        raw_flow,
        "llm_standby -> analysis_queued",
        "AST review",
        "LLM review",
        "LLM standby",
        "Waiting environments",
        "evaluation queued",
        "evaluating",
        "ast_review",
        "llm_review",
        "llm_standby",
        "waiting_environments",
        "evaluation_queued",
        "evaluation",
        "missing_llm_gateway_token",
        "llm_provider_unavailable",
        "llm_provider_rate_limited",
        "llm_provider_timeout",
        "not rejection, escalation, or evaluation",
        "does not create `LlmVerdict`, `EvaluationJob`, `AdminReviewDecision`, or weights",
    )
    for term in required_terms:
        assert term in docs

    stale_terms = (
        "Waiting for miner " + "action",
        "`analysis" + "_running`",
        "waiting_miner" + "_action",
        "launch " + "locks",
        "OpenRouter review is " + "off until",
        "review stays " + "inert",
    )
    for term in stale_terms:
        assert term not in docs


def test_miner_env_auto_enqueue_and_launch_idempotency_are_documented() -> None:
    docs = all_docs_text()

    required_terms = (
        "locks/env-ready",
        "enqueues exactly once",
        "enqueue exactly once",
        "Repeat writes or repeated empty confirmation after lock return a conflict",
        "POST /submissions/{id}/launch` returns an existing queued or running job "
        "idempotently without duplicating it",
        "PUT /submissions/{id}/env` and `POST /submissions/{id}/env/confirm-empty`",
    )
    for term in required_terms:
        assert term in docs


def test_max_thirty_benchmark_policy_is_documented() -> None:
    docs = all_docs_text()

    required_terms = (
        "at most 30 benchmark tasks",
        "at most 30 task evaluations",
        "evaluation_task_count: 30",
        "evaluation_concurrency: 4",
        "config values above 30 are rejected",
        "capped by runtime helpers",
        "harbor_n_concurrent` is separate",
    )
    for term in required_terms:
        assert term in docs

    assert "evaluation_task_count: " + "4" not in docs


def test_gateway_llm_policy_is_documented_and_enforced() -> None:
    """VAL-ACAT-014: docs/policy forbid Base gateway; measured OpenRouter only."""

    docs = all_docs_text()
    rules = "\n".join(read(path) for path in sorted((ROOT / ".rules").glob("*.md")))

    required_terms = (
        "BaseIntelligence/baseagent",
        "BASE_LLM_GATEWAY_URL",
        "BASE_GATEWAY_TOKEN",
        "base_gateway_forbidden",
        "unauthorized_llm_provider",
        "hardcoded_llm_model",
        "measured OpenRouter",
        "automatically flags",
    )
    for term in required_terms:
        assert term in docs

    assert "BaseIntelligence/baseagent" in rules
    # Forbidden-as-residue (named in reject), never required-as-legal-route.
    assert "BASE_LLM_GATEWAY_URL" in rules
    assert "BASE_GATEWAY_TOKEN" in rules
    assert "must route all LLM traffic through the platform gateway" not in rules
    assert "measured OpenRouter" in rules or "tools-only" in rules


def test_no_legacy_llm_provider_policy_remains() -> None:
    scanned = "\n".join(read(path) for path in provider_policy_scan_paths())
    hits = forbidden_provider_tokens_in(scanned)
    assert not hits, f"forbidden provider/model reference(s) still present: {hits}"


def test_provider_policy_scan_covers_docs_index_and_scripts() -> None:
    paths = provider_policy_scan_paths()

    assert DOCS_INDEX_README in paths, "docs/README.md must be scanned for provider drift"

    scripts = script_paths()
    assert scripts, "expected shell/python scripts under scripts/ to scan"
    assert all(path.suffix in {".py", ".sh"} for path in scripts)
    assert set(scripts) <= set(paths)
    assert (SCRIPTS_DIR / "submit_agent.py") in paths

    # src/ is intentionally excluded: analyzer/pipeline.py enumerates provider names
    # (incl. YUNWU / yunwu.ai) in its output-sanitization regex and must keep scrubbing them.
    assert not any("src" in path.parts for path in paths)


def test_yunwu_is_a_forbidden_provider_token() -> None:
    assert "yunwu" in PROVIDER_FORBIDDEN_TOKENS


def test_provider_policy_guard_flags_forbidden_references() -> None:
    # DeepSeek / non-measured providers remain forbidden advertising.
    for token in (
        "deepseek",
        "deepseek-v4-pro",
        "chutes",
        "yunwu",
        "YUNWU_API_KEY",
        "DEEPSEEK_API_KEY",
    ):
        assert forbidden_provider_tokens_in(f"the agent calls {token} directly"), token
    # Measured OpenRouter product prose is allowed; raw provider marketing not needed.
    assert not forbidden_provider_tokens_in(
        "Legal path: measured OpenRouter under the review harness."
    )


def test_docs_use_placeholder_only_curl_examples() -> None:
    docs = all_docs_text()

    assert "<api-base-url>" in docs
    assert "<signature>" in docs
    assert "<owner-signature>" in docs
    assert "curl -X POST '<api-base-url>/submissions'" in docs
    assert "curl '<api-base-url>/submissions/<submission-id>/status'" in docs
    assert "curl -N" in docs


def test_docs_do_not_contain_obvious_real_secrets() -> None:
    docs_by_path = {path: read(path) for path in DOC_PATHS}

    for name, pattern in FORBIDDEN_SECRET_PATTERNS.items():
        for path, text in docs_by_path.items():
            assert not pattern.search(text), f"{name} matched {path}"


def test_frontend_platform_api_contract_is_documented() -> None:
    contract = read(FRONTEND_API_CONTRACT)
    platform_miner_doc = read(PLATFORM_MINER_README) if PLATFORM_MINER_README else None
    platform_has_frontend_contract = (
        platform_miner_doc is not None
        and "POST /challenges/agent-challenge/submissions" in platform_miner_doc
    )
    challenge_docs = "\n".join(
        (read(FRONTEND_API_CONTRACT), read(MINER_README), read(VALIDATOR_README))
    )

    required_routes = (
        "/v1/registry",
        "/challenges/agent-challenge/benchmarks",
        "/challenges/agent-challenge/submissions/{id}/status",
        "/challenges/agent-challenge/submissions/{id}/events",
        "/challenges/agent-challenge/leaderboard",
    )
    for route in required_routes:
        assert route in contract
        if platform_has_frontend_contract:
            assert route in platform_miner_doc
        assert route in challenge_docs

    required_terms = (
        "POST /v1/challenges/agent-challenge/submissions",
        "raw ZIP bridge",
        "POST /challenges/agent-challenge/submissions",
        "JSON base64",
        "latest 100 submissions newest-first",
        "one best scoring row per hotkey",
        "Pagination, filtering, and client-selected sorting are deferred to future v2",
        "/internal/*",
        "/health",
        "/version",
    )
    for term in required_terms:
        if platform_has_frontend_contract:
            assert term in platform_miner_doc
        assert term in challenge_docs


def test_submission_versioning_and_task_event_contracts_are_documented() -> None:
    public_guides = "\n".join((read(MINER_README), read(VALIDATOR_README)))

    required_terms = (
        "first successful submitter owns the normalized name globally within Agent Challenge",
        "duplicate_code_hash",
        "name_taken",
        "Duplicate hash checks take precedence over name ownership checks",
        "Duplicate artifact or code hashes are rejected globally, regardless of name or miner",
        "`v1`, `v2`, `v3`",
        "family_id",
        "display_name",
        "version_number",
        "version_label",
        "version_count",
        "latest_submission_id",
        "is_latest_version",
        "GET /submissions/{submission_id}/versions",
        "GET /submissions/{submission_id}/task-events",
        "GET /submissions/{submission_id}/task-events/stream",
        "TaskLogEvent.sequence",
        "cursor=0",
        "task_id",
        "event_type",
        "task_event_cursor_invalid",
        "Last-Event-ID",
        "cursor` takes precedence",
        "task.completed",
        "task.failed",
        "64KB/event",
        "10MB/task",
        "50MB/submission",
        "task_log_cap_reached",
        "submission_log_cap_reached",
        "cap_reached=true",
    )
    for term in required_terms:
        assert term in public_guides


def test_docs_security_boundaries_for_versions_and_task_events_are_documented() -> None:
    public_guides = "\n".join((read(MINER_README), read(VALIDATOR_README)))
    all_docs = all_docs_text().lower()

    forbidden_public_payload_terms = (
        "raw DB ids",
        "artifact paths",
        "worker paths",
        "stdout/stderr refs",
        "log refs",
        "private paths",
        "refs",
        "tokens",
        "signatures",
        "nonces",
        "normalized names",
        "canonical hashes",
        "raw artifact paths",
        "worker internals",
    )
    for term in forbidden_public_payload_terms:
        assert term in public_guides

    assert "not the raw `submission_family_id` database key" in public_guides
    assert "raw unbounded" in public_guides
    assert "permanent unlimited" in public_guides
    assert "unlimited logs" in public_guides
    assert "do not document or depend on unlimited logs" in public_guides.lower()
    assert "raw unbounded log downloads" not in all_docs
    assert "unlimited log downloads" not in all_docs


def test_frontend_contract_no_longer_marks_bridge_aliases_missing() -> None:
    contract = read(FRONTEND_API_CONTRACT)

    assert "Task 4" not in contract
    assert "Task 6" not in contract
    assert "MISSING" not in contract
    assert "POST /internal/v1/bridge/submissions" in contract
    assert "GET /v1/submissions/{id}" in contract
    assert "GET /v1/submissions/{id}/status" in contract


def test_signed_request_contract_is_documented_in_guides_and_config() -> None:
    docs = all_docs_text()
    config = read(CONFIG_EXAMPLE).replace("\\n", "\n")

    for text in (docs, config):
        assert CANONICAL_STRING in text
        for header in SIGNED_HEADERS:
            assert header in text
        assert "300" in text
        assert "(hotkey, nonce)" in text
        assert "409" in text


def test_owner_controls_contract_is_documented() -> None:
    validator_doc = read(VALIDATOR_README)
    config = read(CONFIG_EXAMPLE)

    assert OWNER_HOTKEY in validator_doc
    assert OWNER_HOTKEY in config
    for term in ("revalidate", "override", "suspicious", "/owner/audit"):
        assert term in validator_doc
    assert "append-only" in validator_doc
    assert "effective_status" in validator_doc
    assert "raw submission status" in validator_doc
    assert "persisted job evidence" in validator_doc
    assert "body hash/request hash, request timestamp" in validator_doc
    assert "body hash, canonical request" not in validator_doc
    assert "impersonation" not in validator_doc.lower()


def test_validator_role_is_documented_as_legacy_inert() -> None:
    docs = all_docs_text()
    config = read(CONFIG_EXAMPLE)

    for text in (docs, config):
        assert "validator_role" in text
        assert "legacy" in text
        assert "inert" in text

    combined = "\n".join((docs, config))
    assert "do not enqueue, claim, run, or evaluate" not in combined
    assert "on a master validator" not in combined
    assert "Only a `master`" not in combined


def test_zip_rules_container_and_hardcoding_limits_are_documented() -> None:
    docs = all_docs_text()
    config = read(CONFIG_EXAMPLE)

    for text in (docs, config):
        assert "1048576" in text
        assert "1MB" in text
        assert "compressed" in text
        assert "zip_too_large" in text
        assert "cpus=4.0" in text
        assert "memory=8g" in text
        assert "timeout_seconds=3600" in text
        assert "network=none" in text
    assert ".rules" in docs
    assert "Missing `.rules` returns `error`" in docs
    assert "evidence-based, bounded, owner-auditable" in docs
    assert "not proof" in docs


def test_effective_status_weight_contract_is_documented() -> None:
    docs = all_docs_text()

    for status in (
        "completed",
        "overridden_valid",
        "suspicious",
        "invalid",
        "error",
        "overridden_invalid",
    ):
        assert status in docs
    assert "can produce weights" in docs or "can appear on the leaderboard" in docs
    assert "excluded" in docs


def test_per_task_aggregation_knob_documented_in_config_example() -> None:
    config = read(CONFIG_EXAMPLE)

    # The operator-facing per-task aggregation knob is documented with its default.
    assert 'per_task_aggregation: "mean"' in config
    # Both accepted modes appear.
    for token in ("mean", "best-of-k"):
        assert token in config
    # The per-task-TRIAL vs per-job-TASK "best-of-k" distinction is clarified so an
    # operator does not confuse it with the keep_good_tasks_policy "best-of-k".
    assert "distinct from" in config
    assert "keep_good_tasks_policy" in config


def test_config_example_matches_security_defaults() -> None:
    config = read(CONFIG_EXAMPLE)
    settings = ChallengeSettings()

    assert f"validator_role: {settings.validator_role}" in config
    assert f"owner_hotkey: {settings.owner_hotkey}" in config
    assert f"signing_ttl_seconds: {settings.signing_ttl_seconds}" in config
    assert f"zip_max_bytes: {settings.zip_max_bytes}" in config
    assert f"docker_cpus: {settings.docker_cpus}" in config
    assert f"docker_memory: {settings.docker_memory}" in config
    assert f"evaluation_timeout_seconds: {settings.evaluation_timeout_seconds}" in config


def test_terminal_bench_own_runner_is_the_only_backend() -> None:
    root_readme = read(README)
    validator_doc = read(VALIDATOR_README)
    config = read(CONFIG_EXAMPLE)
    combined = "\n".join((root_readme, validator_doc, config))

    assert "terminal_bench_execution_backend: own_runner" in config
    assert "CHALLENGE_TERMINAL_BENCH_EXECUTION_BACKEND=own_runner" in validator_doc
    assert "only supported execution backend" in combined
    assert "There is no runtime Harbor install path" in validator_doc
    assert "There is no execution-backend rollback" in validator_doc
    assert "any other value is rejected by settings" in validator_doc
    assert 'docker_backend="cli"' not in combined
    assert 'harbor_install_mode="runtime"' not in combined
    assert "Roll back to `harbor`" not in combined


def test_docs_do_not_claim_automatic_background_evaluation() -> None:
    combined = "\n".join((read(README), read(VALIDATOR_README), read(CONFIG_EXAMPLE)))

    assert "automatic background evaluation" not in combined.lower()


def test_dependency_and_ci_contract_stays_minimal() -> None:
    pyproject = read(PYPROJECT)
    ci = read(CI)

    assert "ruff check ." in ci
    assert "pytest" in ci
    assert "langchain" not in pyproject.lower()
    assert "substrateinterface" not in pyproject.lower()


def test_docker_executor_capability_contract_is_documented() -> None:
    validator_doc = read(VALIDATOR_README)

    assert "docker_executor" in validator_doc
    assert "required_capabilities" in validator_doc
    assert "/run/secrets/base/docker_broker_token" in validator_doc
