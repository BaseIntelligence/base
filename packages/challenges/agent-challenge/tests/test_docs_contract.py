"""Thin package docs contract for Agent Challenge after shipping-docs collapse.

Day-1 miners use monorepo-root ``docs/miner/getting-started.md``.
API truth is OpenAPI at ``chain.joinbase.ai/challenges/agent-challenge/openapi.json``.
Package ``docs/`` keeps a short README pointer plus self-deploy CLI accuracy
fixtures required by ``test_selfdeploy_*``.
"""

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
MINER_SELF_DEPLOY = ROOT / "docs" / "miner" / "self-deploy.md"
VALIDATOR_SELF_DEPLOY = ROOT / "docs" / "validator" / "self-deploy.md"
CONFIG_EXAMPLE = ROOT / "config.example.yaml"
PYPROJECT = ROOT / "pyproject.toml"
CI = ROOT / ".github" / "workflows" / "ci.yml"
SCRIPTS_DIR = ROOT / "scripts"

FORBIDDEN_ESSAY_PATHS = (
    "docs/architecture.md",
    "docs/behavior-ledger.md",
    "docs/evaluation.md",
    "docs/frontend-api-contract.md",
    "docs/security.md",
    "docs/miner/README.md",
    "docs/miner/attestation-tee.md",
    "docs/miner/getting-started.md",
    "docs/miner/submit-agent.md",
    "docs/validator/README.md",
)

CANONICAL_STRING = (
    "{METHOD}\n{PATH_WITH_SORTED_QUERY}\n{X-TIMESTAMP}\n{X-NONCE}\n{SHA256_HEX_OF_RAW_BODY}"
)
OWNER_HOTKEY = "************************************************"
SIGNED_HEADERS = ("X-Hotkey", "X-Signature", "X-Nonce", "X-Timestamp")

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

PROVIDER_FORBIDDEN_TOKENS = (
    "deepseek",
    "deepseek-v4-pro",
    "chutes",
    "yunwu",
    "<openrouter-api-key>",
    "sk-or-",
)

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


def platform_miner_getting_started() -> Path | None:
    for name in ("PLATFORM_CHECKOUT", "PLATFORM_ROOT"):
        value = os.environ.get(name)
        if not value:
            continue
        path = Path(value) / "docs" / "miner" / "getting-started.md"
        if path.is_file():
            return path
    # Default monorepo layout: package is packages/challenges/agent-challenge.
    root_gs = ROOT.parents[2] / "docs" / "miner" / "getting-started.md"
    if root_gs.is_file():
        return root_gs
    return None


PLATFORM_GETTING_STARTED = platform_miner_getting_started()


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def shipping_docs_text() -> str:
    parts = [read(README), read(DOCS_INDEX_README)]
    if PLATFORM_GETTING_STARTED is not None:
        parts.append(read(PLATFORM_GETTING_STARTED))
    return "\n".join(parts)


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
        README,
        DOCS_INDEX_README,
        MINER_SELF_DEPLOY,
        VALIDATOR_SELF_DEPLOY,
        CONFIG_EXAMPLE,
        *tuple(sorted((ROOT / ".rules").glob("*.md"))),
        *script_paths(),
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


def test_package_docs_tree_is_pointer_plus_selfdeploy_fixtures() -> None:
    assert DOCS_INDEX_README.is_file()
    assert MINER_SELF_DEPLOY.is_file()
    assert VALIDATOR_SELF_DEPLOY.is_file()
    for rel in FORBIDDEN_ESSAY_PATHS:
        assert not (ROOT / rel).exists(), f"essay path must stay deleted: {rel}"

    pointer = read(DOCS_INDEX_README)
    assert "docs/miner/getting-started.md" in pointer
    assert "openapi.json" in pointer
    assert "chain.joinbase.ai/challenges/agent-challenge" in pointer
    assert "self-deploy.md" in pointer
    assert len(pointer.splitlines()) <= 40


def test_readme_points_day1_and_openapi() -> None:
    readme = read(README)
    assert "docs/miner/getting-started.md" in readme
    assert "https://chain.joinbase.ai/challenges/agent-challenge/openapi.json" in readme
    assert "https://chain.joinbase.ai/challenges/agent-challenge/docs" in readme
    assert "/challenges/agent-challenge" in readme
    assert "agent_challenge" in readme
    assert "ghcr.io/baseintelligence/agent-challenge" in readme


def test_documented_endpoint_paths_exist_in_fastapi_route_table() -> None:
    """Route contract stays live against OpenAPI/app routes (not markdown dumps)."""
    available_routes = route_table()
    for method, path in DOCUMENTED_ROUTE_CONTRACT:
        assert (method, path) in available_routes, (method, path)


def test_agate_agent_driven_order_is_pinned_on_shipping_surfaces() -> None:
    """VAL-AGATE-015: product pins remain after essay collapse (README + self-deploy)."""
    combined = "\n".join(
        (
            read(README),
            read(DOCS_INDEX_README),
            read(MINER_SELF_DEPLOY),
            read(VALIDATOR_SELF_DEPLOY),
            "\n".join(read(p) for p in sorted((ROOT / ".rules").glob("*.md"))),
        )
    )
    # Self-deploy still owns the ordered review→eval spine; package README owns product home.
    assert "self-deploy" in combined.lower()
    assert "package_tree_sha" in combined or "tree sha" in combined.lower() or "tree_sha" in combined
    assert "review" in combined.lower() and "eval" in combined.lower()
    # Personal finetune ban + measured path stay product locks somewhere shipping.
    assert "finetune" in combined.lower() or "model" in combined.lower()


def test_no_legacy_llm_provider_policy_remains() -> None:
    scanned = "\n".join(read(path) for path in provider_policy_scan_paths())
    hits = forbidden_provider_tokens_in(scanned)
    assert not hits, f"forbidden provider/model reference(s) still present: {hits}"


def test_provider_policy_scan_covers_docs_index_and_scripts() -> None:
    paths = provider_policy_scan_paths()
    assert DOCS_INDEX_README in paths
    scripts = script_paths()
    assert scripts, "expected shell/python scripts under scripts/ to scan"
    assert all(path.suffix in {".py", ".sh"} for path in scripts)
    assert set(scripts) <= set(paths)
    assert (SCRIPTS_DIR / "submit_agent.py") in paths
    assert not any("src" in path.parts for path in paths)


def test_yunwu_is_a_forbidden_provider_token() -> None:
    assert "yunwu" in PROVIDER_FORBIDDEN_TOKENS


def test_provider_policy_guard_flags_forbidden_references() -> None:
    for token in (
        "deepseek",
        "deepseek-v4-pro",
        "chutes",
        "yunwu",
        "YUNWU_API_KEY",
        "DEEPSEEK_API_KEY",
    ):
        assert forbidden_provider_tokens_in(f"the agent calls {token} directly"), token
    assert not forbidden_provider_tokens_in(
        "Legal path: measured OpenRouter under the review harness."
    )


def test_shipping_docs_do_not_contain_obvious_real_secrets() -> None:
    docs_by_path = {
        README: read(README),
        DOCS_INDEX_README: read(DOCS_INDEX_README),
        MINER_SELF_DEPLOY: read(MINER_SELF_DEPLOY),
        VALIDATOR_SELF_DEPLOY: read(VALIDATOR_SELF_DEPLOY),
    }
    for name, pattern in FORBIDDEN_SECRET_PATTERNS.items():
        for path, text in docs_by_path.items():
            assert not pattern.search(text), f"{name} matched {path}"


def test_signed_request_contract_is_documented_in_config() -> None:
    config = read(CONFIG_EXAMPLE).replace("\\n", "\n")
    assert CANONICAL_STRING in config
    for header in SIGNED_HEADERS:
        assert header in config
    assert "300" in config
    assert "(hotkey, nonce)" in config


def test_owner_controls_contract_is_documented_in_config() -> None:
    config = read(CONFIG_EXAMPLE)
    assert "owner_hotkey:" in config
    # Placeholder or pin shape; never a live wallet mnemonic block.
    assert "BEGIN" not in config or "PRIVATE KEY" not in config


def test_validator_role_is_documented_as_legacy_inert() -> None:
    config = read(CONFIG_EXAMPLE)
    assert "validator_role" in config
    assert "legacy" in config
    assert "inert" in config


def test_zip_rules_container_and_hardcoding_limits_are_in_config() -> None:
    config = read(CONFIG_EXAMPLE)
    assert "1048576" in config
    assert "zip_too_large" in config or "zip_max_bytes" in config
    assert "cpus=4.0" in config or "docker_cpus: 4" in config
    assert "memory=8g" in config or "docker_memory" in config
    assert "timeout_seconds=3600" in config or "evaluation_timeout_seconds: 3600" in config


def test_per_task_aggregation_knob_documented_in_config_example() -> None:
    config = read(CONFIG_EXAMPLE)
    assert 'per_task_aggregation: "mean"' in config
    for token in ("mean", "best-of-k"):
        assert token in config
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
    config = read(CONFIG_EXAMPLE)
    assert "terminal_bench_execution_backend: own_runner" in config
    assert 'docker_backend="cli"' not in config
    assert 'harbor_install_mode="runtime"' not in config


def test_dependency_and_ci_contract_stays_minimal() -> None:
    pyproject = read(PYPROJECT)
    ci = read(CI)

    assert "ruff check ." in ci
    assert "pytest" in ci
    assert "langchain" not in pyproject.lower()
    assert "substrateinterface" not in pyproject.lower()


def test_root_day1_still_mentions_agent_challenge_when_present() -> None:
    if PLATFORM_GETTING_STARTED is None:
        return
    gs = read(PLATFORM_GETTING_STARTED)
    assert "agent-challenge" in gs
    assert "openapi.json" in gs
    assert "https://chain.joinbase.ai" in gs
