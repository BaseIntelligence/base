"""Master-embed challenge scaffold contracts (VAL-MEMB-001, VAL-MEMB-002).

Master image installs monorepo prism-challenge + agent-challenge and supervises
localhost uvicorn alongside the public proxy. Public /challenges/* proxy path
logic stays in app_proxy (httpx); this module only locks image + entrypoint.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE_MASTER = REPO_ROOT / "docker/Dockerfile.master"
ENTRYPOINT = REPO_ROOT / "docker/master-entrypoint.sh"
COMPOSE_YML = REPO_ROOT / "deploy/compose/docker-compose.yml"
DOCS_COMPOSE = REPO_ROOT / "docs/compose.md"
DOCS_MASTER = REPO_ROOT / "docs/master/README.md"
DOCS_MONOREPO = REPO_ROOT / "docs/monorepo.md"
APP_PROXY = REPO_ROOT / "src/base/master/app_proxy.py"


def _non_comment_lines(text: str) -> str:
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


# --- Dockerfile.master installs monorepo challenge packages (VAL-MEMB-001) ---


def test_master_dockerfile_exists() -> None:
    assert DOCKERFILE_MASTER.is_file()
    assert ENTRYPOINT.is_file()


def test_master_dockerfile_installs_challenge_packages() -> None:
    text = DOCKERFILE_MASTER.read_text(encoding="utf-8")
    instructions = _non_comment_lines(text)

    assert "packages/challenges/prism" in text
    assert "packages/challenges/agent-challenge" in text
    assert 'uv pip install --system -e ".[master]"' in instructions
    assert "uv export --package agent-challenge" in instructions
    assert "uv export --package prism-challenge" in instructions
    assert "uv build --package agent-challenge" in instructions
    assert "uv build --package prism-challenge" in instructions
    assert "--no-deps" in instructions
    assert "--no-emit-package base" in instructions
    # Build-time import smoke for both challenge ASGI modules + uvicorn.
    assert "prism_challenge.app" in instructions
    assert "agent_challenge.app" in instructions
    assert "uvicorn" in instructions


def test_master_dockerfile_has_no_external_challenge_clone() -> None:
    text = DOCKERFILE_MASTER.read_text(encoding="utf-8")
    instructions = _non_comment_lines(text)
    for marker in (
        "git clone",
        "AGENT_CHALLENGE_REF",
        "PRISM_REF",
        "github.com/BaseIntelligence/prism",
        "github.com/BaseIntelligence/agent-challenge",
    ):
        assert marker not in instructions, marker


def test_master_dockerfile_wires_entrypoint_and_proxy_port() -> None:
    text = DOCKERFILE_MASTER.read_text(encoding="utf-8")
    assert "base-master-entrypoint" in text
    assert 'ENTRYPOINT ["/usr/local/bin/base-master-entrypoint"]' in text
    assert 'CMD ["base", "master", "proxy"' in text
    assert "EXPOSE 8081" in text
    assert "BASE_MASTER_EMBED_CHALLENGES=1" in text
    assert "18080" in text
    assert "18081" in text
    assert "/var/lib/base/challenges/prism" in text
    assert "/var/lib/base/challenges/agent-challenge" in text


# --- Supervisor launches proxy + localhost challenge uvicons (VAL-MEMB-002) ---


def test_master_entrypoint_script_is_executable_shell() -> None:
    text = ENTRYPOINT.read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash")
    # Syntax check without running live processes.
    completed = subprocess.run(
        ["bash", "-n", str(ENTRYPOINT)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_master_entrypoint_binds_challenges_to_loopback_ports() -> None:
    text = ENTRYPOINT.read_text(encoding="utf-8")
    assert 'PRISM_HOST="${BASE_MASTER_PRISM_HOST:-127.0.0.1}"' in text
    assert 'PRISM_PORT="${BASE_MASTER_PRISM_PORT:-18080}"' in text
    assert 'AC_HOST="${BASE_MASTER_AC_HOST:-127.0.0.1}"' in text
    assert 'AC_PORT="${BASE_MASTER_AC_PORT:-18081}"' in text
    assert "uvicorn prism_challenge.app:app" in text
    assert "uvicorn agent_challenge.app:app" in text
    assert '--host "${PRISM_HOST}"' in text
    assert '--port "${PRISM_PORT}"' in text
    assert '--host "${AC_HOST}"' in text
    assert '--port "${AC_PORT}"' in text
    # Must not advertise challenge ASGI on all interfaces by default.
    assert not re.search(r'--host\s+["\']?0\.0\.0\.0', text)


def test_master_entrypoint_data_paths_and_shared_tokens() -> None:
    text = ENTRYPOINT.read_text(encoding="utf-8")
    assert "/var/lib/base/challenges/prism" in text
    assert "/var/lib/base/challenges/agent-challenge" in text
    assert "PRISM_SHARED_TOKEN_FILE" in text
    assert "CHALLENGE_SHARED_TOKEN_FILE" in text
    assert "PRISM_DATABASE_URL" in text
    assert "CHALLENGE_DATABASE_URL" in text
    assert "PRISM_MASTER_BASE_URL" in text
    assert "http://127.0.0.1:8081" in text


def test_master_entrypoint_dual_run_opt_out() -> None:
    """Compose challenge-* services remain optional during dual-run (M1)."""
    text = ENTRYPOINT.read_text(encoding="utf-8")
    assert "BASE_MASTER_EMBED_CHALLENGES" in text
    assert "embed_truthy" in text
    assert "skipping embedded challenge ASGI" in text


def test_master_entrypoint_runs_master_proxy_command() -> None:
    text = ENTRYPOINT.read_text(encoding="utf-8")
    assert "base master proxy" in text
    # Entrypoint waits on the master child; does not drop proxy path args.
    assert 'wait "${MASTER_PID}"' in text or "wait " in text


# --- Docs ports + public proxy path unchanged ---


def test_docs_document_embed_ports() -> None:
    compose = DOCS_COMPOSE.read_text(encoding="utf-8")
    master = DOCS_MASTER.read_text(encoding="utf-8")
    monorepo = DOCS_MONOREPO.read_text(encoding="utf-8")
    blob = "\n".join((compose, master, monorepo))
    assert "18080" in blob
    assert "18081" in blob
    assert "127.0.0.1" in blob
    assert "master-entrypoint" in blob or "embedded" in blob.lower()
    assert "/var/lib/base/challenges" in blob


def test_public_challenge_proxy_paths_unchanged() -> None:
    """Public /challenges/{prism,agent-challenge} prefixes stay on proxy httpx."""
    proxy = APP_PROXY.read_text(encoding="utf-8")
    assert "/challenges/" in proxy
    # Generic httpx forwarder remains; no ASGI mount rewrite of challenge apps.
    assert "httpx" in proxy


def test_compose_still_allows_dual_run_challenge_service() -> None:
    """M1 keeps optional challenge-prism service; drop is a later milestone."""
    text = COMPOSE_YML.read_text(encoding="utf-8")
    assert "challenge-prism:" in text
    assert "base-master-validator:" in text
    assert "8081" in text
