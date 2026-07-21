"""Master-embed docs seal contracts (VAL-MEMB-009, VAL-MEMB-011 companions).

Locks public challenge path prefixes, embed topology narrative in architecture /
compose / deploy docs, and safety wording (no master set_weights; no multi-writer
SQLite; secrets names-only guidance).
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

PUBLIC_SLUGS = (
    "/challenges/prism",
    "/challenges/agent-challenge",
)

# Operator-facing docs that must advertise both public challenge prefixes.
SLUG_DOC_PATHS = (
    "docs/architecture.md",
    "docs/compose.md",
    "docs/deploy.md",
    "docs/master/README.md",
    "docs/security.md",
    "docs/challenges.md",
    "docs/SOURCE_OF_TRUTH.md",
)

APP_PROXY = REPO_ROOT / "src/base/master/app_proxy.py"
COMPOSE_YML = REPO_ROOT / "deploy/compose/docker-compose.yml"
ENTRYPOINT = REPO_ROOT / "docker/master-entrypoint.sh"


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def test_public_challenge_paths_unchanged_across_proxy_and_docs() -> None:
    """VAL-MEMB-009: /challenges/prism and /challenges/agent-challenge stay public."""
    proxy = APP_PROXY.read_text(encoding="utf-8")
    assert "/challenges/" in proxy
    assert '"/challenges/{slug}"' in proxy or "'/challenges/{slug}'" in proxy
    assert (
        '"/challenges/{slug}/{path:path}"' in proxy
        or "'/challenges/{slug}/{path:path}'" in proxy
    )
    # Proxy still uses httpx reverse-proxy (not full ASGI mount rewrite).
    assert "httpx" in proxy

    for rel in SLUG_DOC_PATHS:
        text = _read(rel)
        for slug in PUBLIC_SLUGS:
            assert slug in text, f"{rel} missing public slug {slug}"


def test_architecture_compose_deploy_describe_master_embed() -> None:
    """Docs architecture/compose/deploy document embed localhost topology."""
    arch = _read("docs/architecture.md")
    compose = _read("docs/compose.md")
    deploy = _read("docs/deploy.md")
    master = _read("docs/master/README.md")
    blob = "\n".join((arch, compose, deploy, master))

    assert "127.0.0.1:18080" in blob
    assert "127.0.0.1:18081" in blob
    assert "base-master-validator" in blob
    assert "master-postgres" in blob
    assert "embedded" in blob.lower() or "embeds" in blob.lower()
    # Shipping default: no separate challenge-* services as required topology.
    assert "no" in blob.lower() and "challenge-prism" in blob
    assert "weight-only" in blob.lower() or "weight only" in blob.lower()
    assert "https://chain.joinbase.ai" in blob
    assert "set_weights" in blob
    # Master never submits on-chain.
    assert re.search(r"never.*set_weights|set_weights.*never", blob, re.I)

    # Compose file still has no challenge-* services.
    compose_yml = COMPOSE_YML.read_text(encoding="utf-8")
    assert "challenge-prism:" not in compose_yml
    assert "challenge-agent-challenge:" not in compose_yml

    # Entrypoint still binds loopback only.
    entry = ENTRYPOINT.read_text(encoding="utf-8")
    assert "127.0.0.1" in entry
    assert "18080" in entry
    assert "18081" in entry


def test_docs_do_not_require_separate_challenge_compose_services() -> None:
    """Shipping docs must not present challenge-* as required cardinality."""
    for rel in ("docs/architecture.md", "docs/deploy.md", "docs/compose.md"):
        text = _read(rel)
        # Forbidden required-cardinality challenge service rows.
        assert "one `challenge-<slug>`" not in text, rel
        assert "one long-lived `challenge-<slug>`" not in text, rel
        assert "| one `challenge-<slug>` |" not in text, rel


def test_safety_docs_no_multi_writer_and_no_master_set_weights() -> None:
    """VAL-MEMB-011: sole writer + no master set_weights + secrets files."""
    security = _read("docs/security.md")
    arch = _read("docs/architecture.md")
    blob = "\n".join((security, arch))

    assert "multi-writer" in blob.lower() or "sole writer" in blob.lower()
    assert "never" in arch.lower() and "set_weights" in arch
    # Secrets stay file-backed / not embedded in manifests.
    secrets_ok = (
        "0600" in security or "*_FILE" in security or "secret files" in security.lower()
    )
    assert secrets_ok
    # No instruction for master to set_weights.
    assert not re.search(
        r"master\s+(must|should|can|will)\s+set_weights",
        blob,
        re.I,
    )


def test_validator_docs_remain_weight_only_joinbase() -> None:
    """Cross-check M3 narrative remains visible after docs seal."""
    validator = _read("docs/validator/README.md")
    assert "https://chain.joinbase.ai" in validator
    assert "weight-only" in validator.lower() or "weight only" in validator.lower()
