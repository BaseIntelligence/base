"""Master-embed docs seal contracts (VAL-MEMB-009, VAL-MEMB-011 companions).

Locks public challenge path prefixes, embed topology narrative in remaining
shipping docs (compose/validator/README), and safety wording.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

PUBLIC_SLUGS = (
    "/challenges/prism",
    "/challenges/agent-challenge",
)

# Operator-facing shipping docs that must advertise both public challenge prefixes.
SLUG_DOC_PATHS = (
    "docs/compose.md",
    "docs/validator.md",
    "docs/miner/getting-started.md",
    "README.md",
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
    assert "httpx" in proxy

    for rel in SLUG_DOC_PATHS:
        text = _read(rel)
        for slug in PUBLIC_SLUGS:
            assert slug in text, f"{rel} missing public slug {slug}"


def test_architecture_compose_deploy_describe_master_embed() -> None:
    """Compose/validator shipping docs document embed localhost topology."""
    compose = _read("docs/compose.md")
    validator = _read("docs/validator.md")
    readme = _read("README.md")
    blob = "\n".join((compose, validator, readme))

    assert "127.0.0.1:18080" in blob or "18080" in blob
    assert "127.0.0.1:18081" in blob or "18081" in blob
    assert "base-master-validator" in blob
    assert "master-postgres" in blob
    assert "embedded" in blob.lower() or "embeds" in blob.lower()
    assert "no" in blob.lower() and "challenge-prism" in blob
    assert "weight-only" in blob.lower() or "weight only" in blob.lower()
    assert "https://chain.joinbase.ai" in blob
    assert "set_weights" in blob
    assert re.search(r"never.*set_weights|set_weights.*never", blob, re.I)

    compose_yml = COMPOSE_YML.read_text(encoding="utf-8")
    assert "challenge-prism:" not in compose_yml
    assert "challenge-agent-challenge:" not in compose_yml

    entry = ENTRYPOINT.read_text(encoding="utf-8")
    assert "127.0.0.1" in entry
    assert "18080" in entry
    assert "18081" in entry


def test_docs_do_not_require_separate_challenge_compose_services() -> None:
    """Shipping docs must not present challenge-* as required cardinality."""
    for rel in ("docs/compose.md", "docs/validator.md", "README.md"):
        text = _read(rel)
        assert "one `challenge-<slug>`" not in text, rel
        assert "one long-lived `challenge-<slug>`" not in text, rel
        assert "| one `challenge-<slug>` |" not in text, rel


def test_safety_docs_no_multi_writer_and_no_master_set_weights() -> None:
    """VAL-MEMB-011: sole writer + no master set_weights + secrets files."""
    compose = _read("docs/compose.md")
    validator = _read("docs/validator.md")
    readme = _read("README.md")
    blob = "\n".join((compose, validator, readme))

    assert "multi-writer" in blob.lower() or "sole writer" in blob.lower()
    assert "never" in blob.lower() and "set_weights" in blob
    secrets_ok = (
        "0600" in blob or "*_FILE" in blob or "secret" in blob.lower()
    )
    assert secrets_ok
    assert not re.search(
        r"master\s+(must|should|can|will)\s+set_weights",
        blob,
        re.I,
    )


def test_validator_docs_remain_weight_only_joinbase() -> None:
    """Cross-check weight-only narrative remains visible after docs seal."""
    validator = _read("docs/validator.md")
    assert "https://chain.joinbase.ai" in validator
    assert "weight-only" in validator.lower() or "weight only" in validator.lower()
