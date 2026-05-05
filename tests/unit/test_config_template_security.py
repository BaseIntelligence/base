from __future__ import annotations

from pathlib import Path

from platform_network.config.loader import load_settings
from platform_network.security.tokens import generate_token, hash_token, verify_token
from platform_network.template_engine import (
    ChallengeTemplateContext,
    render_challenge_template,
)


def test_token_hash_verify() -> None:
    token = generate_token()
    assert verify_token(token, hash_token(token))
    assert not verify_token("wrong", hash_token(token))


def test_load_settings_yaml(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("network:\n  netuid: 42\n", encoding="utf-8")
    assert load_settings(config).network.netuid == 42


def test_render_challenge_template(tmp_path: Path) -> None:
    out = tmp_path / "challenge"
    files = render_challenge_template(
        out, ChallengeTemplateContext.from_slug("demo-challenge")
    )
    assert Path("pyproject.toml") in files
    assert (out / "src" / "demo_challenge" / "app.py").exists()
