"""Red/green absence tests for the removed Base LLM gateway surface."""

from __future__ import annotations

import importlib
import importlib.metadata
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from base.config.loader import load_settings
from base.config.settings import Settings
from base.master.agent_challenge_compat import (
    AGENT_CHALLENGE_INCOMPATIBLE_CODE,
    agent_challenge_incompatibility,
    is_agent_challenge_slug,
)
from base.master.app_proxy import create_proxy_app


def test_llm_gateway_module_is_not_importable() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("base.master.llm_gateway")


def test_settings_schema_has_no_gateway_object() -> None:
    assert "gateway" not in Settings.model_fields
    settings = Settings()
    dumped = settings.model_dump()
    assert "gateway" not in dumped
    assert "gateway_url" not in dumped.get("validator", {}).get("agent", {})
    assert "gateway_url" not in dumped.get("worker", {}).get("agent", {})


def test_clean_settings_load_without_gateway_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "BASE_GATEWAY__TOKEN_SECRET",
        "BASE_GATEWAY_TOKEN",
        "BASE_LLM_GATEWAY_URL",
        "GATEWAY_TOKEN",
        "CENTRAL_GATEWAY_TOKEN",
        "PRISM_LLM_GATEWAY_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    cfg = tmp_path / "master.yaml"
    cfg.write_text("environment: development\n", encoding="utf-8")
    settings = load_settings(cfg)
    assert settings.environment == "development"


@pytest.mark.parametrize(
    "payload",
    [
        {"gateway": {"provider_mode": "mock"}},
        {"validator": {"agent": {"gateway_url": "http://example"}}},
        {"worker": {"agent": {"gateway_url": "http://example"}}},
    ],
)
def test_legacy_gateway_config_is_rejected(tmp_path: Path, payload: dict) -> None:
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="removed LLM gateway"):
        load_settings(cfg)


def test_legacy_gateway_env_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BASE_GATEWAY_TOKEN", "should-not-load")
    cfg = tmp_path / "clean.yaml"
    cfg.write_text("environment: development\n", encoding="utf-8")
    with pytest.raises(ValueError, match="removed LLM gateway"):
        load_settings(cfg)


def test_gateway_free_digest_allowlist_env_does_not_reject_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Allowlist for attestation-only AC digests is not a restored LLM gateway."""

    digest = "sha256:" + ("b" * 64)
    monkeypatch.setenv("BASE_AGENT_CHALLENGE_GATEWAY_FREE_DIGESTS", digest)
    # Ensure residual provider keys are not present.
    for key in (
        "BASE_GATEWAY_TOKEN",
        "BASE_LLM_GATEWAY_URL",
        "GATEWAY_TOKEN",
        "CENTRAL_GATEWAY_TOKEN",
        "PRISM_LLM_GATEWAY_URL",
    ):
        monkeypatch.delenv(key, raising=False)
    cfg = tmp_path / "clean.yaml"
    cfg.write_text("environment: development\n", encoding="utf-8")
    settings = load_settings(cfg)
    assert settings is not None


def test_cli_has_no_mint_central_gate_token() -> None:
    from base.cli_app import main as cli_main

    master = cli_main.master_app
    command_names = {cmd.name for cmd in master.registered_commands}
    assert "mint-central-gate-token" not in command_names
    assert not hasattr(cli_main, "master_mint_central_gate_token")


def test_proxy_openapi_has_no_llm_gateway_paths() -> None:
    """VAL-ACAT-043: Base never restores /llm/v1 proxy routes post unlock."""

    class FakeCache:
        def get(self) -> dict[str, int]:
            return {}

    class FakeNonce:
        async def reserve(self, **_: object) -> None:
            return None

    class FakeRegistry:
        async def list(self):  # pragma: no cover - not exercised
            return []

        async def get(self, slug: str):  # pragma: no cover
            raise KeyError(slug)

    class MinerVerifier:
        async def verify(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError("unused")

    app = create_proxy_app(
        registry=FakeRegistry(),
        miner_verifier=MinerVerifier(),  # type: ignore[arg-type]
        agent_challenge_attested_routes_enabled=True,
    )
    client = TestClient(app)
    openapi = client.get("/openapi.json").json()
    paths = openapi.get("paths", {})
    joined = "\n".join(paths)
    assert "/llm/v1" not in joined
    assert "gateway_token" not in joined
    assert "chat/completions" not in joined
    assert "architecture" not in joined or "/v1/architectures/" not in joined
    # Former gateway path returns normal not-found through the ASGI app.
    for method in ("get", "post", "put", "delete"):
        for path in (
            "/llm/v1/chat/completions",
            "/v1/llm/chat/completions",
            "/llm/v1/models",
        ):
            response = getattr(client, method)(path)
            assert response.status_code == 404, (method, path)


def test_agent_challenge_incompatibility_diagnostic() -> None:
    assert is_agent_challenge_slug("agent-challenge")
    diagnostic = agent_challenge_incompatibility().as_dict()
    assert diagnostic["code"] == AGENT_CHALLENGE_INCOMPATIBLE_CODE
    assert "removed LLM gateway" in diagnostic["message"]
    assert "Do not set a legacy gateway token" in diagnostic["message"]
    assert "adapter" in diagnostic["message"].lower()


def test_seed_prism_challenges_blocks_agent_challenge_without_gateway_free_digest() -> (
    None
):
    """VAL-ACAT-042: unpinned / pre-upgrade AC remains refuse diagnostic."""

    import asyncio

    from base.cli_app.main import seed_prism_challenges
    from base.master.registry import ChallengeNotFoundError

    class Registry:
        def __init__(self) -> None:
            self.created = 0
            self.updated = 0
            self.has_agent = True

        async def get(self, slug: str):
            if slug == "prism":
                return type(
                    "R",
                    (),
                    {
                        "metadata": {},
                        "env": {},
                        "secrets": [],
                        "required_capabilities": [],
                        "image": "ghcr.io/example/prism:1@sha256:" + ("d" * 64),
                    },
                )()
            if slug == "agent-challenge" and self.has_agent:
                # Legacy pre-upgrade pin (not on gateway-free allowlist).
                return type(
                    "R",
                    (),
                    {
                        "metadata": {},
                        "env": {},
                        "secrets": [],
                        "required_capabilities": [],
                        "image": (
                            "ghcr.io/example/agent-challenge:legacy@sha256:"
                            + ("a" * 64)
                        ),
                    },
                )()
            raise ChallengeNotFoundError(slug)

        async def create(self, payload):  # pragma: no cover
            self.created += 1
            return payload

        async def update(self, slug, payload):  # pragma: no cover
            self.updated += 1
            return payload

    registry = Registry()
    result = asyncio.run(seed_prism_challenges(registry, settings=Settings()))
    assert result["agent-challenge"] == AGENT_CHALLENGE_INCOMPATIBLE_CODE
    # Prism may still be updated; agent-challenge must not be rewritten.
    assert result.get("prism") in {"created", "updated"}


def test_gateway_not_listed_as_direct_dependency() -> None:
    dist = importlib.metadata.distribution("base")
    requires = "\n".join(dist.requires or [])
    for banned in ("openai", "langchain", "anthropic", "tiktoken"):
        assert banned not in requires.lower()


def test_proxy_architecture_report_paths_return_not_found_without_upstream() -> None:
    """VAL-GATE-015: report paths not-found before challenge resolve / forward."""

    class Registry:
        def __init__(self) -> None:
            self.get_calls = 0
            self.list_calls = 0

        async def list(self):
            self.list_calls += 1
            return []

        async def get(self, slug: str):
            self.get_calls += 1
            raise KeyError(slug)

    class MinerVerifier:
        async def verify(self, *args, **kwargs):  # pragma: no cover
            raise AssertionError("unused")

    registry = Registry()
    app = create_proxy_app(
        registry=registry,
        miner_verifier=MinerVerifier(),  # type: ignore[arg-type]
    )
    client = TestClient(app)
    paths = (
        "/v1/architectures/arch-1/report",
        "/architectures/arch-1/report",
        "/challenges/prism/architectures/arch-1/report",
        "/challenges/prism/v1/architectures/arch-1/report",
        "/challenges/demo/architectures/missing/report",
    )
    for path in paths:
        response = client.get(path)
        assert response.status_code == 404, path
    # Short-circuit must not resolve the challenge or call registry.
    assert registry.get_calls == 0
    assert registry.list_calls == 0


def test_assignment_progress_and_result_reject_gateway_fields() -> None:
    """VAL-GATE-017: progress/result schemas 4xx on gateway_* and nested bags."""

    from pydantic import ValidationError

    from base.schemas.assignment import (
        AssignmentProgressRequest,
        AssignmentResultRequest,
    )

    with pytest.raises(ValidationError) as progress_exc:
        AssignmentProgressRequest.model_validate(
            {"gateway_token": "tok", "meta": {"ok": 1}}
        )
    assert "gateway" in str(progress_exc.value).lower()

    with pytest.raises(ValidationError) as progress_meta_exc:
        AssignmentProgressRequest.model_validate(
            {"meta": {"gateway_url": "http://example/llm"}}
        )
    assert "gateway" in str(progress_meta_exc.value).lower()

    with pytest.raises(ValidationError) as result_exc:
        AssignmentResultRequest.model_validate(
            {
                "success": True,
                "payload": {
                    "score": 1.0,
                    "gateway_token": "tok",
                    "provider": {"model": "gpt-4o", "api_key": "secret"},
                },
            }
        )
    body = str(result_exc.value).lower()
    assert "gateway" in body or "provider" in body

    with pytest.raises(ValidationError):
        AssignmentResultRequest.model_validate(
            {
                "success": True,
                "BASE_LLM_GATEWAY_URL": "http://example",
                "payload": {},
            }
        )

    # Clean payload still validates (including ExecutionProof-shaped provider metadata).
    clean = AssignmentResultRequest.model_validate(
        {
            "success": True,
            "payload": {
                "score": 0.5,
                "proof": {
                    "provider": {
                        "name": "lium",
                        "executor_id": "ex-1",
                        "pod_id": "pod-1",
                    }
                },
            },
        }
    )
    assert clean.success is True
    assert clean.payload["score"] == 0.5

    sdk_progress = __import__(
        "base.challenge_sdk.schemas", fromlist=["AssignmentProgressRequest"]
    ).AssignmentProgressRequest
    with pytest.raises(ValidationError):
        sdk_progress.model_validate({"meta": {"gateway_token": "x"}})
