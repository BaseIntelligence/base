"""Thin package docs contract for Prism after shipping-docs collapse.

Shipping day-1 lives in monorepo-root ``docs/miner/getting-started.md``.
API truth is OpenAPI at ``chain.joinbase.ai/challenges/prism/openapi.json``.
Package ``docs/`` is a short pointer only; product pins stay on package README.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from prism_challenge.evaluator.interface import (
    ARCHITECTURE_FACTORY_NAME,
    DEFAULT_ARCHITECTURE_ENTRYPOINT,
    DEFAULT_TRAINING_ENTRYPOINT,
    TRAINING_ENTRYPOINT_NAME,
)
from prism_challenge.evaluator.schemas import RUN_MANIFEST_V2_FILENAME, ExecutionMode

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
DOCS_README = ROOT / "docs" / "README.md"
CONFIG_EXAMPLE = ROOT / "config.example.yaml"

# Large essay trees must not return under package docs/.
FORBIDDEN_ESSAY_PATHS = (
    "docs/overview.md",
    "docs/architecture.md",
    "docs/submissions.md",
    "docs/scoring.md",
    "docs/official-comparison.md",
    "docs/scaling.md",
    "docs/security.md",
    "docs/api.md",
    "docs/operators.md",
    "docs/miner/README.md",
    "docs/miner/getting-started.md",
    "docs/miner/concepts.md",
    "docs/miner/troubleshooting.md",
    "docs/validator/README.md",
)


def read_readme() -> str:
    return README.read_text(encoding="utf-8")


def test_package_docs_tree_is_pointer_only() -> None:
    assert DOCS_README.is_file()
    for rel in FORBIDDEN_ESSAY_PATHS:
        assert not (ROOT / rel).exists(), f"essay path must stay deleted: {rel}"

    # Only a short docs/README.md under docs/ (no nested essay dirs required).
    docs_files = sorted(p for p in (ROOT / "docs").rglob("*") if p.is_file())
    assert docs_files == [DOCS_README]

    pointer = DOCS_README.read_text(encoding="utf-8")
    assert "docs/miner/getting-started.md" in pointer
    assert "openapi.json" in pointer
    assert "chain.joinbase.ai/challenges/prism" in pointer
    assert len(pointer.splitlines()) <= 40


def test_readme_points_day1_and_openapi() -> None:
    readme = read_readme()
    assert "docs/miner/getting-started.md" in readme
    assert "https://chain.joinbase.ai/challenges/prism/openapi.json" in readme
    assert "https://chain.joinbase.ai/challenges/prism/docs" in readme
    assert "/challenges/prism" in readme
    assert "prism_challenge" in readme
    assert "ghcr.io/baseintelligence/prism" in readme


def test_readme_describes_the_v2_product() -> None:
    readme = read_readme()
    readme_lower = readme.lower()

    for expected in (
        "research lab",
        "new architecture",
        "two-script",
        "FineWeb-Edu",
        "prequential",
        "bits-per-byte",
        "held-out",
        "LLM gateway",
        "deterministic",
        "validator",
        "124",
        "350",
        "0.50",
    ):
        assert expected.lower() in readme_lower, f"README missing v2 concept: {expected}"
    assert "tiny-1m" in readme_lower or "transformer-tiny-1m" in readme_lower
    assert "mamba-tiny" in readme_lower
    assert DEFAULT_ARCHITECTURE_ENTRYPOINT in readme
    assert DEFAULT_TRAINING_ENTRYPOINT in readme
    assert f"{ARCHITECTURE_FACTORY_NAME}(ctx)" in readme
    assert f"{TRAINING_ENTRYPOINT_NAME}(ctx)" in readme


def test_no_tee_provider_trust_pins_on_readme_and_config() -> None:
    readme = read_readme().lower()
    config_example = CONFIG_EXAMPLE.read_text(encoding="utf-8")

    assert "provider" in readme and ("trust" in readme or "image_pin" in readme or "image pin" in readme)
    assert "lium" in readme and "targon" in readme
    assert "pinned_image_digest" in config_example
    assert "\ntee:" not in config_example
    assert not any(line.strip().startswith("tee:") for line in config_example.splitlines())
    assert "tee verifier" in readme or "no production tee" in readme or "image_pin" in readme


def test_execution_modes_match_code_enum() -> None:
    # Modes remain live code; docs essays no longer enumerate them.
    values = {mode.value for mode in ExecutionMode}
    assert "local_cpu_smoke" not in values
    assert RUN_MANIFEST_V2_FILENAME == "prism_run_manifest.v2.json"


def test_architecture_lab_routes_replace_dead_nas_routes(client: TestClient) -> None:
    """Serving layer contract: architectures live; flat training-variants stay gone."""

    architectures = client.get("/v1/architectures")
    assert architectures.status_code == 200, architectures.text
    body = architectures.json()
    assert set(body.keys()) == {"epoch_id", "architectures"}
    assert isinstance(body["architectures"], list)

    assert client.get("/v1/training-variants").status_code == 404

    nested = client.get("/v1/architectures/does-not-exist/variants")
    assert nested.status_code == 404, nested.text
    assert nested.json()["detail"] == "architecture not found"
