"""Host landmine: master-entrypoint Prism env defaults for worker-plane policy.

PROD POLICY baked into docker/master-entrypoint.sh:
- CPU_REEXEC_TEST_MODE defaults false (eval never on master)
- ADMISSION_REQUIRES_WORKER defaults false
- plagiarism LLM defaults off until secrets present
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = REPO_ROOT / "docker/master-entrypoint.sh"


def test_entrypoint_embeds_worker_plane_prod_defaults() -> None:
    """Given entrypoint script, When read, Then worker-plane prod defaults present."""
    text = ENTRYPOINT.read_text(encoding="utf-8")
    assert (
        "PRISM_WORKER_PLANE__CPU_REEXEC_TEST_MODE=${PRISM_WORKER_PLANE__CPU_REEXEC_TEST_MODE:-false}"
        in text
    )
    assert (
        "PRISM_WORKER_PLANE__ADMISSION_REQUIRES_WORKER="
        "${PRISM_WORKER_PLANE__ADMISSION_REQUIRES_WORKER:-false}"
    ) in text
    assert "PRISM_WORKER_PLANE__MASTER_BASE_URL=" in text
    assert "PRISM_PLAGIARISM_LLM_ENABLED=${PRISM_PLAGIARISM_LLM_ENABLED:-false}" in text
    assert "PRISM_OPENROUTER_MODEL=${PRISM_OPENROUTER_MODEL:-x-ai/grok-4.5}" in text
    assert "PRISM_CONSTATION_BASE_URL=" in text
    # Optional token only when parent set it
    assert "PRISM_CONSTATION_INTERNAL_TOKEN" in text
    assert "CPU_REEXEC must stay false" in text or "admission_requires_worker" in text
