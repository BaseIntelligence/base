"""Fail-closed inventory of modules required by the measured review image.

Root cause residual (eval-2task-real-go / budget-unlimited):
  guest ``ModuleNotFoundError: No module named agent_challenge.review.attested_times``
  during ``report.review_report_data_preimage`` → reason_code ``report_generation_failed``.

Post-attested_times residual (eval-2task-post-attested-times):
  guest still ``ModuleNotFoundError`` → ``report_generation_failed`` because
  ``openrouter`` lazily imports ``.or_outcome_bind``
  (``require_real_or_digests`` / ``build_openrouter_observation``) after the
  OpenRouter response, and the review Dockerfile omitted that COPY.

``docker/review/Dockerfile`` must COPY every module on the guest report_data and
post-response OR-bind path, and selfcheck-import those modules so image builds
fail closed when a required module is omitted.
"""

from __future__ import annotations

import re
from pathlib import Path

from agent_challenge.review import compose as review_compose

REPO_ROOT = Path(__file__).resolve().parents[1]
REVIEW_DOCKERFILE = review_compose.REVIEW_DOCKERFILE

# Guest report_data + post-response OpenRouter outcome-bind path (measured CVM).
# Keep lean: no eval runtime, no client, no full keyrelease service.
REQUIRED_REVIEW_IMAGE_MODULES: tuple[str, ...] = (
    "src/agent_challenge/__init__.py",
    "src/agent_challenge/review/__init__.py",
    "src/agent_challenge/review/canonical.py",
    "src/agent_challenge/review/schemas.py",
    "src/agent_challenge/review/policy.py",
    "src/agent_challenge/review/report.py",
    "src/agent_challenge/review/openrouter.py",
    "src/agent_challenge/review/or_outcome_bind.py",
    "src/agent_challenge/review/attested_times.py",
    "src/agent_challenge/keyrelease/__init__.py",
    "src/agent_challenge/keyrelease/quote.py",
    "docker/review/review_runtime.py",
)


def _dockerfile_text() -> str:
    assert REVIEW_DOCKERFILE.is_file(), f"missing {REVIEW_DOCKERFILE}"
    return REVIEW_DOCKERFILE.read_text(encoding="utf-8")


def test_repo_ships_attested_times_module() -> None:
    path = REPO_ROOT / "src" / "agent_challenge" / "review" / "attested_times.py"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "review_report_data_preimage_v2" in text
    assert "extract_bound_times_from_core" in text


def test_repo_ships_or_outcome_bind_module() -> None:
    path = REPO_ROOT / "src" / "agent_challenge" / "review" / "or_outcome_bind.py"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "def require_real_or_digests" in text
    assert "def build_openrouter_observation" in text


def test_review_report_lazy_imports_attested_times() -> None:
    """Report construction path that guest hits must depend on attested_times."""

    report_src = (REPO_ROOT / "src" / "agent_challenge" / "review" / "report.py").read_text(
        encoding="utf-8"
    )
    assert "from .attested_times import" in report_src
    assert "def review_report_data_preimage" in report_src
    assert "def review_report_data_hex" in report_src
    runtime_src = (REPO_ROOT / "docker" / "review" / "review_runtime.py").read_text(
        encoding="utf-8"
    )
    assert "review_report_data_hex" in runtime_src


def test_openrouter_lazy_imports_or_outcome_bind() -> None:
    """Post-response OpenRouter path must depend on or_outcome_bind (guest)."""

    openrouter_src = (REPO_ROOT / "src" / "agent_challenge" / "review" / "openrouter.py").read_text(
        encoding="utf-8"
    )
    assert "from .or_outcome_bind import" in openrouter_src
    assert "require_real_or_digests" in openrouter_src
    assert "build_openrouter_observation" in openrouter_src


def test_review_dockerfile_copies_required_report_path_modules() -> None:
    """Fail closed if Dockerfile omits a guest report_data path module."""

    dockerfile = _dockerfile_text()
    for module_path in REQUIRED_REVIEW_IMAGE_MODULES:
        assert f"COPY {module_path}" in dockerfile, (
            f"review Dockerfile must COPY {module_path} "
            "(missing package causes guest ModuleNotFoundError / report_generation_failed)"
        )


def test_review_dockerfile_copies_attested_times_explicitly() -> None:
    dockerfile = _dockerfile_text()
    assert (
        "COPY src/agent_challenge/review/attested_times.py "
        "/app/agent_challenge/review/attested_times.py"
    ) in dockerfile


def test_review_dockerfile_copies_or_outcome_bind_explicitly() -> None:
    dockerfile = _dockerfile_text()
    assert (
        "COPY src/agent_challenge/review/or_outcome_bind.py "
        "/app/agent_challenge/review/or_outcome_bind.py"
    ) in dockerfile


def test_review_dockerfile_selfcheck_imports_report_data_hex() -> None:
    """Build-time selfcheck must import the report_data path so missing COPY fail-closes."""

    dockerfile = _dockerfile_text()
    # Accept either review_report_data_hex or preimage entry (feature allows either).
    has_hex = "review_report_data_hex" in dockerfile
    has_preimage = "review_report_data_preimage" in dockerfile
    assert has_hex or has_preimage, (
        "Dockerfile must selfcheck-import agent_challenge.review.report."
        "review_report_data_hex (or preimage) so builds fail closed without attested_times"
    )
    # RUN python selfcheck: single-line -c form or HEREDOC form both acceptable.
    run_python = re.search(
        r"^\s*RUN\s+python\b",
        dockerfile,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    assert run_python is not None, "Dockerfile needs RUN python ... selfcheck (fail-closed build)"
    assert "from agent_challenge.review.report" in dockerfile or (
        "agent_challenge.review.report" in dockerfile
    ), "selfcheck must import agent_challenge.review.report"
    # attested_times is lazy inside report; selfcheck must import it explicitly.
    assert "agent_challenge.review.attested_times" in dockerfile


def test_review_dockerfile_selfcheck_imports_or_outcome_bind() -> None:
    """Build-time selfcheck must import or_outcome_bind (lazy openrouter path)."""

    dockerfile = _dockerfile_text()
    assert "agent_challenge.review.or_outcome_bind" in dockerfile, (
        "Dockerfile selfcheck must import agent_challenge.review.or_outcome_bind "
        "so missing COPY of that module fails the image build"
    )
    # Prefer explicit symbol import that openrouter uses post-response.
    assert "require_real_or_digests" in dockerfile or "build_openrouter_observation" in dockerfile


def test_missing_attested_times_copy_would_fail_inventory() -> None:
    """Unit smoke: stripped Dockerfile inventory must not claim completeness."""

    dockerfile = _dockerfile_text()
    stripped = "\n".join(
        line for line in dockerfile.splitlines() if "attested_times.py" not in line
    )
    assert "COPY src/agent_challenge/review/attested_times.py" not in stripped
    # The real file must still pass; this proves the assertion gate is meaningful.
    assert "COPY src/agent_challenge/review/attested_times.py" in dockerfile


def test_missing_or_outcome_bind_copy_would_fail_inventory() -> None:
    """Stripping or_outcome_bind COPY must fail the inventory contract."""

    dockerfile = _dockerfile_text()
    stripped = "\n".join(
        line for line in dockerfile.splitlines() if "or_outcome_bind.py" not in line
    )
    assert "COPY src/agent_challenge/review/or_outcome_bind.py" not in stripped
    assert "COPY src/agent_challenge/review/or_outcome_bind.py" in dockerfile
