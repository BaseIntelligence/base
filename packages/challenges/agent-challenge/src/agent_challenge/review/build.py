"""Reproducible build helpers for the separate review image."""

from __future__ import annotations

from pathlib import Path

from agent_challenge.canonical import build as canonical_build

from .compose import (
    REVIEW_DOCKERFILE,
    REVIEW_REQUIREMENTS,
    ReviewBuildDefinition,
    review_build_definition,
)


def validate_review_build_definition() -> canonical_build.BuildDefinitionReport:
    """Validate that the review image uses only digest-pinned base images."""

    definition = review_build_definition()
    if not definition.dockerfile.is_file() or not definition.requirements.is_file():
        raise RuntimeError("review image build inputs are missing")
    report = canonical_build.validate_build_definition(
        definition.dockerfile.read_text(encoding="utf-8")
    )
    if not report.digest_pinned:
        raise RuntimeError("review image has a floating build input")
    if not canonical_build.requirements_are_hash_pinned(
        definition.requirements.read_text(encoding="utf-8")
    ):
        raise RuntimeError("review image dependencies are not fully hash-pinned")
    return report


def build_review_image(
    *,
    context: Path | str | None = None,
    **kwargs: object,
) -> canonical_build.BuildResult:
    """Build the review image with the shared reproducible BuildKit recipe."""

    validate_review_build_definition()
    return canonical_build.build_image(
        context=context,
        dockerfile=REVIEW_DOCKERFILE,
        **kwargs,
    )


def check_review_reproducible(
    *,
    builds: int = 2,
    context: Path | str | None = None,
    **kwargs: object,
) -> canonical_build.ReproCheck:
    """Build review image independently twice and compare immutable digests."""

    validate_review_build_definition()
    return canonical_build.check_reproducible(
        builds=builds,
        context=context,
        dockerfile=REVIEW_DOCKERFILE,
        **kwargs,
    )


__all__ = [
    "REVIEW_DOCKERFILE",
    "REVIEW_REQUIREMENTS",
    "ReviewBuildDefinition",
    "build_review_image",
    "check_review_reproducible",
    "validate_review_build_definition",
]
