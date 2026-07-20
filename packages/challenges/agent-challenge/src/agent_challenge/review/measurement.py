"""Measurement helpers for the independently deployed review application."""

from __future__ import annotations

from pathlib import Path

from agent_challenge.canonical.measurement import CanonicalMeasurement, build_canonical_measurement

from .compose import generate_review_app_compose, review_app_compose_hash


def build_review_measurement(
    *,
    review_image: str,
    app_identity: str,
    metadata_path: Path | str,
    cpu: int,
    memory: int | str,
    dstack_mr_bin: str | None = None,
) -> CanonicalMeasurement:
    """Reproduce the review image measurement over its own exact compose bytes."""

    compose = generate_review_app_compose(
        review_image=review_image,
        app_identity=app_identity,
    )
    record = build_canonical_measurement(
        metadata_path=metadata_path,
        cpu=cpu,
        memory=memory,
        compose=compose,
        dstack_mr_bin=dstack_mr_bin,
    )
    if record.compose_hash != review_app_compose_hash(compose):  # pragma: no cover - invariant
        raise RuntimeError("review compose hash drifted during measurement")
    return record


__all__ = ["build_review_measurement"]
