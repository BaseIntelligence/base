"""Tests for the canonical image build+push helper (returns a pullable ref).

The helper builds the reproducible canonical image and pushes it to a registry,
returning the pullable ``repo@sha256`` ref the deploy path pins as the
orchestrator image (never a bare/floating tag). These tests pin the pushed-ref
assembly and the reproducible buildx invocation without touching a real daemon.
"""

from __future__ import annotations

import json

import pytest

from agent_challenge.canonical import build as cbuild


def test_repository_of_strips_tag_keeps_namespace():
    assert (
        cbuild.repository_of("docker.io/mathiiss/agent-challenge-canonical:live")
        == "docker.io/mathiiss/agent-challenge-canonical"
    )
    # No tag -> unchanged.
    assert (
        cbuild.repository_of("docker.io/mathiiss/agent-challenge-canonical")
        == "docker.io/mathiiss/agent-challenge-canonical"
    )
    # A digest ref -> the repository component only.
    assert cbuild.repository_of("docker.io/mathiiss/x@sha256:" + "a" * 64) == "docker.io/mathiiss/x"


def test_pushed_image_ref_is_repo_at_digest():
    pushed = cbuild.PushedImage(
        repository="docker.io/mathiiss/agent-challenge-canonical",
        digest="sha256:" + "a" * 64,
    )
    assert pushed.ref == "docker.io/mathiiss/agent-challenge-canonical@sha256:" + "a" * 64
    # It is digest-pinned per the compose guard (no bare tag).
    assert cbuild.DIGEST_PIN_RE.search(pushed.ref)


def test_build_push_argv_is_reproducible_registry_push():
    argv = cbuild.build_push_argv(
        image_name="docker.io/mathiiss/agent-challenge-canonical:live",
        dockerfile="/repo/docker/canonical/Dockerfile",
        context="/repo",
        metadata_path="/tmp/meta.json",
        source_date_epoch=1700000000,
        builder="repro-builder",
    )
    joined = " ".join(argv)
    assert "buildx" in argv and "build" in argv
    assert "--builder" in argv and "repro-builder" in argv
    assert "--provenance=false" in argv and "--sbom=false" in argv
    assert "SOURCE_DATE_EPOCH=1700000000" in joined
    # Registry push with reproducible layer-timestamp rewrite.
    out = argv[argv.index("--output") + 1]
    assert "type=image" in out
    assert "name=docker.io/mathiiss/agent-challenge-canonical:live" in out
    assert "push=true" in out
    assert "rewrite-timestamp=true" in out


def test_build_and_push_image_returns_pullable_digest(tmp_path):
    digest = "sha256:" + "c" * 64

    def fake_runner(argv, **kwargs):
        # Emulate buildx writing the metadata file with the pushed digest.
        meta_idx = argv.index("--metadata-file")
        meta_path = argv[meta_idx + 1]
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(
                {"containerimage.digest": digest, "image.name": "docker.io/mathiiss/x:live"},
                fh,
            )

        class _P:
            returncode = 0
            stderr = ""
            stdout = ""

        return _P()

    pushed = cbuild.build_and_push_image(
        image_name="docker.io/mathiiss/x:live",
        runner=fake_runner,
        context=str(tmp_path),
        dockerfile=str(cbuild.CANONICAL_DOCKERFILE),
    )
    assert pushed.ref == "docker.io/mathiiss/x@" + digest
    assert cbuild.assert_pullable(pushed.ref) == pushed.ref


def test_build_and_push_image_raises_on_missing_digest(tmp_path):
    def fake_runner(argv, **kwargs):
        meta_idx = argv.index("--metadata-file")
        with open(argv[meta_idx + 1], "w", encoding="utf-8") as fh:
            json.dump({}, fh)  # no digest

        class _P:
            returncode = 0
            stderr = ""
            stdout = ""

        return _P()

    with pytest.raises(RuntimeError):
        cbuild.build_and_push_image(
            image_name="docker.io/mathiiss/x:live",
            runner=fake_runner,
            context=str(tmp_path),
        )


def test_build_and_push_image_raises_on_build_failure(tmp_path):
    def fake_runner(argv, **kwargs):
        class _P:
            returncode = 1
            stderr = "boom"
            stdout = ""

        return _P()

    with pytest.raises(RuntimeError, match="push"):
        cbuild.build_and_push_image(
            image_name="docker.io/mathiiss/x:live",
            runner=fake_runner,
            context=str(tmp_path),
        )
