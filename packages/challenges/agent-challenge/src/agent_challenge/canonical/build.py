"""Reproducible build harness + reproducibility guard for the canonical image.

The canonical eval image is built with BuildKit in reproducible mode: a
digest-pinned base image, ``SOURCE_DATE_EPOCH`` + ``rewrite-timestamp`` layer
normalisation, and provenance/SBOM attestations disabled. Under these inputs two
independent clean builds of the same source produce byte-identical image content
and therefore an identical image digest (VAL-IMG-001).

This module also provides the static build-definition checks (digest pinning /
no floating tags, locked+hashed dependencies) and the reproducibility guard that
builds twice and compares digests -- so an injected non-deterministic input is
detected as a digest mismatch rather than silently passing (VAL-IMG-003).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_DIR = REPO_ROOT / "docker" / "canonical"
CANONICAL_DOCKERFILE = CANONICAL_DIR / "Dockerfile"
CANONICAL_REQUIREMENTS = CANONICAL_DIR / "requirements.txt"

# Digest-pinned base image. Keep in sync with docker/canonical/Dockerfile.
CANONICAL_BASE_IMAGE = (
    "python:3.12-slim@sha256:423ed6ab25b1921a477529254bfeeabf5855151dc2c3141699a1bfc852199fbf"
)
# Fixed epoch so timestamps are normalised identically across builds.
DEFAULT_SOURCE_DATE_EPOCH = 1700000000

DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
DIGEST_PIN_RE = re.compile(r"@sha256:[0-9a-f]{64}")
_FROM_RE = re.compile(r"^\s*FROM\s+(?:--platform=\S+\s+)?(\S+)(?:\s+[Aa][Ss]\s+(\S+))?")
_ARG_RE = re.compile(r"^\s*ARG\s+([A-Za-z_][A-Za-z0-9_]*)=(.+?)\s*$")


# --------------------------------------------------------------------------- #
# Static build-definition analysis
# --------------------------------------------------------------------------- #


@dataclass
class BuildDefinitionReport:
    resolved_bases: list[str]
    stage_aliases: list[str] = field(default_factory=list)
    floating_tags: list[str] = field(default_factory=list)

    @property
    def digest_pinned(self) -> bool:
        return bool(self.resolved_bases) and not self.floating_tags


def _substitute_args(ref: str, args: dict[str, str]) -> str:
    def repl(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        return args.get(name, match.group(0))

    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)", repl, ref)


def validate_build_definition(dockerfile_text: str) -> BuildDefinitionReport:
    """Resolve every ``FROM`` base image and flag any non-digest-pinned ref."""

    args: dict[str, str] = {}
    stage_aliases: list[str] = []
    resolved_bases: list[str] = []

    for raw in dockerfile_text.splitlines():
        arg_match = _ARG_RE.match(raw)
        if arg_match:
            args[arg_match.group(1)] = arg_match.group(2).strip()
            continue
        from_match = _FROM_RE.match(raw)
        if not from_match:
            continue
        ref = _substitute_args(from_match.group(1), args)
        alias = from_match.group(2)
        # A FROM that references an earlier stage alias is not a base image.
        if ref not in stage_aliases:
            resolved_bases.append(ref)
        if alias:
            stage_aliases.append(alias)

    floating = [
        base for base in resolved_bases if not DIGEST_PIN_RE.search(base) and base != "scratch"
    ]
    return BuildDefinitionReport(
        resolved_bases=resolved_bases,
        stage_aliases=stage_aliases,
        floating_tags=floating,
    )


@dataclass(frozen=True)
class PinnedRequirement:
    name: str
    version: str | None
    hashes: tuple[str, ...]


def parse_requirements(text: str) -> list[PinnedRequirement]:
    """Parse a pip requirements file into pinned-requirement records."""

    # Join backslash line continuations so a multi-line ``--hash`` block is one entry.
    joined = re.sub(r"\\\s*\n", " ", text)
    requirements: list[PinnedRequirement] = []
    for raw in joined.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("--") and "==" not in line:
            continue
        spec = line.split()[0]
        version: str | None = None
        name = spec
        if "==" in spec:
            name, version = spec.split("==", 1)
        hashes = tuple(re.findall(r"--hash=([a-z0-9]+:[0-9a-f]+)", line))
        requirements.append(PinnedRequirement(name=name, version=version, hashes=hashes))
    return requirements


def requirements_are_hash_pinned(text: str) -> bool:
    parsed = parse_requirements(text)
    if not parsed:
        return False
    return all(req.version and req.hashes for req in parsed)


# --------------------------------------------------------------------------- #
# Docker availability
# --------------------------------------------------------------------------- #


def docker_available() -> bool:
    try:
        proc = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - env dependent
        return False
    return proc.returncode == 0


def buildx_available() -> bool:
    if not docker_available():  # pragma: no cover - env dependent
        return False
    try:
        proc = subprocess.run(
            ["docker", "buildx", "version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - env dependent
        return False
    return proc.returncode == 0


# Prefix for ephemeral builders created for OCI export (must be docker-container).
_EPHEMERAL_BUILDER_PREFIX = "ac-canonical-oci-"


@contextmanager
def ephemeral_docker_container_builder(
    *,
    name: str | None = None,
    runner: Callable[..., object] = subprocess.run,
) -> Iterator[str]:
    """Create a temporary ``docker-container`` buildx builder and tear it down.

    The default ``docker`` driver (GHA / stock Docker without the containerd image
    store) rejects the OCI exporter and ``rewrite-timestamp``. A
    ``docker-container`` driver hosts BuildKit in a container and supports both,
    which is required for the reproducible digest path (VAL-IMG-001 / 003).
    """

    builder_name = name or f"{_EPHEMERAL_BUILDER_PREFIX}{uuid.uuid4().hex[:12]}"
    create = runner(
        [
            "docker",
            "buildx",
            "create",
            "--name",
            builder_name,
            "--driver",
            "docker-container",
            "--bootstrap",
        ],
        capture_output=True,
        text=True,
    )
    if getattr(create, "returncode", 1) != 0:  # pragma: no cover - env dependent
        raise RuntimeError(
            "failed to create docker-container buildx builder "
            f"{builder_name!r} (exit {getattr(create, 'returncode', '?')}):\n"
            f"{getattr(create, 'stderr', '')}"
        )
    try:
        yield builder_name
    finally:
        runner(
            ["docker", "buildx", "rm", "-f", builder_name],
            capture_output=True,
            text=True,
        )


# --------------------------------------------------------------------------- #
# Reproducible build + guard
# --------------------------------------------------------------------------- #


@dataclass
class BuildResult:
    digest: str
    metadata: dict


@dataclass
class ReproCheck:
    digests: list[str]

    @property
    def reproducible(self) -> bool:
        return (
            len(self.digests) >= 2
            and len(set(self.digests)) == 1
            and all(DIGEST_RE.match(d) for d in self.digests)
        )


def build_image(
    *,
    context: Path | str | None = None,
    dockerfile: Path | str | None = None,
    source_date_epoch: int = DEFAULT_SOURCE_DATE_EPOCH,
    base_image: str | None = None,
    oci_dest: Path | str | None = None,
    load_tag: str | None = None,
    no_cache: bool = True,
    extra_build_args: dict[str, str] | None = None,
    builder: str | None = None,
) -> BuildResult:
    """Build the canonical image reproducibly and return its image digest.

    With ``load_tag`` the image is loaded into the local docker daemon (for
    functional inspection); otherwise it is exported as a reproducible OCI
    archive and the digest is read from the build metadata.

    OCI export + ``rewrite-timestamp`` require a ``docker-container`` buildx
    driver. When ``builder`` is omitted and the build uses the OCI exporter, an
    ephemeral ``docker-container`` builder is created for this call (and
    removed afterwards) so CI runners on the default ``docker`` driver succeed.
    """

    context = Path(context) if context is not None else REPO_ROOT
    dockerfile = Path(dockerfile) if dockerfile is not None else CANONICAL_DOCKERFILE

    needs_oci_builder = load_tag is None and builder is None

    def _run_build(active_builder: str | None) -> BuildResult:
        with tempfile.TemporaryDirectory() as tmp:
            meta_path = Path(tmp) / "metadata.json"
            cmd = [
                "docker",
                "buildx",
                "build",
                "--progress=plain",
            ]
            if active_builder:
                cmd += ["--builder", active_builder]
            cmd += [
                "-f",
                str(dockerfile),
                "--provenance=false",
                "--sbom=false",
                "--build-arg",
                f"SOURCE_DATE_EPOCH={source_date_epoch}",
                "--metadata-file",
                str(meta_path),
            ]
            if no_cache:
                cmd.append("--no-cache")
            if base_image:
                cmd += ["--build-arg", f"BASE_IMAGE={base_image}"]
            for key, value in (extra_build_args or {}).items():
                cmd += ["--build-arg", f"{key}={value}"]
            if load_tag:
                cmd += ["--load", "-t", load_tag]
            else:
                dest = Path(oci_dest) if oci_dest else Path(tmp) / "image.tar"
                cmd += ["--output", f"type=oci,dest={dest},rewrite-timestamp=true"]
            cmd.append(str(context))

            env = {**os.environ, "SOURCE_DATE_EPOCH": str(source_date_epoch)}
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if proc.returncode != 0:  # pragma: no cover - surfaced on build failure
                raise RuntimeError(
                    f"canonical image build failed (exit {proc.returncode}):\n{proc.stderr}"
                )
            metadata = json.loads(meta_path.read_text())

        digest = metadata.get("containerimage.digest", "")
        return BuildResult(digest=digest, metadata=metadata)

    if needs_oci_builder:
        with ephemeral_docker_container_builder() as ephemeral:
            return _run_build(ephemeral)
    return _run_build(builder)


def check_reproducible(
    *,
    builds: int = 2,
    dest_dir: Path | str | None = None,
    builder: str | None = None,
    **build_kwargs,
) -> ReproCheck:
    """Build the image ``builds`` times (clean) and collect the digests.

    Shares a single ephemeral ``docker-container`` buildx builder across the
    N builds when the caller did not supply ``builder``, so each build uses the
    OCI exporter without depending on the host daemon's default driver.
    """

    digests: list[str] = []

    def _run_all(active_builder: str | None) -> None:
        for index in range(builds):
            oci_dest = None
            if dest_dir is not None:
                oci_dest = Path(dest_dir) / f"image-{index}.tar"
            result = build_image(oci_dest=oci_dest, builder=active_builder, **build_kwargs)
            digests.append(result.digest)

    if builder is None:
        # Pass the shared ephemeral name so build_image does not create N builders.
        with ephemeral_docker_container_builder() as ephemeral:
            _run_all(ephemeral)
    else:
        _run_all(builder)
    return ReproCheck(digests=digests)


# --------------------------------------------------------------------------- #
# Build + push (publish a pullable, digest-pinned canonical image)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PushedImage:
    """A pushed image: its repository + the pushed manifest digest.

    :attr:`ref` is the pullable, digest-pinned ``repo@sha256:<64hex>`` the deploy
    path pins as the orchestrator image (never a mutable tag).
    """

    repository: str
    digest: str

    @property
    def ref(self) -> str:
        return f"{self.repository}@{self.digest}"


def repository_of(image_name: str) -> str:
    """Return the repository component of ``image_name`` (drop any ``:tag``/``@digest``).

    A registry host may contain a ``:port`` -- only a trailing tag on the final
    path component is stripped, so ``registry:5000/ns/repo:tag`` keeps its port.
    """

    name = image_name.split("@", 1)[0]
    head, sep, tail = name.rpartition("/")
    if ":" in tail:
        tail = tail.split(":", 1)[0]
    return f"{head}{sep}{tail}"


def assert_pullable(ref: str) -> str:
    """Return ``ref`` if it is a digest-pinned ``repo@sha256`` ref, else raise."""

    if not isinstance(ref, str) or not DIGEST_PIN_RE.search(ref):
        raise RuntimeError(f"pushed image ref is not digest-pinned: {ref!r}")
    return ref


def build_push_argv(
    *,
    image_name: str,
    dockerfile: Path | str,
    context: Path | str,
    metadata_path: Path | str,
    source_date_epoch: int = DEFAULT_SOURCE_DATE_EPOCH,
    base_image: str | None = None,
    builder: str | None = None,
    no_cache: bool = False,
) -> list[str]:
    """Construct the reproducible ``docker buildx build`` push command.

    Pushes to a registry (``type=image,...,push=true``) with layer-timestamp
    rewrite so the pushed image is reproducible, provenance/SBOM disabled, and the
    fixed ``SOURCE_DATE_EPOCH`` matching the OCI-export build path. A registry
    push with ``rewrite-timestamp`` requires a ``docker-container`` buildx driver
    (``--builder``); the default ``docker`` driver rejects it.
    """

    argv = ["docker", "buildx", "build", "--progress=plain"]
    if builder:
        argv += ["--builder", builder]
    argv += [
        "-f",
        str(dockerfile),
        "--provenance=false",
        "--sbom=false",
        "--build-arg",
        f"SOURCE_DATE_EPOCH={source_date_epoch}",
        "--metadata-file",
        str(metadata_path),
    ]
    if no_cache:
        argv.append("--no-cache")
    if base_image:
        argv += ["--build-arg", f"BASE_IMAGE={base_image}"]
    argv += [
        "--output",
        f"type=image,name={image_name},push=true,rewrite-timestamp=true",
        str(context),
    ]
    return argv


def build_and_push_image(
    *,
    image_name: str,
    context: Path | str | None = None,
    dockerfile: Path | str | None = None,
    source_date_epoch: int = DEFAULT_SOURCE_DATE_EPOCH,
    base_image: str | None = None,
    builder: str | None = None,
    no_cache: bool = False,
    runner: Callable[..., object] = subprocess.run,
) -> PushedImage:
    """Build the canonical image and push it, returning the pullable digest ref.

    The pushed manifest digest is read from the buildx ``--metadata-file``
    (``containerimage.digest``) and combined with the image repository to form the
    pullable ``repo@sha256`` ref. Raises :class:`RuntimeError` on a failed push or
    a metadata file that carries no digest.
    """

    context = Path(context) if context is not None else REPO_ROOT
    dockerfile = Path(dockerfile) if dockerfile is not None else CANONICAL_DOCKERFILE

    with tempfile.TemporaryDirectory() as tmp:
        meta_path = Path(tmp) / "metadata.json"
        argv = build_push_argv(
            image_name=image_name,
            dockerfile=dockerfile,
            context=context,
            metadata_path=meta_path,
            source_date_epoch=source_date_epoch,
            base_image=base_image,
            builder=builder,
            no_cache=no_cache,
        )
        env = {**os.environ, "SOURCE_DATE_EPOCH": str(source_date_epoch)}
        proc = runner(argv, capture_output=True, text=True, env=env)
        if getattr(proc, "returncode", 1) != 0:
            raise RuntimeError(
                f"canonical image push failed (exit {getattr(proc, 'returncode', '?')}):\n"
                f"{getattr(proc, 'stderr', '')}"
            )
        metadata = json.loads(meta_path.read_text())

    digest = metadata.get("containerimage.digest", "")
    if not DIGEST_RE.match(digest):
        raise RuntimeError(f"push metadata carried no valid containerimage.digest (got {digest!r})")
    return PushedImage(repository=repository_of(image_name), digest=digest)


def _main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI
    parser = argparse.ArgumentParser(prog="agent-challenge-canonical-build")
    parser.add_argument("--build", action="store_true", help="build once and print the digest")
    parser.add_argument(
        "--check-reproducible",
        action="store_true",
        help="build twice and report whether the digests match",
    )
    parser.add_argument("--builds", type=int, default=2)
    parser.add_argument(
        "--push",
        metavar="IMAGE_NAME",
        default=None,
        help="build+push to this registry ref (repo[:tag]) and print the pullable repo@sha256",
    )
    parser.add_argument(
        "--builder",
        default=None,
        help="buildx builder to use for --push (a docker-container driver is "
        "required for reproducible rewrite-timestamp registry pushes)",
    )
    args = parser.parse_args(argv)

    if args.push:
        pushed = build_and_push_image(image_name=args.push, builder=args.builder)
        print(pushed.ref)
        return 0

    if args.check_reproducible:
        result = check_reproducible(builds=args.builds)
        print(json.dumps({"digests": result.digests, "reproducible": result.reproducible}))
        return 0 if result.reproducible else 1

    result = build_image()
    print(result.digest)
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI shim
    raise SystemExit(_main())
