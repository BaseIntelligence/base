#!/usr/bin/env python3
"""Regenerate docker/canonical/requirements.txt from uv.lock.

The canonical eval image installs a small, locked, hashed dependency set: the
own_runner runtime closure (root: ``pydantic``), plus ``cryptography`` (used by
``agent_challenge.golden.crypto`` to decrypt the golden in-enclave) and
``dstack-sdk`` (used to emit the TDX quote / cc-eventlog in the CVM). This script
walks the transitive closure of those roots from uv.lock -- honouring dependency
environment markers for the image's target platform -- and pins every package to
its exact version + wheel hashes, so the requirements file never drifts from the
lockfile and the image installs with ``pip install --require-hashes``.

Target platform: linux/amd64, CPython 3.12 (the canonical image base). Every
wheel hash for each package is emitted (as ``pip-compile --generate-hashes``
does) so ``pip`` selects the best-compatible wheel for the base image and matches
its hash against the pinned set; sdist hashes are intentionally omitted so a
source build (non-reproducible) can never be selected.

Regenerate after any lock change with:
    python scripts/gen_canonical_requirements.py
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.markers import Marker

REPO_ROOT = Path(__file__).resolve().parent.parent
UV_LOCK = REPO_ROOT / "uv.lock"
OUTPUT = REPO_ROOT / "docker" / "canonical" / "requirements.txt"

# Runtime roots the canonical image imports. Their full lockfile closure is
# pinned below.
#   * pydantic     -- own_runner result schema (always imported).
#   * cryptography -- in-enclave AES-256-GCM golden decrypt (golden.crypto).
#   * dstack-sdk   -- in-CVM TDX quote + cc-eventlog emission.
ROOTS = (
    "pydantic",
    "cryptography",
    "dstack-sdk",
)

# Marker environment for the image target (linux/amd64, CPython 3.12). Fixed so
# the resolved closure is deterministic regardless of the host running this
# script.
TARGET_MARKER_ENV = {
    "implementation_name": "cpython",
    "platform_python_implementation": "CPython",
    "python_full_version": "3.12.10",
    "python_version": "3.12",
    "sys_platform": "linux",
    "platform_system": "Linux",
    "os_name": "posix",
    "platform_machine": "x86_64",
}

HEADER = """\
# Locked, hashed runtime dependencies for the canonical eval image.
#
# Roots: pydantic (own_runner result schema), cryptography (in-enclave golden
# decrypt) and dstack-sdk (in-CVM TDX quote + cc-eventlog). This pins each root
# and its full dependency closure to exact versions + every wheel hash, so the
# image build is reproducible and every dependency is immutable. Installed with
# `pip install --require-hashes`.
#
# Generated from uv.lock (single source of truth). Regenerate after any lock
# change with: python scripts/gen_canonical_requirements.py
"""


def _marker_satisfied(raw_marker: str | None) -> bool:
    """Whether a dependency edge's marker holds for the image target."""

    if not raw_marker:
        return True
    return bool(Marker(raw_marker).evaluate(TARGET_MARKER_ENV))


def _resolve_closure(by_name: dict[str, dict]) -> set[str]:
    """Transitive closure of :data:`ROOTS`, honouring dependency markers."""

    closure: set[str] = set()
    stack = list(ROOTS)
    while stack:
        name = stack.pop()
        if name in closure:
            continue
        package = by_name.get(name)
        if package is None:
            raise SystemExit(f"{name} not found in uv.lock")
        closure.add(name)
        for dep in package.get("dependencies", []):
            if _marker_satisfied(dep.get("marker")):
                stack.append(dep["name"])
    return closure


def _wheel_hashes(package: dict) -> list[str]:
    """Every wheel hash for ``package`` (sorted, deterministic)."""

    hashes = sorted({wheel["hash"] for wheel in package.get("wheels", []) if wheel.get("hash")})
    if not hashes:
        raise SystemExit(f"no wheel hashes for {package['name']} in uv.lock")
    return hashes


def render(by_name: dict[str, dict]) -> str:
    closure = _resolve_closure(by_name)
    lines = [HEADER.rstrip("\n")]
    for name in sorted(closure):
        package = by_name[name]
        hashes = _wheel_hashes(package)
        entry = [f"{name}=={package['version']} \\"]
        for index, digest in enumerate(hashes):
            suffix = " \\" if index < len(hashes) - 1 else ""
            entry.append(f"    --hash={digest}{suffix}")
        lines.append("\n".join(entry))
    return "\n".join(lines) + "\n"


def main() -> int:
    data = tomllib.loads(UV_LOCK.read_text())
    by_name = {pkg["name"]: pkg for pkg in data["package"]}
    closure = _resolve_closure(by_name)
    OUTPUT.write_text(render(by_name))
    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)} ({len(closure)} packages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
