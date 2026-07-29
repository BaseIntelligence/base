"""Digest-manifest default path: precedence + installed-layout + wheel membership.

The default must resolve to an existing ``dataset-digest.json`` in BOTH the
source checkout and an installed wheel. A fixed ``Path(__file__).parents[N]``
walk is layout-dependent and breaks under site-packages.
"""

from __future__ import annotations

import ast
import inspect
import os
import subprocess
import textwrap
import zipfile
from pathlib import Path

import pytest

from agent_challenge.evaluation.own_runner_backend import (
    DIGEST_MANIFEST_ENV,
    _resolve_manifest_path,
)

_PKG_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_DIGEST = _PKG_ROOT / "golden" / "dataset-digest.json"
_WHEEL_MEMBER_SUFFIX = "agent_challenge/golden/dataset-digest.json"


def test_resolve_manifest_path_explicit_arg_wins_over_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given: explicit path and env both set.
    When: resolve with the explicit path.
    Then: returns the explicit path (not the env value).
    """
    explicit = tmp_path / "explicit-digest.json"
    explicit.write_text("{}", encoding="utf-8")
    env_path = tmp_path / "env-digest.json"
    env_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(DIGEST_MANIFEST_ENV, str(env_path))

    resolved = _resolve_manifest_path(explicit)

    assert resolved == explicit


def test_resolve_manifest_path_env_wins_over_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given: env set, no explicit path.
    When: resolve with None.
    Then: returns the env path (not the packaged/repo default).
    """
    env_path = tmp_path / "env-only-digest.json"
    env_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(DIGEST_MANIFEST_ENV, str(env_path))

    resolved = _resolve_manifest_path(None)

    assert resolved == env_path


def test_resolve_manifest_path_default_is_existing_file_with_env_cleared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given: no explicit path and env unset.
    When: resolve default.
    Then: path exists on disk (source checkout or packaged resource).

    A source-layout-only ``parents[3]`` check is NOT enough — that passes today
    and would not have caught the installed-wheel bug.
    """
    monkeypatch.delenv(DIGEST_MANIFEST_ENV, raising=False)

    resolved = _resolve_manifest_path(None)

    assert resolved.is_file(), f"default digest manifest missing: {resolved}"
    assert resolved.name == "dataset-digest.json"


def test_resolve_manifest_path_default_not_fixed_parents3_only() -> None:
    """Given: the resolver source.
    When: inspected for the default branch strategy.
    Then: must not rely solely on a bare ``parents[3]`` index (installed layout
    drops the ``src/`` segment and overshoots into site-packages' grandparent).
    """
    source = textwrap.dedent(inspect.getsource(_resolve_manifest_path))
    # Allow parents walks that search multiple ancestors; forbid the sole
    # fixed-index default that caused the production bug.
    tree = ast.parse(source)
    fixed_parents3_only = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        # Match ``.parents[3]`` (Constant 3 or legacy Num)
        slice_val: object | None = None
        if isinstance(node.slice, ast.Constant):
            slice_val = node.slice.value
        if slice_val != 3:
            continue
        # value should be Attribute named parents
        if isinstance(node.value, ast.Attribute) and node.value.attr == "parents":
            fixed_parents3_only = True
            break

    assert not fixed_parents3_only, (
        "_resolve_manifest_path still uses a fixed parents[3] default; "
        "use importlib.resources and/or a multi-parent existence walk"
    )


def test_wheel_includes_dataset_digest_json(tmp_path: Path) -> None:
    """Given: a hatch wheel build of agent-challenge.
    When: the wheel namelist is inspected.
    Then: member ends with agent_challenge/golden/dataset-digest.json.
    """
    out_dir = tmp_path / "wheel-out"
    out_dir.mkdir()
    repo_root = _PKG_ROOT.parents[2]  # packages/challenges/agent-challenge -> monorepo root
    # Prefer monorepo root when present; fall back to package dir for isolated builds.
    if not (repo_root / "pyproject.toml").is_file():
        repo_root = _PKG_ROOT
    env = {**os.environ, "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR", "/var/tmp/uv-cache")}
    proc = subprocess.run(
        [
            "uv",
            "build",
            "--package",
            "agent-challenge",
            "--wheel",
            "--out-dir",
            str(out_dir),
        ],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"uv build failed:\n{proc.stdout}\n{proc.stderr}"
    wheels = list(out_dir.glob("*.whl"))
    assert len(wheels) == 1, wheels
    names = zipfile.ZipFile(wheels[0]).namelist()
    hits = [n for n in names if n.endswith(_WHEEL_MEMBER_SUFFIX)]
    assert hits, (
        f"wheel missing {_WHEEL_MEMBER_SUFFIX}; sample members: {names[:30]}"
    )


def test_default_digest_not_site_packages_grandparent_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given: env cleared.
    When: default is resolved in the current layout.
    Then: path must not look like the broken installed walk
    ``.../pythonX.Y/golden/dataset-digest.json`` (parents[3] overshoot).
    """
    monkeypatch.delenv(DIGEST_MANIFEST_ENV, raising=False)

    resolved = _resolve_manifest_path(None)
    parts = resolved.parts
    # Broken shape: .../lib/python3.12/golden/dataset-digest.json
    if len(parts) >= 3 and parts[-2] == "golden" and parts[-1] == "dataset-digest.json":
        parent_name = parts[-3]
        assert not parent_name.startswith("python"), (
            f"default resolved to site-packages overshoot path: {resolved}"
        )
    assert resolved.is_file()



def test_package_dockerfile_copies_force_include_digest_before_pip_install() -> None:
    """hatch force-include needs golden/dataset-digest.json in the image build context.

    CI failed with:
      FileNotFoundError: Forced include not found: /app/golden/dataset-digest.json
    when Dockerfile only COPYed src/ + pyproject and then ran ``pip install .``.
    Both runtime and terminal-bench-runner stages must COPY the file first.
    """
    dockerfile = (_PKG_ROOT / "Dockerfile").read_text(encoding="utf-8")
    pyproject = (_PKG_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"golden/dataset-digest.json"' in pyproject or (
        "golden/dataset-digest.json" in pyproject
    ), "pyproject must force-include golden/dataset-digest.json"
    # Exact path used by hatch force-include source key.
    copy_line = "COPY golden/dataset-digest.json"
    assert copy_line in dockerfile, (
        "Dockerfile must COPY golden/dataset-digest.json so hatch force-include "
        "finds /app/golden/dataset-digest.json during pip install ."
    )
    assert dockerfile.count(copy_line) >= 2, (
        "both runtime and terminal-bench-runner stages need the golden COPY "
        f"(found {dockerfile.count(copy_line)})"
    )
    # Each pip install of the package must be preceded by the golden COPY in-file order.
    install_marker = "RUN pip install --no-cache-dir ."
    positions = []
    start = 0
    while True:
        idx = dockerfile.find(install_marker, start)
        if idx < 0:
            break
        positions.append(idx)
        start = idx + len(install_marker)
    assert positions, "expected at least one package pip install in Dockerfile"
    for idx in positions:
        preceding = dockerfile[:idx]
        assert copy_line in preceding, (
            "COPY golden/dataset-digest.json must appear before each "
            "RUN pip install --no-cache-dir ."
        )
