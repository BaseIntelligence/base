from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path


def _run(*argv: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env={
            **{key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
            "SOURCE_DATE_EPOCH": "1704067200",
        },
    )


def test_wheel_installs_canonical_sdk_without_source_checkout(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    wheelhouse = tmp_path / "wheelhouse"
    rebuild_wheelhouse = tmp_path / "rebuild-wheelhouse"
    environments = (tmp_path / "environment-a", tmp_path / "environment-b")
    wheelhouse.mkdir()
    rebuild_wheelhouse.mkdir()

    _run("uv", "build", "--wheel", "--out-dir", str(wheelhouse), cwd=repository)
    _run(
        "uv",
        "build",
        "--wheel",
        "--out-dir",
        str(rebuild_wheelhouse),
        cwd=repository,
    )
    wheel = next(wheelhouse.glob("base-*.whl"))
    rebuilt_wheel = next(rebuild_wheelhouse.glob("base-*.whl"))
    wheel_digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert hashlib.sha256(rebuilt_wheel.read_bytes()).hexdigest() == wheel_digest

    inventories: list[str] = []
    for environment in environments:
        _run("uv", "venv", "--python", "3.12", str(environment))
        python = environment / "bin" / "python"
        _run("uv", "pip", "install", "--python", str(python), str(wheel))
        inventories.append(_run("uv", "pip", "freeze", "--python", str(python)).stdout)
    assert inventories[0] == inventories[1]

    environment = environments[0]
    python = environment / "bin" / "python"
    probe = _run(
        str(python),
        "-I",
        "-c",
        (
            "import importlib.metadata,json;"
            "from base.challenge_sdk.version import RELEASE_MANIFEST;"
            "import base.challenge_sdk as sdk;"
            "print(json.dumps({"
            "'distribution_version':importlib.metadata.version('base'),"
            "'module_path':sdk.__file__,"
            "'manifest':RELEASE_MANIFEST.model_dump(mode='json')"
            "},sort_keys=True))"
        ),
        cwd=tmp_path,
    )
    evidence = json.loads(probe.stdout)

    assert evidence["distribution_version"] == evidence["manifest"]["artifact_version"]
    assert evidence["manifest"]["distribution_name"] == "base"
    assert evidence["manifest"]["release_id"] == (
        f"v{evidence['distribution_version']}"
    )
    assert evidence["module_path"].startswith(str(environment))
    assert str(repository) not in evidence["module_path"]
    assert len(wheel_digest) == 64
