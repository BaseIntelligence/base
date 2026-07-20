from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import zipfile
from email.parser import BytesParser
from email.policy import default
from pathlib import Path

from base.challenge_sdk.app_factory import create_challenge_app
from base.challenge_sdk.auth import build_internal_auth_dependency
from base.challenge_sdk.config import ChallengeSettings
from base.challenge_sdk.executor import DockerExecutor
from base.challenge_sdk.proof import ExecutionProof
from base.challenge_sdk.roles import public_route
from base.challenge_sdk.schemas import HealthResponse, VersionResponse, WeightsResponse
from base.challenge_sdk.version import ARTIFACT_VERSION, RELEASE_MANIFEST


def _build_wheel(repository: Path, wheelhouse: Path, environment: dict[str, str]) -> None:
    uv = shutil.which("uv")
    if uv is not None:
        command = [uv, "build", "--wheel", "--out-dir", str(wheelhouse)]
    else:
        command = [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--wheel-dir",
            str(wheelhouse),
            str(repository),
        ]
    subprocess.run(
        command,
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )


def _base_requirement_from_wheel(wheel: Path) -> str:
    with zipfile.ZipFile(wheel) as archive:
        metadata_path = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        metadata = BytesParser(policy=default).parsebytes(archive.read(metadata_path))
    return next(
        requirement
        for requirement in metadata.get_all("Requires-Dist", [])
        if requirement.startswith("base @ ")
    )


def test_prism_uses_only_the_canonical_base_sdk() -> None:
    assert importlib.util.find_spec("prism_challenge.sdk") is None
    assert importlib.metadata.version("base") == ARTIFACT_VERSION
    assert RELEASE_MANIFEST.artifact_version == ARTIFACT_VERSION

    shared_symbols = (
        create_challenge_app,
        build_internal_auth_dependency,
        ChallengeSettings,
        DockerExecutor,
        ExecutionProof,
        public_route,
        HealthResponse,
        VersionResponse,
        WeightsResponse,
    )
    assert all(symbol.__module__.startswith("base.") for symbol in shared_symbols)


def test_prism_wheel_contains_no_vendored_sdk(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    clean_environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}

    _build_wheel(repository, wheelhouse, clean_environment)
    wheel = next(wheelhouse.glob("prism_challenge-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()

    assert not any(name.startswith("prism_challenge/sdk/") for name in names)


def test_prism_service_build_uses_only_the_immutable_base_wheel() -> None:
    repository = Path(__file__).resolve().parents[1]
    dockerfile = (repository / "Dockerfile").read_text(encoding="utf-8")
    assert "base @ https://github.com/BaseIntelligence/base/releases/download/v3.1.2/" in (
        repository / "pyproject.toml"
    ).read_text(encoding="utf-8")
    assert "git" not in dockerfile.lower()


def test_clean_artifacts_resolve_one_base_sdk(tmp_path: Path) -> None:
    prism_repository = Path(__file__).resolve().parents[1]
    wheelhouse = tmp_path / "wheelhouse"
    environment = tmp_path / "environment"
    wheelhouse.mkdir()
    clean_environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}

    _build_wheel(prism_repository, wheelhouse, clean_environment)
    prism_wheel = next(wheelhouse.glob("prism_challenge-*.whl"))
    base_requirement = _base_requirement_from_wheel(prism_wheel)

    subprocess.run(
        [sys.executable, "-m", "venv", str(environment)],
        check=True,
        capture_output=True,
        text=True,
        env=clean_environment,
    )
    python = environment / "bin" / "python"
    subprocess.run(
        [str(python), "-m", "pip", "install", base_requirement],
        check=True,
        capture_output=True,
        text=True,
        env=clean_environment,
    )
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            str(prism_wheel),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=clean_environment,
    )

    probe = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            (
                "import importlib.metadata,importlib.util,json;"
                "from base.challenge_sdk.version import RELEASE_MANIFEST;"
                "from prism_challenge.config import PrismSettings;"
                "print(json.dumps({"
                "'base_version':importlib.metadata.version('base'),"
                "'base_requirement':importlib.metadata.requires('prism-challenge'),"
                "'manifest':RELEASE_MANIFEST.model_dump(mode='json'),"
                "'prism_sdk':importlib.util.find_spec('prism_challenge.sdk'),"
                "'sdk_version':PrismSettings().sdk_version"
                "},sort_keys=True,default=str))"
            ),
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        env=clean_environment,
    )
    evidence = json.loads(probe.stdout)

    assert evidence["base_version"] == evidence["manifest"]["artifact_version"]
    assert evidence["sdk_version"] == evidence["manifest"]["sdk_contract_version"]
    assert evidence["prism_sdk"] is None
    assert any(
        requirement.startswith(
            "base @ https://github.com/BaseIntelligence/base/releases/download/v3.1.2/"
        )
        for requirement in evidence["base_requirement"]
    )
