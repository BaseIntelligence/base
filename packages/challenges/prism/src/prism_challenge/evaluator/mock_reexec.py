"""Mockable CPU re-execution of the prism validator harness (architecture.md sections 7, 10).

There is no GPU (and no live broker) in the test/CI environment, so the GPU re-execution the
validator dispatches to its OWN broker is exercised here by running the SAME challenge-owned
``runner.py`` in a single local CPU process (``nproc=1``, deterministic). This drives the real
forced-random-init + online-loss capture + prequential-bpb authoring and persists a real
``prism_run_manifest.v2`` + ``trained_state.pt`` WITHOUT a GPU, so the scoring/anti-cheat path is
validated end-to-end. The real GPU broker run is wired at deploy.

The mock is a drop-in for ``DockerExecutor.run``: tests monkeypatch
``prism_challenge.evaluator.container.DockerExecutor.run`` with :func:`cpu_reexec_run` so the
production ``PrismContainerEvaluator.evaluate`` path is unchanged. Before running, the mock asserts
the dispatched :class:`DockerRunSpec` is network-isolated (``network=none``) and mounts ONLY the
workspace + writable artifacts dir (never the secret val/test split), mirroring the deploy posture
(VAL-PRISM-013/014/028/039).
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, MutableSequence
from pathlib import Path

from base.challenge_sdk.executor import DockerRunResult, DockerRunSpec

WORKSPACE_TARGET = "/workspace"
ARTIFACTS_TARGET = "/artifacts"
# The single-process CPU re-execution honours the scored ``nproc=1`` launch shape regardless of the
# spec's recorded gpu_count: it always runs rank 0 of a world of size 1.
LOCAL_WORLD_SIZE = 1
LOCAL_PAYLOAD_FILENAME = "payload.local.json"

ExecutorRun = Callable[[object, DockerRunSpec, float], DockerRunResult]


class MockCpuReexecError(RuntimeError):
    """The dispatched prism run spec violated the network-isolated, workspace+artifacts posture."""


def _as_text(value: str | bytes | None) -> str:
    """Coerce captured subprocess output to str, never None (``text=True`` already yields str)."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _mount_source(spec: DockerRunSpec, target: str) -> Path:
    for mount in spec.mounts:
        if mount.target == target:
            return mount.source
    raise MockCpuReexecError(f"prism run spec is missing the {target} mount")


def assert_network_isolated(spec: DockerRunSpec) -> None:
    """Reject a prism run spec that is not network-isolated to workspace + artifacts only.

    The eval container must run ``network=none`` with exactly the read-only workspace and the
    writable artifacts dir mounted; the secret val/test held-out split is NEVER mounted into it
    (architecture.md sections 5, 6, 11; VAL-PRISM-028/039).
    """
    if spec.limits.network != "none":
        raise MockCpuReexecError(
            f"prism eval container must run network=none, got {spec.limits.network!r}"
        )
    targets = {mount.target for mount in spec.mounts}
    if not targets <= {WORKSPACE_TARGET, ARTIFACTS_TARGET}:
        extra = sorted(targets - {WORKSPACE_TARGET, ARTIFACTS_TARGET})
        raise MockCpuReexecError(
            f"prism eval container mounts only workspace + artifacts; unexpected mounts: {extra}"
        )
    writable = {mount.target for mount in spec.mounts if not mount.read_only}
    if writable - {ARTIFACTS_TARGET}:
        raise MockCpuReexecError(
            "prism eval container exposes a writable mount other than the artifacts dir"
        )


def _localize_payload(
    workspace: Path,
    *,
    artifacts_dir: Path,
    train_data_dir: Path,
    reference_tokenizer_dir: Path | None,
) -> Path:
    """Rewrite the container payload's data paths to host paths for the local CPU run.

    In deploy the broker bind-mounts the locked train split (and reference tokenizers) at the
    container paths the payload records and maps the artifacts mount onto the host artifact dir;
    with no broker here the mock redirects ``context.data_dir``/``artifacts_dir`` (and the reference
    tokenizer dir) to the host paths and forces the single-process world, leaving every other
    challenge-authored field untouched so the runner behaves identically.
    """
    payload_path = workspace / "payload.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    context = payload.get("context")
    if not isinstance(context, dict):
        context = {}
        payload["context"] = context
    context["data_dir"] = str(train_data_dir)
    context["artifacts_dir"] = str(artifacts_dir)
    if reference_tokenizer_dir is not None:
        context["reference_tokenizer_dir"] = str(reference_tokenizer_dir)
    # The host has no ``/workspace`` mount, so redirect a staged resume checkpoint to its host path
    # under the local workspace (the broker would otherwise expose it at the container path).
    if context.get("resume_checkpoint_dir"):
        context["resume_checkpoint_dir"] = str(workspace / "resume_checkpoint")
    context["world_size"] = LOCAL_WORLD_SIZE
    context["rank"] = 0
    context["local_rank"] = 0
    context["distributed_backend"] = None
    local_payload_path = workspace / LOCAL_PAYLOAD_FILENAME
    local_payload_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return local_payload_path


def run_cpu_reexec(
    spec: DockerRunSpec,
    timeout_seconds: float,
    *,
    train_data_dir: Path,
    reference_tokenizer_dir: Path | None = None,
    python_executable: str | None = None,
) -> DockerRunResult:
    """Run the challenge ``runner.py`` for ``spec`` in one local CPU process (``nproc=1``).

    ``train_data_dir`` is the host-staged locked FineWeb-Edu train split the broker would otherwise
    bind-mount read-only. The run is deterministic (the runner forces the seed before any miner
    code) and produces the real ``prism_run_manifest.v2`` + ``trained_state.pt`` in the artifacts
    mount, so the prequential-bpb scoring + anti-cheat path is validated without a GPU.
    """
    assert_network_isolated(spec)
    workspace = _mount_source(spec, WORKSPACE_TARGET)
    artifacts = _mount_source(spec, ARTIFACTS_TARGET)
    runner = workspace / "runner.py"
    if not runner.is_file():
        raise MockCpuReexecError(f"prism workspace mount has no runner.py: {workspace}")
    local_payload = _localize_payload(
        workspace,
        artifacts_dir=artifacts,
        train_data_dir=Path(train_data_dir),
        reference_tokenizer_dir=(
            Path(reference_tokenizer_dir) if reference_tokenizer_dir is not None else None
        ),
    )
    artifacts.mkdir(parents=True, exist_ok=True)

    # Start from the spec env (gateway base URLs / determinism flags), then pin the local CPU
    # single-process runtime. No provider key is injected here (the spec carries none).
    env: dict[str, str] = {str(key): str(value) for key, value in spec.env.items()}
    env.update(
        {
            "PRISM_PROJECT_ROOT": str(workspace / "project"),
            "PRISM_DATA_DIR": str(train_data_dir),
            "PRISM_ARTIFACT_OUTPUT_PATH": str(artifacts),
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": str(LOCAL_WORLD_SIZE),
        }
    )
    if reference_tokenizer_dir is not None:
        env["PRISM_REFERENCE_TOKENIZER_DIR"] = str(reference_tokenizer_dir)
    # The local CPU process must not see a CUDA device even if the host advertises one.
    env["CUDA_VISIBLE_DEVICES"] = ""

    try:
        completed = subprocess.run(
            [python_executable or sys.executable, str(runner), str(local_payload)],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        # Mirror the real broker ``DockerExecutor.run`` timeout contract: a run that exceeds its
        # timeout maps to ``DockerRunResult(timed_out=True)`` (returncode 124, partial output) so
        # the mock and the real executor surface the same result shape to callers.
        return DockerRunResult(
            container_name=spec.name or "prism-cpu-reexec",
            stdout=_as_text(exc.stdout),
            stderr=_as_text(exc.stderr),
            returncode=124,
            timed_out=True,
        )
    return DockerRunResult(
        container_name=spec.name or "prism-cpu-reexec",
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
    )


def cpu_reexec_run(
    *,
    train_data_dir: Path,
    reference_tokenizer_dir: Path | None = None,
    python_executable: str | None = None,
    captured_specs: MutableSequence[DockerRunSpec] | None = None,
) -> ExecutorRun:
    """Build a ``DockerExecutor.run``-shaped callable for monkeypatching in CPU re-exec tests.

    The returned ``(self, spec, timeout_seconds)`` function ignores the bound executor ``self`` (no
    real broker/Docker is contacted), optionally records the dispatched spec into ``captured_specs``
    for ``network=none`` / mount assertions, and runs the challenge runner on CPU.
    """

    def _run(_self: object, spec: DockerRunSpec, timeout_seconds: float) -> DockerRunResult:
        if captured_specs is not None:
            captured_specs.append(spec)
        return run_cpu_reexec(
            spec,
            timeout_seconds,
            train_data_dir=train_data_dir,
            reference_tokenizer_dir=reference_tokenizer_dir,
            python_executable=python_executable,
        )

    return _run
