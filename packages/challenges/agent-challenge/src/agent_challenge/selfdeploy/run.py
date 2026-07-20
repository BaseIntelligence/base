"""Run the canonical eval and surface its result, failing closed (VAL-DEPLOY-011).

The miner points the run at the validator key-release endpoint; the in-CVM
``own_runner`` backend obtains the golden key from *exactly* that endpoint before
scoring and fails closed (a parseable score-0 result, no attestation) if the
endpoint is unreachable or denies the quote. This module drives that backend with
the endpoint wired in, captures its single ``BASE_BENCHMARK_RESULT=`` line, and
surfaces the outcome so a misconfigured endpoint yields a clear miner-facing error
and NO fabricated attested result.

The backend is injectable so the flow is testable offline; the default runner is
:func:`agent_challenge.evaluation.own_runner_backend.main` (imported lazily to
keep this module light).
"""

from __future__ import annotations

import contextlib
import io
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from agent_challenge.keyrelease.client import KEY_RELEASE_FAILED_REASON, KEY_RELEASE_URL_ENV
from agent_challenge.selfdeploy.result import ResultSurfaceError, SurfacedResult, surface_result

#: Signature of a backend runner: argv -> exit code (writes the result line to stdout).
BackendMain = Callable[[Sequence[str]], int]


class RunError(RuntimeError):
    """The eval run could not be executed at all (before any result line)."""


@dataclass(frozen=True)
class RunOutcome:
    """The outcome of a self-deploy eval run."""

    exit_code: int
    attested: bool
    surfaced: SurfacedResult | None
    stdout: str
    clear_error: str | None

    @property
    def succeeded(self) -> bool:
        """True only when the run produced a genuine attested result."""

        return self.exit_code == 0 and self.attested


@contextlib.contextmanager
def _patched_environ(overrides: Mapping[str, str]):
    """Temporarily apply ``overrides`` to ``os.environ`` and restore afterwards."""

    saved: dict[str, str | None] = {}
    try:
        for key, value in overrides.items():
            saved[key] = os.environ.get(key)
            os.environ[key] = value
        yield
    finally:
        for key, previous in saved.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


def _default_backend_main() -> BackendMain:
    from agent_challenge.evaluation.own_runner_backend import main as backend_main

    return backend_main


def run_eval(
    *,
    job_dir: str,
    task_ids: Sequence[str],
    key_release_url: str,
    binding_env: Mapping[str, str] | None = None,
    extra_args: Sequence[str] = (),
    backend_main: BackendMain | None = None,
) -> RunOutcome:
    """Run the eval with the key-release endpoint wired in; fail closed on failure.

    Wires ``key_release_url`` into the backend via
    :data:`~agent_challenge.keyrelease.client.KEY_RELEASE_URL_ENV` (so the run
    requests golden material from exactly that endpoint), invokes the backend,
    captures its result line, and surfaces the outcome. When the endpoint is
    unreachable or denies the quote the backend fails closed and this returns a
    :class:`RunOutcome` with ``attested=False`` and a clear error, surfacing NO
    attested result (VAL-DEPLOY-011).
    """

    if not isinstance(key_release_url, str) or not key_release_url.strip():
        raise RunError("a validator key-release endpoint URL is required to run the eval")
    if not task_ids:
        raise RunError("at least one --task is required to run the eval")

    runner = backend_main if backend_main is not None else _default_backend_main()

    argv: list[str] = ["run", "--job-dir", job_dir]
    for task_id in task_ids:
        argv += ["--task", task_id]
    argv += list(extra_args)

    overrides: dict[str, str] = {KEY_RELEASE_URL_ENV: key_release_url.strip()}
    if binding_env:
        overrides.update({str(k): str(v) for k, v in binding_env.items()})

    buffer = io.StringIO()
    with _patched_environ(overrides), contextlib.redirect_stdout(buffer):
        exit_code = int(runner(argv))
    stdout = buffer.getvalue()

    try:
        surfaced = surface_result(stdout)
    except ResultSurfaceError:
        return RunOutcome(
            exit_code=exit_code,
            attested=False,
            surfaced=None,
            stdout=stdout,
            clear_error=(
                "the eval produced no parseable result line; no golden key was obtained and "
                "no attested result was produced"
            ),
        )

    if surfaced.attested and exit_code == 0:
        return RunOutcome(
            exit_code=exit_code,
            attested=True,
            surfaced=surfaced,
            stdout=stdout,
            clear_error=None,
        )

    reason = surfaced.reason_code or "unknown"
    if reason == KEY_RELEASE_FAILED_REASON:
        message = (
            "key-release failed against the configured validator endpoint "
            f"({key_release_url.strip()!r}): no golden key was obtained, so the eval failed closed "
            "and produced NO attested result or score"
        )
    else:
        message = (
            f"the eval failed closed (reason_code={reason!r}); no attested result was produced"
        )
    return RunOutcome(
        exit_code=exit_code,
        attested=False,
        surfaced=surfaced,
        stdout=stdout,
        clear_error=message,
    )


__all__ = [
    "BackendMain",
    "RunError",
    "RunOutcome",
    "run_eval",
]
