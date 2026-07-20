"""Production reference agents for the own-runner backend (Task 22, Phase A).

The full-set oracle parity gate needs two *production* reference agents that
reproduce stock harbor 0.13.1's behaviour WITHOUT importing harbor:

* :class:`OracleAgent` -- a faithful, harbor-free reproduction of
  ``harbor/agents/oracle.py``'s ``OracleAgent.run``. It executes the task's
  reference ``solve.sh`` (staged into the live container by the preparer, see
  :func:`stage_solution_into`) through ``environment.exec`` with harbor's
  ``DEBIAN_FRONTEND=noninteractive`` plus the agent's ``extra_env``, so the
  verifier -- which runs AFTER the agent on the SAME environment -- observes a
  solved ``/app``. This is the resolved=1 ceiling for solvable tasks. It is NOT
  the trivial ``_OracleAgent`` test double (which merely returns ``"DONE"`` and
  solves nothing).

* :class:`NopAgent` -- a pure no-op reproducing ``harbor/agents/nop.py``: it
  touches nothing, so the verifier observes the pristine task and scores
  resolved=0. This is the floor used to prove the gate distinguishes a solving
  agent from a non-solving one.

Faithfulness notes vs. harbor's ``OracleAgent`` (Linux ``.sh`` path, the only
shape the full set exercises):

* harbor stages the solution dir from inside ``run`` via
  ``environment.upload_dir`` and then runs the discovered solve script. Here the
  agent class is fixed once per job (the driver constructs ONE ``agent_class``
  with fixed kwargs) while the solution is per-task, so staging cannot flow
  through agent construction. We split it into :func:`stage_solution_into`, which
  the per-task preparer calls against the same live environment -- the net effect
  on the container is identical (solution materialised at
  :data:`SOLUTION_CONTAINER_DIR`, harbor's ``EnvironmentPaths.solution_dir`` on
  Linux).
* harbor runs ``bash <path>`` for ``.sh`` scripts after a ``chmod +x`` (its
  ``needs_chmod`` is True for ``.sh``). Invoking via ``bash <script>`` makes the
  executable bit irrelevant, so we skip the separate ``chmod`` exec -- the
  command's observable effect on ``/app`` is identical.
* harbor builds ``env = {"DEBIAN_FRONTEND": "noninteractive", **extra_env}`` and
  passes it to a single ``environment.exec``; we reproduce that exactly. (The
  full set's two Phase-A tasks declare no ``[solution].env`` block, so the
  optional ``solution.env`` merge harbor performs is a no-op here and is left to
  a later task.)
* The agent's wall-clock budget is enforced by the driver's
  :func:`asyncio.timeout`, so -- like harbor when ``agent_timeout_sec`` is unset
  -- we pass no explicit ``timeout_sec`` to the exec.

This module imports nothing from harbor and does not modify the exec-bridge, the
driver, or the SDK contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

#: Container path the reference solution is staged at, matching harbor's
#: ``EnvironmentPaths.solution_dir`` on Linux (``/solution``). The oracle runs
#: ``bash {SOLUTION_CONTAINER_DIR}/solve.sh``.
SOLUTION_CONTAINER_DIR = "/solution"

#: Harbor's solution exec env floor. ``OracleAgent.run`` always sets
#: ``DEBIAN_FRONTEND=noninteractive`` (so apt-get in solve.sh never blocks on a
#: prompt) and merges the agent's ``extra_env`` on top.
DEBIAN_FRONTEND_ENV = {"DEBIAN_FRONTEND": "noninteractive"}


def stage_solution_into(environment: Any, source_dir: Path | str) -> None:
    """Copy a host solution dir into the container at :data:`SOLUTION_CONTAINER_DIR`.

    This is the per-trial seam the preparer calls (the agent class is fixed per
    job, the solution is per task). It mirrors harbor's
    ``environment.upload_dir(source_dir=..., target_dir=solution_dir)`` so the
    container ends up identical to a stock-harbor oracle run before
    :meth:`OracleAgent.run` executes the staged ``solve.sh``.
    """
    environment.upload_dir(Path(source_dir), SOLUTION_CONTAINER_DIR)


class OracleAgent:
    """Harbor-free reproduction of harbor's ``OracleAgent`` (Linux ``.sh`` path).

    Constructed by the driver as ``OracleAgent(logs_dir=, model_name=,
    **extra)`` (harbor-factory parity); ``extra_env`` -- when the driver forwards
    the trial's ``agent_env`` -- is injected into the ``solve.sh`` exec env. The
    solution itself is staged separately by :func:`stage_solution_into`.
    """

    def __init__(
        self,
        *,
        logs_dir: Path | str | None = None,
        model_name: str | None = None,
        extra_env: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._logs_dir = Path(logs_dir) if logs_dir is not None else None
        self._model_name = model_name
        self._extra_env: dict[str, str] = dict(extra_env or {})

    async def setup(self, environment: Any) -> None:
        """No-op, matching harbor's ``OracleAgent.setup`` (returns immediately)."""
        return None

    async def run(
        self,
        instruction: str,
        environment: Any,
        context: Any | None = None,
    ) -> str | None:
        """Execute the staged ``solve.sh`` and return its merged output.

        Reproduces harbor's single solution exec: ``bash <solution>/solve.sh``
        with ``env = {"DEBIAN_FRONTEND": "noninteractive", **extra_env}`` and no
        explicit per-command timeout (the driver's wall-clock governs the run).
        The returned stdout becomes the agent's ``agent.log`` evidence.
        """
        command = f"bash {SOLUTION_CONTAINER_DIR}/solve.sh"
        env = {**DEBIAN_FRONTEND_ENV, **self._extra_env}
        result = await environment.exec(command, env=env)
        return result.stdout


class NopAgent:
    """Harbor-free reproduction of harbor's ``NopAgent`` -- the resolved=0 floor.

    Touches nothing: ``setup`` and ``run`` are no-ops, so the verifier observes
    the pristine task environment and scores it unresolved.
    """

    def __init__(
        self,
        *,
        logs_dir: Path | str | None = None,
        model_name: str | None = None,
        extra_env: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._logs_dir = Path(logs_dir) if logs_dir is not None else None
        self._model_name = model_name
        self._extra_env: dict[str, str] = dict(extra_env or {})

    async def setup(self, environment: Any) -> None:
        """No-op (matches ``harbor/agents/nop.py``)."""
        return None

    async def run(
        self,
        instruction: str,
        environment: Any,
        context: Any | None = None,
    ) -> None:
        """No-op: returns None and issues no exec (the resolved=0 floor)."""
        return None


__all__ = [
    "DEBIAN_FRONTEND_ENV",
    "SOLUTION_CONTAINER_DIR",
    "NopAgent",
    "OracleAgent",
    "stage_solution_into",
]
