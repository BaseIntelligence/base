"""Evaluation orchestration, benchmarks, and weight exports.

The public API historically surfaced by ``from .runner import *`` is exposed
lazily via :pep:`562` module ``__getattr__``. Importing a lightweight submodule
(e.g. ``own_runner_backend``, which the lean canonical CVM image runs) therefore
no longer eagerly pulls the heavy orchestration stack (sqlalchemy / fastapi /
bittensor) that :mod:`agent_challenge.evaluation.runner` imports at module load.
Accessing any ``runner`` name off this package (``evaluation.create_evaluation_job``)
still works — it triggers the ``runner`` import on first use.
"""

from __future__ import annotations

import importlib
from typing import Any

#: The heavy orchestration submodule whose public API this package re-exports.
_RUNNER_SUBMODULE = "runner"


def __getattr__(name: str) -> Any:
    # Dunder lookups (e.g. ``__all__``, ``__wrapped__``) must not trigger the
    # heavy import; let them raise so normal attribute resolution is unaffected.
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # Resolve ``runner`` via importlib rather than ``from . import runner``: the
    # latter re-enters this ``__getattr__`` (through ``_handle_fromlist``'s
    # ``hasattr``) when the submodule fails to import, causing a RecursionError
    # in the lean image where runner's heavy deps (sqlalchemy/fastapi/bittensor)
    # are absent. importlib imports the submodule directly (no re-entrancy) and
    # surfaces a clean AttributeError when it cannot load.
    try:
        runner = importlib.import_module(f"{__name__}.{_RUNNER_SUBMODULE}")
    except ImportError as exc:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r} "
            f"({_RUNNER_SUBMODULE!r} submodule is unavailable: {exc})"
        ) from exc
    if name == _RUNNER_SUBMODULE:
        return runner
    return getattr(runner, name)
