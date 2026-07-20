"""Lean-image import guard for the evaluation package (M2 heavy-import blocker).

The canonical CVM image is a lean own_runner wrapper (pydantic only) and must be
able to run ``python -m agent_challenge.canonical.entrypoint run ...`` without the
heavy orchestration stack (sqlalchemy / fastapi / bittensor) that
``agent_challenge.evaluation.runner`` imports. Historically
``evaluation/__init__`` did ``from .runner import *`` eagerly, so importing ANY
evaluation submodule (e.g. ``own_runner_backend``) pulled the whole stack.

These tests pin the fix: the package exposes ``runner``'s public API lazily
(:pep:`562`), so a bare ``own_runner_backend`` / entrypoint import works with the
heavy modules unavailable, while the historical
``from agent_challenge.evaluation import create_evaluation_job`` access still
resolves.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

# Modules the lean canonical image does NOT ship; importing them must not be
# required to load own_runner_backend / the entrypoint.
_HEAVY_MODULES = ("sqlalchemy", "fastapi", "bittensor")


def _run_in_lean_interpreter(body: str) -> subprocess.CompletedProcess[str]:
    """Run ``body`` in a fresh interpreter with the heavy modules blocked."""

    script = textwrap.dedent(
        """
        import sys
        for _mod in {heavy!r}:
            sys.modules[_mod] = None  # any import of these now raises ImportError
        {body}
        """
    ).format(heavy=list(_HEAVY_MODULES), body=textwrap.indent(textwrap.dedent(body), ""))
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )


def test_own_runner_backend_imports_without_heavy_stack() -> None:
    proc = _run_in_lean_interpreter(
        """
        import agent_challenge.evaluation.own_runner_backend as backend
        assert hasattr(backend, "main")
        print("OK", backend.__name__)
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK agent_challenge.evaluation.own_runner_backend" in proc.stdout


def test_entrypoint_run_imports_backend_without_heavy_stack() -> None:
    # ``entrypoint run <args>`` reaches _run_eval, which imports own_runner_backend;
    # this import must not require sqlalchemy/fastapi/bittensor. Feeding a bogus
    # subcommand makes the backend's own argparse reject it (exit 2) AFTER the
    # import succeeds, so reaching that error proves the lean import worked.
    proc = _run_in_lean_interpreter(
        """
        from agent_challenge.canonical import entrypoint
        try:
            entrypoint.main(["run", "bogus-subcommand"])
        except SystemExit as exc:
            print("EXIT", exc.code)
        """
    )
    combined = proc.stdout + proc.stderr
    # The backend module loaded (its argparse prog appears) and no heavy-import wall.
    assert "sqlalchemy" not in combined, combined
    assert "agent-challenge-own-runner" in combined, combined


def test_entrypoint_check_runs_without_heavy_stack() -> None:
    proc = _run_in_lean_interpreter(
        """
        from agent_challenge.canonical import entrypoint
        rc = entrypoint.main(["check"])
        assert rc == 0
        print("CHECK", rc)
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "CHECK 0" in proc.stdout


def test_importing_heavy_module_still_fails_in_lean_interpreter() -> None:
    # Sanity: the lean-interpreter harness really does block the heavy stack, so
    # the tests above are meaningful (not passing because the block is a no-op).
    proc = _run_in_lean_interpreter(
        """
        try:
            import sqlalchemy  # noqa: F401
        except ImportError:
            print("BLOCKED")
        else:
            print("NOT BLOCKED")
        """
    )
    assert "BLOCKED" in proc.stdout, proc.stderr


def test_attribute_style_backend_import_without_heavy_stack() -> None:
    # ``from agent_challenge.evaluation import own_runner_backend`` (attribute
    # style) must resolve the real lean submodule even when ``runner``'s heavy
    # deps are absent, instead of recursing through the PEP 562 ``__getattr__``.
    proc = _run_in_lean_interpreter(
        """
        from agent_challenge.evaluation import own_runner_backend
        assert hasattr(own_runner_backend, "main")
        print("OK", own_runner_backend.__name__)
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK agent_challenge.evaluation.own_runner_backend" in proc.stdout


def test_runner_name_access_raises_cleanly_without_heavy_stack() -> None:
    # A ``runner``-only name must raise a clean AttributeError (never a
    # RecursionError) when the heavy ``runner`` submodule cannot import in the
    # lean image, so callers get a legible error instead of a recursion crash.
    proc = _run_in_lean_interpreter(
        """
        from agent_challenge import evaluation
        try:
            evaluation.create_evaluation_job  # noqa: B018 - runner-only name
        except RecursionError:
            print("RECURSION")
        except AttributeError:
            print("ATTRIBUTE-ERROR")
        else:
            print("RESOLVED")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "ATTRIBUTE-ERROR" in proc.stdout
    assert "RECURSION" not in proc.stdout


def test_public_runner_api_is_lazily_accessible() -> None:
    # The historical ``from agent_challenge.evaluation import <runner name>`` access
    # must still resolve (lazily) in the full environment.
    from agent_challenge import evaluation

    assert callable(evaluation.create_evaluation_job)
    assert callable(evaluation.run_evaluation_job)


def test_unknown_attribute_still_raises_attribute_error() -> None:
    from agent_challenge import evaluation

    try:
        evaluation.this_name_does_not_exist  # noqa: B018
    except AttributeError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("expected AttributeError for an unknown attribute")
