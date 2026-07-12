"""Master must never construct or invoke set_weights (VAL-WEIGHT-058/CROSS-068).

Structural checks over master package modules, CLI wiring, and aggregation:
a shared Base wheel may still ship validator-only modules, but no supported
master command/configuration/import graph may construct a WeightSetter or call
set_weights.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from base.master import aggregation as aggregation_module
from base.master import service as service_module
from base.master.service import MasterWeightService

FORBIDDEN_SYMBOLS = ("WeightSetter", "set_weights", "create_bittensor_submit_runtime")


def _import_modules_in(source: str) -> list[str]:
    tree = ast.parse(source)
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
    return modules


def test_master_weight_service_has_no_submitter_hook() -> None:
    init_params = inspect.signature(MasterWeightService.__init__).parameters
    assert "weight_setter" not in init_params
    assert "submit" not in inspect.signature(MasterWeightService.run_epoch).parameters
    module_src = inspect.getsource(service_module)
    for symbol in FORBIDDEN_SYMBOLS:
        assert symbol not in module_src


def test_master_aggregation_has_no_chain_submit_path() -> None:
    module_src = inspect.getsource(aggregation_module)
    for symbol in ("WeightSetter", "set_weights", "create_bittensor_submit_runtime"):
        assert symbol not in module_src
    imports = _import_modules_in(module_src)
    assert not any(m.startswith("base.bittensor") for m in imports)
    assert not any(m.startswith("base.validator.weight_submitter") for m in imports)


def test_master_package_modules_do_not_import_submitter() -> None:
    root = Path(service_module.__file__).resolve().parent
    offenders: list[str] = []
    for path in sorted(root.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "weight_submitter" in source or "WeightSetter" in source:
            # Allow documentation mentions only in comments is still forbidden for
            # executable imports; scan AST imports.
            imports = _import_modules_in(source)
            if any(
                name
                in {
                    "base.bittensor.weight_setter",
                    "base.validator.weight_submitter",
                    "base.bittensor.factory",
                }
                or name.endswith("weight_setter")
                or name.endswith("weight_submitter")
                for name in imports
            ):
                offenders.append(path.name)
            # Direct name usage without import is also forbidden except this
            # test's own lists would not apply.
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id == "WeightSetter":
                    offenders.append(f"{path.name}:WeightSetter")
                if isinstance(node, ast.Attribute) and node.attr == "set_weights":
                    offenders.append(f"{path.name}:set_weights")
    assert offenders == []


def test_master_cli_help_does_not_expose_set_weights() -> None:
    from typer.testing import CliRunner

    from base.cli_app.main import app

    result = CliRunner().invoke(app, ["master", "--help"])
    assert result.exit_code == 0
    lowered = result.output.lower()
    assert "set_weights" not in lowered
    assert "submit weights" not in lowered or "never" in lowered
