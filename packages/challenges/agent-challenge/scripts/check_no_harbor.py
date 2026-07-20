#!/usr/bin/env python3
"""CI guard: fail if product code imports the ``harbor`` package.

Terminal-Bench execution runs through the in-tree ``own_runner`` backend. The
legacy ``harbor`` PyPI dependency has been removed, so no shipping module may
import it. This guard parses every product ``.py`` file with the ``ast`` module
(so prose/docstrings that merely mention "harbor" never trigger) and reports any
real ``import harbor`` / ``from harbor import ...`` statement.

Scope:
  - scans ``src/`` and ``scripts/`` (the code that ships / runs in production);
  - excludes ``tests/`` (the parity harness deliberately ``importorskip``s
    harbor to compare against the legacy backend when it is installed).

Exit code is non-zero when any forbidden import is found.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCAN_DIRS = ("src", "scripts")
FORBIDDEN_ROOT = "harbor"


def _imports_harbor(tree: ast.AST) -> list[tuple[int, str]]:
    """Return (lineno, statement) pairs that import the harbor package."""
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root == FORBIDDEN_ROOT:
                    hits.append((node.lineno, f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom):
            # Ignore relative imports (node.level > 0); they cannot be the
            # third-party ``harbor`` package.
            if node.level == 0 and node.module:
                root = node.module.split(".", 1)[0]
                if root == FORBIDDEN_ROOT:
                    names = ", ".join(a.name for a in node.names)
                    hits.append((node.lineno, f"from {node.module} import {names}"))
    return hits


def main() -> int:
    violations: list[str] = []
    for scan_dir in SCAN_DIRS:
        base = REPO_ROOT / scan_dir
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except SyntaxError as exc:  # pragma: no cover - defensive
                violations.append(f"{path}: could not parse ({exc})")
                continue
            for lineno, stmt in _imports_harbor(tree):
                rel = path.relative_to(REPO_ROOT)
                violations.append(f"{rel}:{lineno}: forbidden import -> {stmt}")

    if violations:
        print("check_no_harbor: FAILED — product code must not import 'harbor':")
        for v in violations:
            print(f"  {v}")
        return 1

    print("check_no_harbor: OK — no 'harbor' imports in src/ or scripts/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
