from __future__ import annotations

from pathlib import Path


def test_cli_facade_is_small_and_domain_package_exists() -> None:
    root = Path(__file__).resolve().parents[2]
    cli = root / "src/base/cli.py"

    assert len(cli.read_text(encoding="utf-8").splitlines()) <= 10
    assert (root / "src/base/cli_app/main.py").is_file()
    assert (root / "src/base/master/admin/auth.py").is_file()
    assert (root / "src/base/master/admin/runtime.py").is_file()
