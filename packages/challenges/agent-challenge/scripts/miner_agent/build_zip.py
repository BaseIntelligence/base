"""Deterministic ZIP packaging for the miner agent (stdlib only)."""

from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

MAX_ZIP_BYTES = 1_048_576
_SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
}
_SKIP_SUFFIXES = {".pyc", ".pyo", ".zip"}
_SKIP_NAMES = {"build_zip.py"}  # packaging helper stays out of the submission


def build_zip(agent_dir: Path | str) -> bytes:
    """Package ``agent_dir`` into a reproducible submission ZIP."""
    agent_dir = Path(agent_dir).resolve()
    entry = agent_dir / "agent.py"
    if not entry.is_file():
        raise FileNotFoundError(f"missing required entrypoint: {entry}")
    if "class Agent" not in entry.read_text(encoding="utf-8"):
        raise ValueError(f"{entry} must define a top-level class Agent")

    files: list[Path] = []
    for path in sorted(agent_dir.rglob("*")):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(agent_dir).parts
        if any(part in _SKIP_DIRS for part in rel_parts):
            continue
        if path.name in _SKIP_NAMES:
            continue
        if path.suffix in _SKIP_SUFFIXES:
            continue
        files.append(path)

    buffer = BytesIO()
    fixed_date = (2026, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            arcname = path.relative_to(agent_dir).as_posix()
            info = zipfile.ZipInfo(arcname, date_time=fixed_date)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())

    data = buffer.getvalue()
    if len(data) > MAX_ZIP_BYTES:
        raise ValueError(f"packaged ZIP is {len(data)} bytes, exceeds {MAX_ZIP_BYTES}")
    # Credential-shaped markers (assembled so source itself is clean).
    _m1 = bytes([0x73, 0x6B, 0x2D])  # s k -
    _m2 = bytes([0x42, 0x65, 0x61, 0x72, 0x65, 0x72, 0x20])  # B e a r e r space
    if _m1 in data or _m2 in data:
        raise ValueError("packaged ZIP contains credential-shaped literals")
    return data


def main() -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Build miner agent submission ZIP")
    parser.add_argument(
        "--agent-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing agent.py",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "dist" / "miner_agent.zip",
        help="Output ZIP path",
    )
    args = parser.parse_args()
    data = build_zip(args.agent_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(data)
    print(f"wrote {args.out} ({len(data)} bytes)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
