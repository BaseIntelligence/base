from __future__ import annotations

import hmac
from pathlib import Path


def read_secret(value: str | None = None, file_path: str | None = None) -> str:
    if value:
        return value
    if file_path:
        path = Path(file_path)
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8").strip()
    return ""


def constant_time_match(left: str, right: str) -> bool:
    return bool(left and right and hmac.compare_digest(left, right))
