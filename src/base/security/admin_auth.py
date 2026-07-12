from __future__ import annotations

import hmac
import stat
from pathlib import Path


class SecretFileError(ValueError):
    """Raised when a required secret file is missing, empty, or too permissive."""


def read_secret(value: str | None = None, file_path: str | None = None) -> str:
    if value:
        return value
    if file_path:
        path = Path(file_path)
        if not path.is_file():
            return ""
        return path.read_text(encoding="utf-8").strip()
    return ""


def require_protected_secret_file(
    file_path: str | Path,
    *,
    name: str = "secret",
    max_mode: int = 0o600,
) -> str:
    """Load a secret only when the path is a non-empty, least-privilege file.

    Production and Compose install paths refuse world/group-readable files and
    empty content. Diagnostics identify the secret *name/path class* only, never
    the secret value.
    """

    path = Path(file_path)
    if not path.is_file():
        raise SecretFileError(f"required secret file missing: {name}")
    mode = stat.S_IMODE(path.stat().st_mode)
    # Strictest subset of owner-only (0600 or more restrictive).
    if mode & ~max_mode:
        raise SecretFileError(
            f"secret file {name!r} has too-permissive mode {oct(mode)}; "
            f"require {oct(max_mode)} or stricter"
        )
    try:
        content = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise SecretFileError(f"secret file unreadable: {name}") from exc
    if not content:
        raise SecretFileError(f"required secret file empty: {name}")
    return content


def constant_time_match(left: str, right: str) -> bool:
    return bool(left and right and hmac.compare_digest(left, right))
