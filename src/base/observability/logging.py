from __future__ import annotations

import json
import logging
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _resolve_level(level: int | str) -> int:
    """Coerce a level to an int, mapping names case-insensitively.

    An unknown name falls back to ``INFO`` instead of raising so a misconfigured
    ``observability.log_level`` never crashes process startup.
    """

    if isinstance(level, str):
        resolved = logging.getLevelName(level.upper())
        return resolved if isinstance(resolved, int) else logging.INFO
    return level


def configure_logging(json_logs: bool = True, level: int | str = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        JsonFormatter()
        if json_logs
        else logging.Formatter("%(levelname)s %(name)s %(message)s")
    )
    logging.basicConfig(level=_resolve_level(level), handlers=[handler], force=True)
