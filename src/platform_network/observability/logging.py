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


def configure_logging(json_logs: bool = True, level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        JsonFormatter()
        if json_logs
        else logging.Formatter("%(levelname)s %(name)s %(message)s")
    )
    logging.basicConfig(level=level, handlers=[handler], force=True)
