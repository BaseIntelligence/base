#!/usr/bin/env python
"""Local mission prism service: worker plane ON + admission gate + explicit CPU re-exec test mode.

Part of the cross-repo local end-to-end harness (base advanced/local harness (not shipping)). Serves
the real prism challenge API on a loopback port configured so that:

* the worker plane is ON and the admission gate requires >=1 active worker for the submitting
  hotkey (queried from the base master, reusing the shared bridge token as the internal bearer);
* the repo's OWN CPU re-exec seam is installed as EXPLICIT test-mode config
  (``worker_plane.cpu_reexec_test_mode``) so any re-execution runs deterministically on CPU with no
  GPU/Docker/broker; and
* results are finalized from the base worker plane's forwarded ExecutionProof (no self-evaluation).

CONFIG-DRIVEN (JSON path in ``argv[1]`` / ``$MISSION_PRISM_CONFIG``). NOT for production.
"""

from __future__ import annotations

import atexit
import json
import logging
import sys
from pathlib import Path
from typing import Any

import uvicorn

from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings


def _load_config() -> dict[str, Any]:
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path is None:
        import os

        path = os.environ.get("MISSION_PRISM_CONFIG")
    if not path:
        raise SystemExit("usage: mission_prism.py <config.json>")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_settings(config: dict[str, Any]) -> PrismSettings:
    token = config["token"]
    # Legacy regression mode (VAL-CROSS-006): worker plane OFF + admission OFF so the service
    # behaves byte-for-byte as pre-mission (submissions accepted with no active worker; the
    # /internal/v1/work_units/result + audit routes are inert/404). The CPU re-exec seam stays
    # available for whichever executor runs (a legacy validator dispatch, in this mode).
    worker_plane_enabled = bool(config.get("worker_plane_enabled", True))
    return PrismSettings(
        database_url=f"sqlite+aiosqlite:///{config['db_path']}",
        shared_token=token,
        shared_token_file=None,
        allow_insecure_signatures=True,
        llm_review_enabled=False,
        llm_review_required=False,
        plagiarism_enabled=False,
        distributed_contract_policy="off",
        execution_backend="base_gpu",
        docker_enabled=True,
        docker_backend="broker",
        docker_broker_url="http://mission-broker:8082",
        docker_broker_token=token,
        sequence_length=int(config.get("sequence_length", 16)),
        base_eval_artifact_root=Path(config["artifact_root"]),
        public_submissions_enabled=True,
        worker_plane={
            "enabled": worker_plane_enabled,
            "admission_requires_worker": worker_plane_enabled
            and bool(config.get("admission_requires_worker", True)),
            "master_base_url": config["master_base_url"],
            "cpu_reexec_test_mode": True,
            "cpu_reexec_train_data_dir": config.get("train_data_dir"),
        },
    )


def _flush_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (ValueError, OSError):
            pass


def _configure_harness_logging(level: int = logging.INFO) -> None:
    """Line-buffer stdout/stderr and route logs there so drill logs are
    inspectable after teardown.

    The harness redirects each spawned process' stdout/stderr to a log file and
    tears it down with SIGTERM; block-buffered output would be lost on kill,
    leaving a 0-byte log. Line-buffering flushes every completed log line
    immediately, and an ``atexit`` flush covers the graceful-shutdown path.
    """

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(line_buffering=True)
            except (ValueError, OSError):
                pass
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
        force=True,
    )
    atexit.register(_flush_streams)


def main() -> None:
    _configure_harness_logging()
    config = _load_config()
    settings = build_settings(config)
    uvicorn.run(
        create_app(settings),
        host=str(config.get("host", "127.0.0.1")),
        port=int(config["port"]),
        log_level=str(config.get("log_level", "info")),
    )


if __name__ == "__main__":
    main()
