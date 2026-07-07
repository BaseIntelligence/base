"""``daemon.json`` templates must NOT set ``live-restore``.

``live-restore`` is incompatible with ``docker swarm init``/``join`` (it blocks a
fresh node from joining/initialising the swarm) and is inert on swarm nodes
(Docker disables live-restore while a node is in swarm mode). All three templates
are applied to swarm members, so the key was removed. These regression guards
fail loudly if it ever comes back, and confirm we removed ONLY that key (the
files stay valid JSON and keep their log rotation config).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SWARM_DIR = ROOT / "deploy" / "swarm"

DAEMON_TEMPLATES = (
    "daemon.validator.json",
    "daemon.cpu-worker.json",
    "daemon.worker.json",
)


@pytest.mark.parametrize("filename", DAEMON_TEMPLATES)
def test_daemon_template_has_no_live_restore(filename: str) -> None:
    data = json.load((SWARM_DIR / filename).open(encoding="utf-8"))
    assert "live-restore" not in data, (
        f"{filename} must not set live-restore: it is incompatible with "
        "docker swarm init/join and inert on swarm nodes"
    )


@pytest.mark.parametrize("filename", DAEMON_TEMPLATES)
def test_daemon_template_is_valid_json_and_keeps_log_opts(filename: str) -> None:
    # Regression: we removed ONLY live-restore, so each file still parses and
    # still caps container logs via log-opts.
    data = json.load((SWARM_DIR / filename).open(encoding="utf-8"))
    assert isinstance(data, dict)
    assert "log-opts" in data
