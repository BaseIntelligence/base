"""Swarm tmpfs translation must never silently drop execution semantics.

The own_runner job container installs the miner agent into ``/tmp/.local`` under
a read-only rootfs, so its tmpfs spec carries ``exec``. Docker Swarm's
``--mount type=tmpfs`` cannot express that flag (verified against Docker
29.2.1: ``exec`` is rejected as a non key=value field and ``tmpfs-options`` is
an unknown option), and tmpfs defaults to ``noexec``. Dropping the flag on the
way through therefore produces a mount that loads no shared object -- the exact
production failure that cost hours to diagnose. Refuse instead.
"""

from __future__ import annotations

import pytest

from base.master.docker_orchestrator import DockerOrchestrationError
from base.master.swarm_backend import _tmpfs_mount_arg


def test_exec_is_refused_rather_than_silently_dropped() -> None:
    with pytest.raises(DockerOrchestrationError) as excinfo:
        _tmpfs_mount_arg("/tmp:rw,nosuid,exec,size=2g")
    message = str(excinfo.value)
    assert "exec" in message
    assert "/tmp" in message


def test_default_noexec_spec_still_translates() -> None:
    arg = _tmpfs_mount_arg("/tmp:rw,noexec,nosuid,size=512m")
    assert "type=tmpfs" in arg
    assert "destination=/tmp" in arg
    assert "tmpfs-size=512m" in arg


def test_spec_without_size_translates() -> None:
    assert _tmpfs_mount_arg("/tmp:rw,nosuid,nodev") == "type=tmpfs,destination=/tmp"


def test_unknown_option_is_refused_rather_than_dropped() -> None:
    with pytest.raises(DockerOrchestrationError):
        _tmpfs_mount_arg("/tmp:rw,totally-unknown-option")


def test_relative_path_still_refused() -> None:
    with pytest.raises(DockerOrchestrationError):
        _tmpfs_mount_arg("tmp:size=1m")
