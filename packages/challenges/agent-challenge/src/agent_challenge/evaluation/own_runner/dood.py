"""DooD (Docker-outside-of-Docker) launch policy for the in-CVM orchestrator (M2).

The orchestrator runs INSIDE the Phala TDX CPU CVM and launches Terminal-Bench
task containers as *siblings* on the guest Docker daemon by talking to the
bind-mounted guest Docker unix socket (DooD -- verified as the working
mechanism in the live spike). This module is the single source of that launch
policy:

* the Docker client target is ALWAYS the guest unix socket
  ``unix:///var/run/docker.sock`` -- never a ``tcp://`` endpoint and never an
  inner ``dockerd`` (DinD). A ``tcp://`` ``DOCKER_HOST`` is rejected fail-closed
  (:func:`resolve_docker_host`) rather than silently dialed;
* the guest Docker socket (and the ``dstack.sock`` used for attestation) are
  mounted ONLY on the orchestrator and must NEVER be bind-mounted into a task
  container -- handing a task the socket would let untrusted task code escape to
  the guest daemon. :func:`assert_no_socket_mounts` guards every task-container
  launch spec against that exposure.

The helpers are pure and dependency-free (only the stdlib), so the launch policy
is fully testable offline with the Docker client faked, and so this module stays
import-light for the lean canonical image (it never pulls the heavy
``evaluation`` stack).
"""

from __future__ import annotations

import os
import posixpath
from collections.abc import Iterable, Iterator, Mapping

#: The guest Docker unix socket path bind-mounted into the orchestrator.
DOCKER_SOCKET_PATH = "/var/run/docker.sock"
#: The dstack attestation unix socket path bind-mounted into the orchestrator.
DSTACK_SOCKET_PATH = "/var/run/dstack.sock"
#: The canonical Docker client target for the orchestrator: the guest unix
#: socket only (DooD). Never a ``tcp://`` endpoint, never an inner ``dockerd``.
DOOD_DOCKER_HOST = f"unix://{DOCKER_SOCKET_PATH}"
#: Sockets that must never be mounted into a launched task container.
SENSITIVE_SOCKET_PATHS: tuple[str, ...] = (DOCKER_SOCKET_PATH, DSTACK_SOCKET_PATH)

#: The ``docker run`` flags that declare a bind mount / volume.
_VOLUME_FLAGS = ("-v", "--volume", "--mount")

#: ``--mount`` key names whose value is a host/container filesystem path.
_MOUNT_PATH_KEYS = frozenset({"source", "src", "target", "destination", "dst"})
#: All recognised ``--mount`` keys (used to detect the ``key=value`` spec form).
_MOUNT_KEYS = _MOUNT_PATH_KEYS | frozenset(
    {"type", "readonly", "ro", "bind-propagation", "volume-opt", "consistency"}
)


class DoodConfigError(Exception):
    """The Docker client was pointed at a non-unix (e.g. ``tcp://``) endpoint.

    DooD is unix-socket-only; a ``tcp://`` ``DOCKER_HOST`` is rejected rather
    than dialed (Docker-over-TCP is blocked in the CVM and would be a socket-less
    escape hatch), so this is raised fail-closed instead of falling back.
    """


class DoodSocketExposureError(Exception):
    """A task-container launch spec would bind-mount the docker/dstack socket.

    Mounting the guest Docker (or dstack) socket into a task container would let
    untrusted task code drive the guest daemon and escape isolation, so the
    launch is refused.
    """


def is_tcp_docker_host(host: str | None) -> bool:
    """Return ``True`` if ``host`` is a Docker-over-TCP endpoint (``tcp://``)."""

    return bool(host) and host.strip().lower().startswith("tcp://")


def resolve_docker_host(
    env: Mapping[str, str] | None = None,
    *,
    default: str = DOOD_DOCKER_HOST,
) -> str:
    """Resolve the Docker client target, enforcing the unix-socket-only policy.

    An unset/empty ``DOCKER_HOST`` resolves to the guest unix socket
    (``default``). An explicit ``unix://`` socket is preserved. A ``tcp://``
    endpoint is rejected fail-closed with :class:`DoodConfigError` (never
    dialed).
    """

    mapping = os.environ if env is None else env
    configured = mapping.get("DOCKER_HOST", "").strip()
    if not configured:
        return default
    if is_tcp_docker_host(configured):
        raise DoodConfigError(
            f"DooD is unix-socket-only; refusing Docker-over-TCP host {configured!r}"
        )
    return configured


def dood_docker_env(
    base_env: Mapping[str, str] | None = None,
    *,
    docker_host: str = DOOD_DOCKER_HOST,
) -> dict[str, str]:
    """Return a subprocess environment with ``DOCKER_HOST`` pinned to the socket.

    Copies ``base_env`` (defaulting to the current process env) and forces
    ``DOCKER_HOST`` to the guest unix socket so every Docker interaction resolves
    to DooD, overriding any ambient (possibly ``tcp://``) value.
    """

    env = dict(os.environ if base_env is None else base_env)
    env["DOCKER_HOST"] = docker_host
    return env


def dood_docker_argv(
    docker_bin: str = "docker",
    *args: str,
    docker_host: str = DOOD_DOCKER_HOST,
) -> list[str]:
    """Return a ``docker`` argv that explicitly targets the guest unix socket.

    Pins ``-H <unix socket>`` so the client target is unambiguous even when the
    ambient ``DOCKER_HOST`` is unset or altered.
    """

    return [docker_bin, "-H", docker_host, *args]


def iter_volume_mounts(argv: Iterable[str]) -> Iterator[str]:
    """Yield the mount/volume specs declared by a ``docker run`` argv.

    Handles the separated (``-v SPEC`` / ``--volume SPEC`` / ``--mount SPEC``)
    and inline (``--volume=SPEC`` / ``--mount=SPEC`` / ``-v=SPEC``) forms.
    """

    tokens = list(argv)
    index = 0
    count = len(tokens)
    while index < count:
        token = tokens[index]
        if token in _VOLUME_FLAGS:
            if index + 1 < count:
                yield tokens[index + 1]
            index += 2
            continue
        for flag in ("--volume=", "--mount=", "-v="):
            if token.startswith(flag):
                yield token[len(flag) :]
                break
        index += 1


def _normalize_mount_path(path: str) -> str:
    """Normalise a mount path for path-aware comparison (``None``/empty safe)."""

    stripped = path.strip()
    if not stripped:
        return ""
    return posixpath.normpath(stripped)


def _path_is_at_or_above(mount_path: str, socket_path: str) -> bool:
    """Return ``True`` if mounting ``mount_path`` would expose ``socket_path``.

    True when ``mount_path`` IS the socket, or is an ancestor directory that
    contains it (e.g. ``/var/run`` exposes ``/var/run/docker.sock``). Uses path
    segment boundaries, so a mere textual prefix like ``/var/running`` does NOT
    match ``/var/run``.
    """

    mount_norm = _normalize_mount_path(mount_path)
    socket_norm = _normalize_mount_path(socket_path)
    if not mount_norm or not socket_norm:
        return False
    if mount_norm == socket_norm:
        return True
    prefix = mount_norm if mount_norm.endswith("/") else mount_norm + "/"
    return socket_norm.startswith(prefix)


def _mount_candidate_paths(spec: str) -> list[str]:
    """Extract the host/container paths declared by one mount spec.

    Handles both the ``-v``/``--volume`` ``source:target[:options]`` form and the
    ``--mount type=bind,source=...,target=...`` key/value form.
    """

    spec = spec.strip()
    if not spec:
        return []
    first_field = spec.split(",", 1)[0]
    first_key = first_field.split("=", 1)[0].strip().lower()
    if "=" in first_field and first_key in _MOUNT_KEYS:
        paths: list[str] = []
        for field in spec.split(","):
            key, sep, value = field.partition("=")
            if sep and key.strip().lower() in _MOUNT_PATH_KEYS:
                paths.append(value.strip())
        return paths
    # ``-v``/``--volume`` form: source[:target[:options]]. On Linux the first two
    # colon-separated fields are the host source and container target.
    fields = spec.split(":")
    return [field for field in fields[:2] if field.strip()]


def _mount_exposes_socket(spec: str, socket_paths: Iterable[str]) -> bool:
    """Return ``True`` if ``spec`` mounts (or exposes via a parent dir) a socket."""

    candidates = _mount_candidate_paths(spec)
    return any(
        _path_is_at_or_above(candidate, socket)
        for candidate in candidates
        for socket in socket_paths
    )


def socket_mount_specs(
    argv: Iterable[str],
    *,
    sockets: Iterable[str] = SENSITIVE_SOCKET_PATHS,
) -> list[str]:
    """Return the argv's mount specs that expose a sensitive socket.

    Path-aware: a parent-directory mount (e.g. ``-v /var/run:/var/run``) that
    contains the socket is caught, not only an exact ``/var/run/docker.sock``
    spec, while a lookalike sibling path (``/var/running``) is not.
    """

    socket_paths = tuple(sockets)
    return [
        mount for mount in iter_volume_mounts(argv) if _mount_exposes_socket(mount, socket_paths)
    ]


def has_socket_mount(
    argv: Iterable[str],
    *,
    sockets: Iterable[str] = SENSITIVE_SOCKET_PATHS,
) -> bool:
    """Return ``True`` if the argv bind-mounts the docker/dstack socket."""

    return bool(socket_mount_specs(argv, sockets=sockets))


def assert_no_socket_mounts(
    argv: Iterable[str],
    *,
    sockets: Iterable[str] = SENSITIVE_SOCKET_PATHS,
) -> None:
    """Raise :class:`DoodSocketExposureError` if a task launch mounts a socket."""

    argv = list(argv)
    exposed = socket_mount_specs(argv, sockets=sockets)
    if exposed:
        raise DoodSocketExposureError(
            "refusing to launch a task container that mounts a privileged socket: "
            + ", ".join(exposed)
        )


def spawns_inner_dockerd(argv: Iterable[str]) -> bool:
    """Return ``True`` if ``argv`` would start an inner ``dockerd`` daemon (DinD).

    DooD launches siblings on the existing guest daemon; the orchestrator must
    never bring up its own ``dockerd``. Detects both a direct ``dockerd``
    invocation and a ``dockerd`` embedded in a shell snippet. Note ``docker``
    (the client) is not ``dockerd`` (the daemon).
    """

    for token in argv:
        stripped = token.strip()
        if stripped == "dockerd" or stripped.endswith("/dockerd"):
            return True
        if "dockerd" in token and any(sep in token for sep in (" ", "\n", ";", "&", "\t")):
            return True
    return False


__all__ = [
    "DOCKER_SOCKET_PATH",
    "DOOD_DOCKER_HOST",
    "DSTACK_SOCKET_PATH",
    "SENSITIVE_SOCKET_PATHS",
    "DoodConfigError",
    "DoodSocketExposureError",
    "assert_no_socket_mounts",
    "dood_docker_argv",
    "dood_docker_env",
    "has_socket_mount",
    "is_tcp_docker_host",
    "iter_volume_mounts",
    "resolve_docker_host",
    "socket_mount_specs",
    "spawns_inner_dockerd",
]
