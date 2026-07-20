"""Live-subset registry refs: map a golden task_id to a PULLABLE ``repo@sha256``.

The frozen golden manifest (``golden/dataset-digest.json``) pins each
Terminal-Bench task by a *bare content digest* (``harbor_registry_ref =
"sha256:<64hex>"``) with NO repository, so an in-CVM DooD orchestrator cannot
``docker pull`` it. For a live smoke E2E a small deterministic subset of task
images is published to the miner's public Docker Hub namespace as pullable,
digest-pinned refs and recorded in a SEPARATE side manifest
(``golden/live-registry-refs.json``). That side manifest never touches
``dataset-digest.json`` -- the frozen content digests and the canonical
measurement stay byte-identical.

Resolution is **opt-in and fail-closed**: with no live manifest configured
(no explicit path, no env var) callers get NO live refs and fall back to the
existing per-task behavior, so flag-off / offline runs are byte-identical. When a
manifest IS configured for the live subset, a task in it resolves to its pullable
``repo@sha256`` ref; a task absent from it still uses the legacy behavior.

Import-light (stdlib only) so it loads in the lean canonical image alongside the
DooD orchestrator.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Filename of the side manifest (co-located with ``dataset-digest.json``).
LIVE_REGISTRY_FILENAME = "live-registry-refs.json"

#: Env var naming the live-registry side manifest path. Unset => no live refs
#: (byte-identical legacy behavior). The deploy sets it (to the mounted golden
#: dir path) only for a live subset run.
LIVE_REGISTRY_ENV = "CHALLENGE_OWN_RUNNER_LIVE_REGISTRY"

#: Schema tag of the side manifest.
LIVE_REGISTRY_SCHEMA = "harbor-independence/live-registry-refs@1"

#: Repo default path (``<repo>/golden/live-registry-refs.json``) relative to this
#: file (``src/agent_challenge/canonical/live_registry.py``).
DEFAULT_LIVE_REGISTRY_PATH = Path(__file__).resolve().parents[3] / "golden" / LIVE_REGISTRY_FILENAME

# A PULLABLE ref is repository-qualified (contains a '/') AND digest-pinned by an
# immutable ``@sha256:<64hex>``. This deliberately rejects a bare content digest
# (``sha256:<hex>`` -- the golden manifest's non-pullable form), a floating tag,
# and an un-namespaced ``name@sha256`` so only a real registry ref is accepted.
_PULLABLE_REF_RE = re.compile(r"^(?=[^@]*/)[A-Za-z0-9][\w.\-/:]*@sha256:[0-9a-f]{64}$")


class LiveRegistryError(ValueError):
    """The live-registry side manifest is missing, malformed, or not pullable."""


def _bare_task_name(task_id: str) -> str:
    """Map a (possibly dataset-prefixed) task id to its bare manifest key."""

    return task_id.rsplit("/", 1)[-1]


def is_pullable_ref(ref: Any) -> bool:
    """True iff ``ref`` is a repository-qualified, digest-pinned registry ref."""

    return isinstance(ref, str) and bool(_PULLABLE_REF_RE.match(ref))


def assert_pullable_ref(ref: Any, *, what: str = "registry ref") -> str:
    """Return ``ref`` if it is a pullable ``repo@sha256`` ref, else raise.

    Rejects a bare content digest (``sha256:...``), a floating tag, and an
    un-namespaced ref so the in-CVM orchestrator can never be handed something it
    cannot ``docker pull``.
    """

    if not is_pullable_ref(ref):
        raise LiveRegistryError(
            f"{what} must be a pullable repository-qualified digest ref "
            f"(repo/name@sha256:<64hex>), got {ref!r}"
        )
    return ref  # type: ignore[return-value]


@dataclass(frozen=True)
class LiveRegistry:
    """Parsed live-registry side manifest.

    ``task_refs`` is keyed by the bare task name (matching the golden manifest /
    cache layout) so a dataset-prefixed id resolves to the same entry.
    ``orchestrator_image`` is the pullable canonical image ref the deploy path
    pins (or ``None`` when the manifest records only task refs).
    """

    orchestrator_image: str | None
    task_refs: dict[str, str]
    raw: dict[str, Any]

    def resolve_task_image(self, task_id: str) -> str | None:
        """Return the pullable ref for ``task_id`` (bare or prefixed), else None.

        Fail-closed: an unknown task returns ``None`` so the caller keeps its
        existing per-task behavior.
        """

        return self.task_refs.get(task_id) or self.task_refs.get(_bare_task_name(task_id))

    def __bool__(self) -> bool:
        return bool(self.task_refs) or self.orchestrator_image is not None


def _ref_from_entry(task_id: str, entry: Any) -> str:
    """Extract + validate the pullable ref from a manifest task entry."""

    if isinstance(entry, str):
        ref = entry
    elif isinstance(entry, Mapping):
        ref = entry.get("registry_ref")
    else:
        raise LiveRegistryError(f"live-registry task {task_id!r} is not a string or mapping")
    return assert_pullable_ref(ref, what=f"live-registry ref for task {task_id!r}")


def parse_live_registry(data: Mapping[str, Any]) -> LiveRegistry:
    """Parse + validate a live-registry side-manifest document.

    Every task ref must be a pullable ``repo@sha256`` ref (a bare content digest
    or floating tag is rejected). ``orchestrator_image``, when present, must be
    pullable too. Task keys are normalized to their bare name.
    """

    if not isinstance(data, Mapping):
        raise LiveRegistryError("live-registry manifest is not a JSON object")

    tasks = data.get("tasks", {})
    if not isinstance(tasks, Mapping):
        raise LiveRegistryError("live-registry manifest has no 'tasks' mapping")

    task_refs: dict[str, str] = {}
    for task_id, entry in tasks.items():
        task_refs[_bare_task_name(str(task_id))] = _ref_from_entry(str(task_id), entry)

    orchestrator = data.get("orchestrator_image")
    if orchestrator is not None:
        orchestrator = assert_pullable_ref(orchestrator, what="orchestrator_image")

    return LiveRegistry(orchestrator_image=orchestrator, task_refs=task_refs, raw=dict(data))


def load_live_registry(path: Path | str) -> LiveRegistry:
    """Load + validate a live-registry side manifest from disk (fail-closed)."""

    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LiveRegistryError(f"live-registry manifest not found: {p}") from exc
    except ValueError as exc:
        raise LiveRegistryError(f"invalid live-registry manifest {p}: {exc}") from exc
    return parse_live_registry(data)


def resolve_live_registry(
    *,
    path: Path | str | None = None,
    env: Mapping[str, str] | None = None,
) -> LiveRegistry | None:
    """Resolve the configured live registry (opt-in), else ``None``.

    Precedence: an explicit ``path`` wins; otherwise the ``env`` mapping's
    :data:`LIVE_REGISTRY_ENV` value (if set) is used; otherwise ``None`` (no live
    registry configured => byte-identical legacy behavior). A configured-but-
    broken manifest raises :class:`LiveRegistryError` (a misconfiguration is
    visible, never silently downgraded).
    """

    configured = path
    if configured is None and env is not None:
        configured = (env.get(LIVE_REGISTRY_ENV) or "").strip() or None
    if configured is None:
        return None
    return load_live_registry(configured)


def resolve_live_registry_refs(
    *,
    path: Path | str | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Resolve the live task-image refs (bare-keyed), or ``{}`` when unconfigured.

    Thin wrapper over :func:`resolve_live_registry` returning just the
    ``task_id -> pullable ref`` mapping the DooD builder consults; ``{}`` when no
    manifest is configured so offline / flag-off resolution is byte-identical.
    """

    registry = resolve_live_registry(path=path, env=env)
    return dict(registry.task_refs) if registry is not None else {}


__all__ = [
    "DEFAULT_LIVE_REGISTRY_PATH",
    "LIVE_REGISTRY_ENV",
    "LIVE_REGISTRY_FILENAME",
    "LIVE_REGISTRY_SCHEMA",
    "LiveRegistry",
    "LiveRegistryError",
    "assert_pullable_ref",
    "is_pullable_ref",
    "load_live_registry",
    "parse_live_registry",
    "resolve_live_registry",
    "resolve_live_registry_refs",
]
