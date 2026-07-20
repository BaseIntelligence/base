#!/usr/bin/env python3
"""Populate ``docker/canonical/live-task-cache`` from the pinned harbor task package cache.

Eval guests read task defs only from ``/opt/agent-challenge/task-cache`` (COPY of
``docker/canonical/live-task-cache``). Prepare/select draws from
``TERMINAL_BENCH_2_1_FALLBACK_TASK_IDS`` (30) and/or the full frozen
``golden/dataset-digest.json`` (89). This script copies bare task trees that
already match frozen digests out of the local harbor package cache
(``~/.cache/harbor/tasks/packages/terminal-bench``, Task-1 acquisition output).

Hard rules:
* **No network invent.** Refuse if harbor source or frozen digest is missing.
* **Fail closed on digest mismatch.** Skip/raise rather than bake stale trees.
* **Bare layout.** Guest expects ``<cache_root>/<bare_name>/task.toml`` (no
  content-hash intermediate dir) so Dockerfile COPY is ready for preflight.

Usage::

    uv run python scripts/populate_live_task_cache.py           # full digest set
    uv run python scripts/populate_live_task_cache.py --fallback-only
    uv run python scripts/populate_live_task_cache.py --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from agent_challenge.evaluation.benchmarks import (  # noqa: E402
    TERMINAL_BENCH_2_1_DIGEST_PATH,
    TERMINAL_BENCH_2_1_FALLBACK_TASK_IDS,
)
from agent_challenge.evaluation.own_runner.taskdefs import (  # noqa: E402
    DEFAULT_CACHE_ROOT,
    DigestMismatch,
    TaskDefError,
    bare_task_name,
    load_dataset_digest,
    load_task_from_manifest,
    resolve_task_root,
)

DEFAULT_DEST = _REPO_ROOT / "docker" / "canonical" / "live-task-cache"


def _selected_bare_ids(*, fallback_only: bool) -> list[str]:
    if fallback_only:
        return [bare_task_name(t) for t in TERMINAL_BENCH_2_1_FALLBACK_TASK_IDS]
    manifest = load_dataset_digest(TERMINAL_BENCH_2_1_DIGEST_PATH)
    return sorted(manifest["tasks"])


def populate(
    *,
    source_root: Path,
    dest_root: Path,
    bare_ids: list[str],
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, str]:
    """Copy/verify selected bare tasks into dest_root. Returns bare_id -> status."""

    manifest = load_dataset_digest(TERMINAL_BENCH_2_1_DIGEST_PATH)
    results: dict[str, str] = {}
    dest_root.mkdir(parents=True, exist_ok=True)

    for bare in bare_ids:
        if bare not in manifest["tasks"]:
            results[bare] = "absent_from_digest"
            continue
        try:
            src = resolve_task_root(source_root, bare)
            # Verify source against frozen digest before any write.
            load_task_from_manifest(
                src,
                task_id=bare,
                digest_manifest=manifest,
                verify_digest=True,
            )
        except TaskDefError as exc:
            results[bare] = f"source_error:{type(exc).__name__}:{exc}"
            continue

        dest = dest_root / bare
        if dest.exists():
            try:
                load_task_from_manifest(
                    dest,
                    task_id=bare,
                    digest_manifest=manifest,
                    verify_digest=True,
                )
                if not force:
                    results[bare] = "already_present"
                    continue
            except (DigestMismatch, TaskDefError):
                if dry_run:
                    results[bare] = "would_replace_mismatch"
                    continue
                shutil.rmtree(dest)
            else:
                if force and not dry_run:
                    shutil.rmtree(dest)
                elif force and dry_run:
                    results[bare] = "would_force_replace"
                    continue

        if dry_run:
            results[bare] = "would_copy"
            continue

        shutil.copytree(src, dest, symlinks=False)
        # Post-copy digest gate (fail closed).
        load_task_from_manifest(
            dest,
            task_id=bare,
            digest_manifest=manifest,
            verify_digest=True,
        )
        results[bare] = "copied"

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
        help=f"Pinned harbor package cache root (default: {DEFAULT_CACHE_ROOT})",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Live task-cache destination (default: {DEFAULT_DEST})",
    )
    parser.add_argument(
        "--fallback-only",
        action="store_true",
        help="Populate only TERMINAL_BENCH_2_1_FALLBACK_TASK_IDS (30), not full 89",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report actions without writing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace destination trees even when digests already match",
    )
    args = parser.parse_args(argv)

    if not args.source.is_dir():
        print(f"error: harbor source cache missing: {args.source}", file=sys.stderr)
        print(
            "Populate via Task-1 acquisition (no network invent here). "
            "Expected layout: <source>/<bare>/<content_hash>/task.toml",
            file=sys.stderr,
        )
        return 2
    if not TERMINAL_BENCH_2_1_DIGEST_PATH.is_file():
        print(f"error: frozen digest missing: {TERMINAL_BENCH_2_1_DIGEST_PATH}", file=sys.stderr)
        return 2

    bare_ids = _selected_bare_ids(fallback_only=args.fallback_only)
    results = populate(
        source_root=args.source,
        dest_root=args.dest,
        bare_ids=bare_ids,
        dry_run=args.dry_run,
        force=args.force,
    )

    counts: dict[str, int] = {}
    errors: list[str] = []
    for bare, status in sorted(results.items()):
        counts[status.split(":")[0]] = counts.get(status.split(":")[0], 0) + 1
        if status.startswith("source_error") or status == "absent_from_digest":
            errors.append(f"{bare}: {status}")
            print(f"FAIL  {bare}: {status}", file=sys.stderr)
        else:
            print(f"OK    {bare}: {status}")

    print(
        f"summary selected={len(bare_ids)} "
        + " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    )
    if errors:
        print(f"error: {len(errors)} task(s) failed source verification", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
