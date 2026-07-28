#!/usr/bin/env python3
"""Owned-CVM teardown selection for AC staging (fail-closed).

Staging must never delete a Phala CVM unless this run (or the local work dir)
provably owns it. Account-wide sweeps are opt-in only and never the default.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def normalize_cvm_id(raw: str) -> str:
    return raw.strip()


def load_owned_ids(*paths: Path) -> list[str]:
    """Load unique CVM ids from track files (one id per line). Order preserved."""
    seen: set[str] = set()
    out: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            cid = normalize_cvm_id(line)
            if not cid or cid.startswith("#"):
                continue
            if cid in seen:
                continue
            seen.add(cid)
            out.append(cid)
    return out


def select_teardown_ids(
    *,
    owned_ids: list[str],
    account_ids: list[str] | None = None,
    account_sweep: bool = False,
) -> tuple[list[str], list[str]]:
    """Return (to_delete, rejected_foreign).

    Default: delete only owned_ids. Foreign account ids are never selected.
    When account_sweep=True, account_ids not already owned are still rejected
    from the automatic path — the caller must pass them as owned first. This
    keeps "ownership" as the sole delete criterion even under the opt-in flag.
    The account_sweep flag only controls whether missing ownership is a hard
    error vs a loud warning when the operator intended a full account clean;
    it does NOT expand the delete set beyond owned_ids.
    """
    del account_ids, account_sweep  # ownership is the only delete criterion
    owned = [normalize_cvm_id(i) for i in owned_ids if normalize_cvm_id(i)]
    # Dedup preserve order
    seen: set[str] = set()
    to_delete: list[str] = []
    for cid in owned:
        if cid in seen:
            continue
        seen.add(cid)
        to_delete.append(cid)
    return to_delete, []


def assert_id_owned(cvm_id: str, owned_ids: list[str]) -> None:
    """Hard guard: raise SystemExit if cvm_id is not in the owned set."""
    cid = normalize_cvm_id(cvm_id)
    owned = {normalize_cvm_id(i) for i in owned_ids if normalize_cvm_id(i)}
    if not cid:
        raise SystemExit("refusing delete: empty cvm id")
    if cid not in owned:
        raise SystemExit(
            f"refusing delete of foreign CVM id {cid!r}: not in owned track "
            f"({len(owned)} owned)"
        )


def plan_teardown(
    *,
    owned_paths: list[Path],
    account_ids: list[str] | None = None,
    account_sweep: bool = False,
) -> dict[str, object]:
    owned = load_owned_ids(*owned_paths)
    account = [normalize_cvm_id(i) for i in (account_ids or []) if normalize_cvm_id(i)]
    to_delete, rejected = select_teardown_ids(
        owned_ids=owned,
        account_ids=account,
        account_sweep=account_sweep,
    )
    foreign_on_account = [i for i in account if i not in set(to_delete)]
    return {
        "owned_ids": owned,
        "account_ids": account,
        "account_sweep": account_sweep,
        "will_delete": to_delete,
        "will_not_delete_foreign": foreign_on_account,
        "rejected": rejected,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan owned-only CVM teardown (never selects foreign ids)."
    )
    parser.add_argument(
        "--owned-file",
        action="append",
        default=[],
        dest="owned_files",
        help="Path to owned CVM id track file (repeatable).",
    )
    parser.add_argument(
        "--account-ids-json",
        default="",
        help='Optional JSON list or {"ids":[...]} of account CVM ids (for reporting only).',
    )
    parser.add_argument(
        "--account-sweep",
        action="store_true",
        help="Opt-in flag (loud). Still does NOT expand deletes beyond owned files.",
    )
    parser.add_argument(
        "--check-id",
        default="",
        help="Exit non-zero if this id is not owned (hard guard).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan JSON and exit 0 (default behavior of this tool).",
    )
    args = parser.parse_args(argv)

    owned_paths = [Path(p) for p in args.owned_files]
    account_ids: list[str] = []
    if args.account_ids_json:
        raw = json.loads(args.account_ids_json)
        if isinstance(raw, list):
            account_ids = [str(x) for x in raw]
        elif isinstance(raw, dict):
            account_ids = [str(x) for x in (raw.get("ids") or [])]
        else:
            raise SystemExit("account-ids-json must be a list or object with ids")

    if args.account_sweep:
        print(
            "WARNING: --account-sweep is set but deletes remain owned-only; "
            "foreign account CVMs are never selected.",
            file=sys.stderr,
        )

    plan = plan_teardown(
        owned_paths=owned_paths,
        account_ids=account_ids,
        account_sweep=args.account_sweep,
    )

    if args.check_id:
        assert_id_owned(args.check_id, list(plan["owned_ids"]))  # type: ignore[arg-type]
        print(json.dumps({"ok": True, "id": normalize_cvm_id(args.check_id)}))
        return 0

    print(json.dumps(plan, indent=2, sort_keys=True))
    # dry-run is the only mode of this helper
    _ = args.dry_run
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
