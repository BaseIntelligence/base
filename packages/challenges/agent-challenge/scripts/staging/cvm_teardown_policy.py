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


def account_item_identifiers(item: dict) -> set[str]:
    """All identifiers that may appear in deploy acks or GET /cvms rows."""
    out: set[str] = set()
    for key in ("id", "cvm_id", "vm_uuid", "uuid", "instance_id"):
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            out.add(val.strip())
    return out


def resolve_delete_ids(
    *,
    owned_ids: list[str],
    account_items: list[dict] | None = None,
    account_ids: list[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Map owned track entries to API delete targets.

    Deploy acks often record vm_uuid (UUID) while GET /cvms returns id=cvm_*.
    Returns (api_ids_to_delete, unresolved_owned, foreign_account_api_ids).
    """
    owned = [normalize_cvm_id(i) for i in owned_ids if normalize_cvm_id(i)]
    owned_set = set(owned)

    items = list(account_items or [])
    if not items and account_ids:
        # ids-only fallback: treat each string as an API id with no alias map
        items = [{"id": normalize_cvm_id(i)} for i in account_ids if normalize_cvm_id(i)]

    api_ids: list[str] = []
    seen_api: set[str] = set()
    matched_owned: set[str] = set()

    for item in items:
        if not isinstance(item, dict):
            continue
        idents = account_item_identifiers(item)
        if not idents & owned_set:
            continue
        api_id = ""
        for key in ("id", "cvm_id"):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                api_id = val.strip()
                break
        if not api_id:
            # last resort: any owned ident that looks like cvm_*
            for ident in sorted(idents):
                if ident.startswith("cvm_"):
                    api_id = ident
                    break
        if not api_id:
            # still owned — try deleting by the owned uuid itself
            for ident in sorted(idents & owned_set):
                api_id = ident
                break
        if api_id and api_id not in seen_api:
            seen_api.add(api_id)
            api_ids.append(api_id)
        matched_owned |= idents & owned_set

    # Owned ids that never appeared on the account listing: still attempt delete
    # by the tracked token (selfdeploy teardown may accept uuid).
    unresolved = [o for o in owned if o not in matched_owned]
    for o in unresolved:
        if o not in seen_api:
            seen_api.add(o)
            api_ids.append(o)

    all_account_api = []
    for item in items:
        if not isinstance(item, dict):
            continue
        for key in ("id", "cvm_id"):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                all_account_api.append(val.strip())
                break
    foreign = [i for i in all_account_api if i not in seen_api]
    return api_ids, unresolved, foreign


def select_teardown_ids(
    *,
    owned_ids: list[str],
    account_ids: list[str] | None = None,
    account_items: list[dict] | None = None,
    account_sweep: bool = False,
) -> tuple[list[str], list[str]]:
    """Return (to_delete_api_ids, rejected_foreign_api_ids).

    Default: delete only CVMs owned by track entries (matched via id/cvm_id/vm_uuid).
    Foreign account CVMs are never selected. account_sweep does not expand the set.
    """
    del account_sweep
    to_delete, _unresolved, foreign = resolve_delete_ids(
        owned_ids=owned_ids,
        account_items=account_items,
        account_ids=account_ids,
    )
    return to_delete, foreign


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
    account_items: list[dict] | None = None,
    account_sweep: bool = False,
) -> dict[str, object]:
    owned = load_owned_ids(*owned_paths)
    account = [normalize_cvm_id(i) for i in (account_ids or []) if normalize_cvm_id(i)]
    to_delete, foreign = select_teardown_ids(
        owned_ids=owned,
        account_ids=account,
        account_items=account_items,
        account_sweep=account_sweep,
    )
    return {
        "owned_ids": owned,
        "account_ids": account,
        "account_sweep": account_sweep,
        "will_delete": to_delete,
        "will_not_delete_foreign": foreign,
        "rejected": [],
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
        help='Optional JSON list, {"ids":[...]}, or full GET /cvms payload with items.',
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
    account_items: list[dict] = []
    if args.account_ids_json:
        raw = json.loads(args.account_ids_json)
        if isinstance(raw, list):
            if raw and isinstance(raw[0], dict):
                account_items = [x for x in raw if isinstance(x, dict)]
                account_ids = [
                    str(x.get("id") or x.get("cvm_id") or "")
                    for x in account_items
                    if isinstance(x, dict)
                ]
                account_ids = [i for i in account_ids if i]
            else:
                account_ids = [str(x) for x in raw]
        elif isinstance(raw, dict):
            if isinstance(raw.get("items"), list):
                account_items = [x for x in raw["items"] if isinstance(x, dict)]
            elif isinstance(raw.get("data"), list):
                account_items = [x for x in raw["data"] if isinstance(x, dict)]
            elif isinstance(raw.get("cvms"), list):
                account_items = [x for x in raw["cvms"] if isinstance(x, dict)]
            if account_items:
                account_ids = [
                    str(x.get("id") or x.get("cvm_id") or "")
                    for x in account_items
                ]
                account_ids = [i for i in account_ids if i]
            else:
                account_ids = [str(x) for x in (raw.get("ids") or [])]
        else:
            raise SystemExit("account-ids-json must be a list or object with ids/items")

    if args.account_sweep:
        print(
            "WARNING: --account-sweep is set but deletes remain owned-only; "
            "foreign account CVMs are never selected.",
            file=sys.stderr,
        )

    plan = plan_teardown(
        owned_paths=owned_paths,
        account_ids=account_ids,
        account_items=account_items or None,
        account_sweep=args.account_sweep,
    )

    if args.check_id:
        # Allow delete if id is owned OR resolves as the API id of an owned vm_uuid.
        cid = normalize_cvm_id(args.check_id)
        owned_list = list(plan["owned_ids"])  # type: ignore[arg-type]
        if cid in {normalize_cvm_id(i) for i in owned_list}:
            print(json.dumps({"ok": True, "id": cid}))
            return 0
        will = set(plan.get("will_delete") or [])  # type: ignore[arg-type]
        if cid in will:
            print(json.dumps({"ok": True, "id": cid, "resolved_from_owned": True}))
            return 0
        assert_id_owned(cid, owned_list)
        print(json.dumps({"ok": True, "id": cid}))
        return 0

    print(json.dumps(plan, indent=2, sort_keys=True))
    _ = args.dry_run
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
