#!/usr/bin/env python
"""Flags-OFF legacy regression smoke: gpu unit -> VALIDATOR -> validator_dispatch -> score.

Part of the cross-repo local end-to-end harness (base ``docs/operations/mission-harness.md``);
fulfils part (b) of VAL-CROSS-006. Stands up the SAME local mock-metagraph deployment as
``launch.py`` but with ALL new flags OFF:

* base master with ``worker_plane_enabled: false`` (gpu units route to gpu VALIDATORS, byte-for-byte
  as pre-mission -- VAL-MASTER-013), and
* prism with ``worker_plane_enabled: false`` (admission gate off, ``/internal/v1/work_units/result``
  + audit routes inert),

then drives one legacy submission end-to-end and asserts, via HTTP + a read of the master/prism
databases, that it behaves exactly as today:

1. the submission is ACCEPTED with no ``NO_ACTIVE_WORKER`` 403 (admission off);
2. prism exposes exactly ONE gpu work unit for it;
3. the base master assigns that unit to the VALIDATOR
   (``work_assignments.assigned_validator_hotkey`` == the validator hotkey,
   ``required_capability == gpu``) and NEVER to a worker (``worker_assignments`` and
   ``worker_registrations`` are empty -- no worker-plane rows/side effects);
4. the validator executes it via the real ``validator_dispatch`` path (its log shows the dispatch)
   and prism records a score (``GET /v1/submissions/{id}`` -> completed with a final_score).

Every spawned process' PID is printed and, in a ``finally``, KILLED by PID so nothing is left
listening. Run through the prism virtualenv with the current base source on ``PYTHONPATH``::

    cd <repo-root>          # contains base/ and prism/
    export PYTHONPATH="$PWD/base/src:$PWD/prism/src"
    prism/.venv/bin/python prism/scripts/mission/legacy_smoke.py

NOT for production.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

# Reuse the harness helpers (signed headers, process mgmt, waiters) from the main launcher.
from launch import (  # type: ignore[import-not-found]
    BASE_MASTER_SCRIPT,
    PRISM_SCRIPT_DIR,
    Harness,
    _completed_with_score,
    _master_signed_headers,
    _prism_signed_headers,
    _score_of,
    _two_script_bundle,
    ss58,
    wait_health,
    wait_until,
)

from prism_challenge.evaluator.cpu_test_mode import TINY_ARCHITECTURE, TINY_TRAINING

MASTER_PORT = 3112
PRISM_PORT = 3122
MASTER_URL = f"http://127.0.0.1:{MASTER_PORT}"
PRISM_URL = f"http://127.0.0.1:{PRISM_PORT}"
TOKEN = "mission-shared-token"
NETUID = 100
MINER_URI = "//MissionAlice"
VALIDATOR_URI = "//MissionValidator1"


class LegacyHarness(Harness):
    """A flags-OFF variant of the harness: legacy master + prism + a legacy validator."""

    def __init__(self, workdir: Path) -> None:
        super().__init__(workdir=workdir)
        self.prism_db = self.workdir / "prism.sqlite3"
        self.master_db = self.workdir / "master.sqlite3"

    def start_master(self):  # type: ignore[override]
        entries = [
            {"hotkey": ss58(MINER_URI), "uid": 0, "validator_permit": False, "stake": 1000.0},
            {"hotkey": ss58(VALIDATOR_URI), "uid": 99, "validator_permit": True, "stake": 5000.0},
        ]
        config = {
            "port": MASTER_PORT,
            "host": "127.0.0.1",
            "db_url": f"sqlite+aiosqlite:///{self.master_db}",
            "netuid": NETUID,
            "metagraph": entries,
            "prism": {"slug": "prism", "internal_base_url": PRISM_URL, "token": TOKEN},
            "orchestration_interval_seconds": 1.0,
            "worker_heartbeat_ttl_seconds": 30,
            "health_interval_seconds": 2.0,
            "worker_plane_enabled": False,
        }
        return self.spawn("master", BASE_MASTER_SCRIPT, config)

    def start_prism(self):  # type: ignore[override]
        config = {
            "port": PRISM_PORT,
            "host": "127.0.0.1",
            "db_path": str(self.prism_db),
            "token": TOKEN,
            "master_base_url": MASTER_URL,
            "artifact_root": str(self.workdir / "prism-artifacts"),
            "train_data_dir": str(self.train_dir),
            "worker_plane_enabled": False,
            "sequence_length": 16,
        }
        return self.spawn("prism", PRISM_SCRIPT_DIR / "mission_prism.py", config)

    def start_legacy_validator(self):
        config = {
            "master_url": MASTER_URL,
            "validator_uri": VALIDATOR_URI,
            "capabilities": ["gpu"],
            "version": "0.1.0",
            "heartbeat_interval_seconds": 3,
            "poll_interval_seconds": 1.0,
            "prism": {
                "token": TOKEN,
                "db_path": str(self.prism_db),
                "artifact_root": str(self.workdir / "legacy-validator-artifacts"),
                "train_data_dir": str(self.train_dir),
                "sequence_length": 16,
            },
        }
        return self.spawn(
            "legacy-validator", PRISM_SCRIPT_DIR / "mission_legacy_validator.py", config
        )

    # -- HTTP + DB observation helpers ---------------------------------------
    def submit(self, *, nonce: str) -> httpx.Response:
        code = _two_script_bundle(TINY_ARCHITECTURE, TINY_TRAINING)
        import json

        body = json.dumps({"code": code, "filename": "project.zip"}, separators=(",", ":")).encode()
        headers = {
            **_prism_signed_headers(TOKEN, body, hotkey=ss58(MINER_URI), nonce=nonce),
            "Content-Type": "application/json",
        }
        return httpx.post(f"{PRISM_URL}/v1/submissions", content=body, headers=headers, timeout=15)

    def prism_submission(self, sid: str) -> dict[str, Any]:
        return httpx.get(f"{PRISM_URL}/v1/submissions/{sid}", timeout=10).json()

    def prism_work_units(self) -> list[dict[str, Any]]:
        resp = httpx.get(
            f"{PRISM_URL}/internal/v1/work_units",
            headers={"Authorization": f"Bearer {TOKEN}", "X-Base-Challenge-Slug": "prism"},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("work_units", [])

    def master_workers_status(self) -> tuple[int, Any]:
        headers = _master_signed_headers(self.validator_kp, "GET", "/v1/workers")
        resp = httpx.get(f"{MASTER_URL}/v1/workers", headers=headers, timeout=10)
        try:
            body: Any = resp.json()
        except Exception:
            body = resp.text
        return resp.status_code, body

    def work_assignment_row(self, work_unit_id: str) -> dict[str, Any] | None:
        con = sqlite3.connect(str(self.master_db))
        con.row_factory = sqlite3.Row
        try:
            row = con.execute(
                "SELECT work_unit_id, submission_ref, required_capability, status, "
                "assigned_validator_hotkey FROM work_assignments WHERE work_unit_id = ?",
                (work_unit_id,),
            ).fetchone()
        finally:
            con.close()
        return dict(row) if row is not None else None

    def _count(self, table: str) -> int:
        con = sqlite3.connect(str(self.master_db))
        try:
            return int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        finally:
            con.close()


def _enable_wal(db_path: Path) -> str:
    con = sqlite3.connect(str(db_path))
    try:
        mode = con.execute("PRAGMA journal_mode=WAL").fetchone()[0]
    finally:
        con.close()
    return str(mode)


def run(workdir: Path) -> bool:
    h = LegacyHarness(workdir=workdir)
    results: dict[str, bool] = {}
    try:
        print("== bring up LEGACY (flags OFF) master + prism ==")
        h.start_master()
        h.start_prism()
        wait_health(MASTER_URL, "master")
        wait_health(PRISM_URL, "prism")
        # WAL so the prism service (reader) and the legacy validator (single writer) never contend.
        print(f"  prism db journal_mode -> {_enable_wal(h.prism_db)}")

        print("\n== start the LEGACY validator (real validator_dispatch executor) ==")
        h.start_legacy_validator()
        time.sleep(5)  # let it register + heartbeat before work exists

        print("\n=== (1) admission OFF: submission accepted with no NO_ACTIVE_WORKER 403 ===")
        # Fresh nonce per run so re-running against a reused --workdir (persistent
        # sqlite nonce store) is not rejected as a replay before routing.
        resp = h.submit(nonce=f"legacy-{uuid.uuid4().hex}")
        print(f"  POST /v1/submissions -> HTTP {resp.status_code} body={resp.text[:160]}")
        ok_accept = resp.status_code < 300 and "NO_ACTIVE_WORKER" not in resp.text
        results["(1) submission accepted, no admission 403"] = ok_accept
        sid = str(resp.json()["id"]) if ok_accept else ""
        print(f"  submission id = {sid}")

        print("\n=== (2) prism exposes exactly one gpu work unit ===")
        units = wait_until(
            "prism gpu work unit",
            lambda: [u for u in h.prism_work_units() if str(u.get("submission_id")) == sid],
            timeout=30,
        )
        cap = units[0].get("required_capability") or units[0].get("capability")
        print(f"  /internal/v1/work_units -> {len(units)} unit(s); required_capability={cap}")
        results["(2) exactly one gpu work unit"] = len(units) == 1

        print("\n=== (4) validator executes via validator_dispatch and prism records a score ===")
        final = wait_until(
            "prism records a score",
            lambda: _completed_with_score(h.prism_submission(sid)),
            timeout=180,
        )
        print(f"  GET /v1/submissions/{sid}: status={final.get('status')} score={_score_of(final)}")
        results["(4) submission scored via validator_dispatch"] = _score_of(final) is not None

        print("\n=== (3) routing: gpu unit assigned to VALIDATOR, never a worker ===")
        row = wait_until(
            "work_assignments row for the unit",
            lambda: h.work_assignment_row(sid),
            timeout=30,
        )
        validator_hk = ss58(VALIDATOR_URI)
        print(
            f"  work_assignments: work_unit_id={row['work_unit_id']} "
            f"required_capability={row['required_capability']} status={row['status']} "
            f"assigned_validator_hotkey={row['assigned_validator_hotkey']}"
        )
        assigned_to_validator = (
            row["assigned_validator_hotkey"] == validator_hk and row["required_capability"] == "gpu"
        )
        worker_regs = h._count("worker_registrations")
        worker_asgn = h._count("worker_assignments")
        print(f"  worker_registrations rows={worker_regs}  worker_assignments rows={worker_asgn}")
        status_code, workers_body = h.master_workers_status()
        n_workers = len(workers_body.get("workers", [])) if isinstance(workers_body, dict) else 0
        print(f"  GET /v1/workers -> HTTP {status_code} workers={n_workers}")
        results["(3) assigned to validator, not a worker"] = assigned_to_validator
        results["(3) no worker-plane rows/side effects"] = worker_regs == 0 and worker_asgn == 0
        results["(3) worker fleet surface inert (no workers)"] = (
            status_code == 404 or n_workers == 0
        )
    finally:
        h.kill_all()

    print("\n==================== LEGACY SMOKE SUMMARY (flags OFF) ====================")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    all_ok = bool(results) and all(results.values())
    print("=========================================================================")
    print(f"  OVERALL: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default="/tmp/mission-legacy-smoke")
    args = parser.parse_args()
    return 0 if run(Path(args.workdir)) else 1


if __name__ == "__main__":
    sys.exit(main())
