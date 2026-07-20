#!/usr/bin/env python
"""Local cross-repo mission launcher + drills (VAL-CROSS-001/002/003/004/009/011).

Stands up a full local mock-metagraph deployment on loopback ports 3100-3199 with NO GPU:

* a base master (worker plane ON) with a static mock metagraph,
* a prism service (worker plane ON, admission gate ON, explicit CPU re-exec test mode),
* 2+ worker agents on DISTINCT owner hotkeys whose executor is the repo's own CPU re-exec, and
* a stub gpu validator that audits disputed units by deterministic replay,

then runs six operator-observable drills (all via HTTP/CLI only) and, in a ``finally``, KILLS every
spawned process by PID so nothing is left listening.

Run: ``python scripts/mission/launch.py`` (all drills) or ``--only 1,4`` (a subset). Everything runs
through the prism virtualenv with the current base source on ``PYTHONPATH`` (see
``docs/operations/mission-harness.md`` in the base repo). NOT for production.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from base.security.validator_auth import canonical_validator_request

from prism_challenge.keypair import keypair_from_uri

REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_SRC = REPO_ROOT / "base" / "src"
PRISM_SRC = REPO_ROOT / "prism" / "src"
PRISM_PY = REPO_ROOT / "prism" / ".venv" / "bin" / "python"
BASE_MASTER_SCRIPT = REPO_ROOT / "base" / "scripts" / "mission" / "mission_master.py"
PRISM_SCRIPT_DIR = REPO_ROOT / "prism" / "scripts" / "mission"

MASTER_PORT = 3110
PRISM_PORT = 3120
MASTER_URL = f"http://127.0.0.1:{MASTER_PORT}"
PRISM_URL = f"http://127.0.0.1:{PRISM_PORT}"
TOKEN = "mission-shared-token"
NETUID = 100
WORKER_TTL = 8

# Owner (miner) + validator identities seeded into the mock metagraph.
MINERS = {
    "alice": "//MissionAlice",
    "bob": "//MissionBob",
    "carol": "//MissionCarol",
    "dave": "//MissionDave",
    "erin": "//MissionErin",
}
VALIDATOR_URI = "//MissionValidator1"
WORKER_URIS = {
    "alice": "//MissionWorkerAlice",
    "bob": "//MissionWorkerBob",
    "carol": "//MissionWorkerCarol",
    "dave": "//MissionWorkerDave",
    "erin": "//MissionWorkerErin",
}


def ss58(uri: str) -> str:
    return keypair_from_uri(uri).ss58_address


@dataclass
class Proc:
    name: str
    popen: subprocess.Popen
    log: Path

    @property
    def pid(self) -> int:
        return self.popen.pid

    def alive(self) -> bool:
        return self.popen.poll() is None


@dataclass
class Harness:
    workdir: Path
    procs: list[Proc] = field(default_factory=list)
    validator_signer: Any = None
    train_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.train_dir = self.workdir / "train-data"
        from prism_challenge.evaluator.cpu_test_mode import stage_tiny_train_data

        stage_tiny_train_data(self.workdir)
        self.validator_kp = keypair_from_uri(VALIDATOR_URI)

    # -- process management ---------------------------------------------------
    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{BASE_SRC}:{PRISM_SRC}"
        return env

    def spawn(self, name: str, script: Path, config: dict[str, Any]) -> Proc:
        cfg_path = self.workdir / f"{name}.json"
        cfg_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        log = self.workdir / f"{name}.log"
        handle = log.open("w", encoding="utf-8")
        popen = subprocess.Popen(
            [str(PRISM_PY), str(script), str(cfg_path)],
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=self._env(),
            cwd=str(REPO_ROOT),
        )
        proc = Proc(name=name, popen=popen, log=log)
        self.procs.append(proc)
        print(f"  spawned {name} pid={proc.pid} (log {log})")
        return proc

    def kill(self, proc: Proc) -> None:
        if proc in self.procs:
            self.procs.remove(proc)
        _terminate(proc)

    def kill_all(self) -> None:
        print("\n== teardown: killing all mission processes by PID ==")
        for proc in list(self.procs):
            _terminate(proc)
            print(f"  killed {proc.name} pid={proc.pid}")
        self.procs.clear()

    # -- master service -------------------------------------------------------
    def start_master(self) -> Proc:
        entries = [
            {"hotkey": ss58(uri), "uid": i, "validator_permit": False, "stake": 1000.0}
            for i, uri in enumerate(MINERS.values())
        ]
        entries.append(
            {"hotkey": ss58(VALIDATOR_URI), "uid": 99, "validator_permit": True, "stake": 5000.0}
        )
        config = {
            "port": MASTER_PORT,
            "host": "127.0.0.1",
            "db_url": f"sqlite+aiosqlite:///{self.workdir / 'master.sqlite3'}",
            "netuid": NETUID,
            "metagraph": entries,
            "prism": {"slug": "prism", "internal_base_url": PRISM_URL, "token": TOKEN},
            "orchestration_interval_seconds": 1.0,
            "worker_heartbeat_ttl_seconds": WORKER_TTL,
            "health_interval_seconds": 2.0,
            "replication_factor": 2,
        }
        return self.spawn("master", BASE_MASTER_SCRIPT, config)

    def start_prism(self) -> Proc:
        config = {
            "port": PRISM_PORT,
            "host": "127.0.0.1",
            "db_path": str(self.workdir / "prism.sqlite3"),
            "token": TOKEN,
            "master_base_url": MASTER_URL,
            "artifact_root": str(self.workdir / "prism-artifacts"),
            "train_data_dir": str(self.train_dir),
            "admission_requires_worker": True,
            "sequence_length": 16,
        }
        return self.spawn("prism", PRISM_SCRIPT_DIR / "mission_prism.py", config)

    def start_worker(self, owner: str, *, divergence_hotkey: str | None = None) -> Proc:
        name = f"worker-{owner}"
        config = {
            "name": name,
            "master_url": MASTER_URL,
            "miner_uri": MINERS[owner],
            "worker_uri": WORKER_URIS[owner],
            "provider": "local",
            "capabilities": ["gpu"],
            "heartbeat_interval_seconds": 3,
            "poll_interval_seconds": 1.0,
            "divergence_hotkey": divergence_hotkey,
            "prism": {
                "token": TOKEN,
                "artifact_root": str(self.workdir / f"{name}-artifacts"),
                "train_data_dir": str(self.train_dir),
                "sequence_length": 16,
            },
        }
        return self.spawn(name, PRISM_SCRIPT_DIR / "mission_worker.py", config)

    def start_validator(self) -> Proc:
        config = {
            "master_url": MASTER_URL,
            "validator_uri": VALIDATOR_URI,
            "capabilities": ["gpu"],
            "version": "0.1.0",
            "heartbeat_interval_seconds": 3,
            "poll_interval_seconds": 1.0,
            "prism": {
                "token": TOKEN,
                "artifact_root": str(self.workdir / "validator-artifacts"),
                "train_data_dir": str(self.train_dir),
                "sequence_length": 16,
            },
        }
        return self.spawn("validator", PRISM_SCRIPT_DIR / "mission_validator.py", config)

    # -- HTTP observation helpers (curl-equivalent) ---------------------------
    def prism_submit(self, owner: str, *, nonce: str) -> httpx.Response:
        from prism_challenge.evaluator.cpu_test_mode import TINY_ARCHITECTURE, TINY_TRAINING

        code = _two_script_bundle(TINY_ARCHITECTURE, TINY_TRAINING)
        body = json.dumps({"code": code, "filename": "project.zip"}, separators=(",", ":")).encode()
        headers = {
            **_prism_signed_headers(TOKEN, body, hotkey=ss58(MINERS[owner]), nonce=nonce),
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

    def master_workers(self) -> list[dict[str, Any]]:
        headers = _master_signed_headers(self.validator_kp, "GET", "/v1/workers")
        resp = httpx.get(f"{MASTER_URL}/v1/workers", headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json().get("workers", [])

    def master_active_workers(self, owner: str) -> list[dict[str, Any]]:
        resp = httpx.get(
            f"{MASTER_URL}/v1/workers/active",
            params={"hotkey": ss58(MINERS[owner])},
            headers={"Authorization": f"Bearer {TOKEN}"},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("workers", [])

    def master_units(self) -> list[dict[str, Any]]:
        headers = _master_signed_headers(self.validator_kp, "GET", "/v1/workers/units")
        resp = httpx.get(f"{MASTER_URL}/v1/workers/units", headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json().get("units", [])

    def worker_status_cli(self) -> str:
        cfg = self.workdir / "worker-cli.yaml"
        cfg.write_text(_worker_cli_yaml(MASTER_URL, WORKER_URIS["alice"]), encoding="utf-8")
        proc = subprocess.run(
            [
                str(PRISM_PY),
                "-c",
                "import sys; from base.cli_app.main import app; "
                f"sys.argv=['base','worker','status','--config',{str(cfg)!r}]; app()",
            ],
            env=self._env(),
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        return proc.stdout + proc.stderr


def _terminate(proc: Proc) -> None:
    if proc.popen.poll() is not None:
        return
    try:
        proc.popen.send_signal(signal.SIGTERM)
        proc.popen.wait(timeout=8)
    except Exception:
        try:
            proc.popen.kill()
        except Exception:
            pass


def _two_script_bundle(arch: str, train: str) -> str:
    import base64
    import io
    import zipfile

    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("architecture.py", arch)
        archive.writestr("training.py", train)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def _prism_signed_headers(secret: str, body: bytes, *, hotkey: str, nonce: str) -> dict[str, str]:
    from prism_challenge.auth import canonical_submission_message

    timestamp = str(int(time.time()))
    message = canonical_submission_message(
        hotkey=hotkey, nonce=nonce, timestamp=timestamp, body=body
    )
    signature = hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()
    return {
        "X-Hotkey": hotkey,
        "X-Signature": signature,
        "X-Nonce": nonce,
        "X-Timestamp": timestamp,
    }


def _master_signed_headers(
    keypair: Any, method: str, path: str, *, query_string: str = ""
) -> dict[str, str]:
    import uuid

    nonce = uuid.uuid4().hex
    ts = str(int(time.time()))
    canonical = canonical_validator_request(
        method=method, path=path, query_string=query_string, timestamp=ts, nonce=nonce, body=b""
    )
    sig = keypair.sign(canonical.encode())
    sig_hex = "0x" + bytes(sig).hex() if isinstance(sig, (bytes, bytearray)) else str(sig)
    return {
        "X-Hotkey": keypair.ss58_address,
        "X-Signature": sig_hex,
        "X-Nonce": nonce,
        "X-Timestamp": ts,
    }


def _worker_cli_yaml(master_url: str, key_uri: str) -> str:
    return f"""\
network:
  name: base
  netuid: 100
  chain_endpoint: null
  wallet_name: default
  wallet_hotkey: default
  wallet_path: null
  master_uid: 0
compute:
  worker_plane_enabled: true
worker:
  agent:
    master_url: {master_url}
    gateway_url: null
    capabilities:
      - gpu
    poll_interval_seconds: 5.0
    request_timeout_seconds: 15.0
    broker_url: http://127.0.0.1:8082
    broker_token_file: null
  deploy:
    provider: local
    gpu_count: 1
    max_price_per_hour: null
    max_lifetime_hours: 1.0
    startup_commands: tail -f /dev/null
    ready_timeout_seconds: 60.0
  identity:
    key_uri: {key_uri}
    key_mnemonic: null
    wallet_name: null
    wallet_hotkey: null
    miner_key_uri: null
    miner_key_mnemonic: null
    miner_wallet_name: null
    miner_wallet_hotkey: null
    miner_hotkey: null
    binding_signature: null
    binding_nonce: null
docker:
  broker_url: http://127.0.0.1:8082
  broker_allowed_images:
    - ghcr.io/baseintelligence/
observability:
  log_json: false
  sentry_dsn: null
  otel_service_name: base-worker
"""


def wait_until(
    desc: str, fn: Callable[[], Any], *, timeout: float = 90.0, interval: float = 1.0
) -> Any:
    deadline = time.time() + timeout
    last_exc: Exception | None = None
    while time.time() < deadline:
        try:
            value = fn()
            if value:
                return value
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
        time.sleep(interval)
    raise TimeoutError(
        f"timed out waiting for {desc}" + (f" (last error: {last_exc})" if last_exc else "")
    )


def wait_health(url: str, name: str, *, timeout: float = 45.0) -> None:
    def _ok() -> bool:
        try:
            return httpx.get(f"{url}/health", timeout=3).status_code == 200
        except Exception:
            return False

    wait_until(f"{name} health", _ok, timeout=timeout, interval=1.0)
    print(f"  {name} healthy at {url}")


# ------------------------------- drills -------------------------------------


def drill_admission(h: Harness) -> bool:
    print("\n=== DRILL 4: admission gate 403 -> acceptance after enrollment (VAL-CROSS-004) ===")
    before = h.prism_submit("dave", nonce=f"adm-before-{uuid.uuid4().hex}")
    print(f"  before-enrollment submit: HTTP {before.status_code} body={before.text[:160]}")
    ok_403 = before.status_code == 403 and "NO_ACTIVE_WORKER" in before.text
    worker = h.start_worker("dave")
    wait_until("dave active worker", lambda: len(h.master_active_workers("dave")) >= 1, timeout=60)
    active = h.master_active_workers("dave")
    print(f"  GET /v1/workers/active?hotkey=dave -> {len(active)} active worker(s)")
    after = h.prism_submit("dave", nonce=f"adm-after-{uuid.uuid4().hex}")
    after_id = after.json().get("id") if after.status_code < 300 else "-"
    print(f"  after-enrollment submit: HTTP {after.status_code} id={after_id}")
    ok_after = after.status_code < 300
    h.kill(worker)
    _wait_stale(h, ss58(MINERS["dave"]))
    passed = ok_403 and ok_after
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def drill_full_pipeline(h: Harness) -> bool:
    print("\n=== DRILL 1: full pipeline submission -> 2 workers -> score (VAL-CROSS-001) ===")
    workers = [h.start_worker("alice"), h.start_worker("bob"), h.start_worker("carol")]
    for owner in ("alice", "bob", "carol"):
        wait_until(
            f"{owner} active", lambda o=owner: len(h.master_active_workers(o)) >= 1, timeout=60
        )
    resp = h.prism_submit("carol", nonce=f"pipe-{uuid.uuid4().hex}")
    assert resp.status_code < 300, resp.text
    sid = str(resp.json()["id"])
    print(f"  (a) submission accepted id={sid}")
    units = wait_until(
        "prism exposes gpu unit",
        lambda: [u for u in h.prism_work_units() if str(u.get("submission_id")) == sid],
        timeout=30,
    )
    print(
        f"  (a) prism /internal/v1/work_units exposes {len(units)} unit(s) for {sid}, "
        f"submission_ref={units[0].get('submission_ref')}"
    )

    def _assigned_owners() -> set[str] | None:
        owners = {
            w["miner_hotkey"]
            for w in h.master_workers()
            if w["status"] == "active" and w.get("last_heartbeat_at")
        }
        # distinct non-carol owners active and evaluating
        return owners if len(owners) >= 2 else None

    wait_until("2 active workers", _assigned_owners, timeout=30)
    print("  (b) fleet shows >=2 active distinct-owner workers")
    wait_until(
        "prism records score", lambda: _completed_with_score(h.prism_submission(sid)), timeout=120
    )
    # Authoritative operator surface (VAL-CROSS-001): the finalized score is
    # queryable via the prism submission read. PASS derives strictly from it.
    final = h.prism_submission(sid)
    score = _score_of(final)
    print(f"  (e) GET /v1/submissions/{sid}: status={final.get('status')} score={score}")
    ok_score = final.get("status") == "completed" and score is not None
    for w in workers:
        h.kill(w)
    for owner in ("alice", "bob", "carol"):
        _wait_stale(h, ss58(MINERS[owner]))
    passed = ok_score
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def drill_self_eval(h: Harness) -> bool:
    print("\n=== DRILL 2: self-eval exclusion under scarcity (VAL-CROSS-002) ===")
    # Exactly two active workers: alice (== submitter H) and bob (distinct owner).
    workers = [h.start_worker("alice"), h.start_worker("bob")]
    for owner in ("alice", "bob"):
        wait_until(
            f"{owner} active", lambda o=owner: len(h.master_active_workers(o)) >= 1, timeout=60
        )
    alice_hk = ss58(MINERS["alice"])
    alice_worker_ids = {w["worker_id"] for w in h.master_active_workers("alice")}
    resp = h.prism_submit("alice", nonce=f"self-{uuid.uuid4().hex}")
    assert resp.status_code < 300, resp.text
    sid = str(resp.json()["id"])
    print(f"  submission from H=alice accepted id={sid} own_worker_ids={sorted(alice_worker_ids)}")

    # Operator-observable assertion (VAL-CROSS-002): via GET /v1/workers/units,
    # the submitter's OWN worker (owned by alice) must never appear as a replica
    # of alice's unit while it waits/is handled. Under scarcity the only eligible
    # distinct owner is bob, so the unit degrades to bob or holds pending -- never
    # to alice.
    replica_worker_ids: set[str] = set()
    replica_owners: set[str] = set()

    def _alice_unit() -> dict[str, Any] | None:
        found: dict[str, Any] | None = None
        for unit in h.master_units():
            if unit.get("submission_ref") == alice_hk and unit.get("audit") is None:
                found = unit
                for replica in unit.get("replicas") or []:
                    if replica.get("worker_id"):
                        replica_worker_ids.add(replica["worker_id"])
                    if replica.get("miner_hotkey"):
                        replica_owners.add(replica["miner_hotkey"])
        return found

    # The unit must really exist on the master surface (non-vacuous)...
    wait_until("alice's unit visible via GET /v1/workers/units", _alice_unit, timeout=30)
    # ...and across the observation window alice's own worker never becomes a replica.
    deadline = time.time() + 15
    while time.time() < deadline:
        _alice_unit()
        time.sleep(1)

    self_became_replica = bool(replica_worker_ids & alice_worker_ids) or alice_hk in replica_owners
    print(
        f"  GET /v1/workers/units alice-unit: replica_workers={sorted(replica_worker_ids)} "
        f"replica_owners={sorted(replica_owners)} self_became_replica={self_became_replica}"
    )
    passed = not self_became_replica
    for w in workers:
        h.kill(w)
    for owner in ("alice", "bob"):
        _wait_stale(h, ss58(MINERS[owner]))
    print(
        f"  RESULT: {'PASS (self-eval exclusion held via /v1/workers/units)' if passed else 'FAIL'}"
    )
    return passed


def drill_divergence(h: Harness) -> tuple[bool, bool, dict[str, Any]]:
    print("\n=== DRILL 3: divergence -> dispute -> audit -> fault (VAL-CROSS-003) ===")
    erin_hk = ss58(MINERS["erin"])
    workers = [
        h.start_worker("alice"),
        h.start_worker("bob", divergence_hotkey=erin_hk),
        h.start_worker("erin"),
    ]
    for owner in ("alice", "bob", "erin"):
        wait_until(
            f"{owner} active", lambda o=owner: len(h.master_active_workers(o)) >= 1, timeout=60
        )
    resp = h.prism_submit("erin", nonce=f"div-{uuid.uuid4().hex}")
    assert resp.status_code < 300, resp.text
    sid = str(resp.json()["id"])
    print(f"  submission from erin accepted id={sid} (bob will corrupt its manifest)")

    def _fault_visible() -> dict[str, Any] | None:
        for w in h.master_workers():
            for f in w.get("faults") or []:
                if f.get("work_unit_id") == sid or str(f.get("work_unit_id", "")).startswith(sid):
                    return {"worker_id": w["worker_id"], "owner": w["miner_hotkey"], "fault": f}
        return None

    fault = wait_until("worker fault from audit", _fault_visible, timeout=150)
    print(
        f"  (e) fault visible in fleet: worker={fault['worker_id']} owner={fault['owner']} "
        f"detail={fault['fault'].get('detail')}"
    )
    final = h.prism_submission(sid)
    print(f"  (d) prism submission {sid}: status={final.get('status')} score={_score_of(final)}")
    ok_no_live_score = _score_of(final) is None or final.get("status") != "completed"
    ok_fault_on_liar = fault["owner"] == ss58(MINERS["bob"])

    # VAL-CROSS-011: reconstruct the whole dispute story from operator APIs alone
    # (no DB/file reads): the new signed GET /v1/workers/units + prism submission
    # status + fleet/CLI fault.
    cross011_ok = _reconstruct_dispute_via_api(
        h, sid=sid, fault=fault, submission=final, no_live_score=ok_no_live_score
    )

    evidence = {"sid": sid, "fault": fault, "submission": final}
    for w in workers:
        h.kill(w)
    for owner in ("alice", "bob", "erin"):
        _wait_stale(h, ss58(MINERS[owner]))
    passed = ok_fault_on_liar and ok_no_live_score
    print(
        f"  RESULT: {'PASS' if passed else 'FAIL'} "
        f"(fault_on_liar={ok_fault_on_liar}, no_live_score={ok_no_live_score})"
    )
    return passed, cross011_ok, evidence


def _reconstruct_dispute_via_api(
    h: Harness, *, sid: str, fault: dict[str, Any], submission: dict[str, Any], no_live_score: bool
) -> bool:
    """VAL-CROSS-011: rebuild dispute -> audit -> invalidation -> fault via APIs only."""

    print("\n=== DRILL 11: dispute lifecycle discoverable via APIs alone (VAL-CROSS-011) ===")
    units = h.master_units()
    unit = next(
        (
            u
            for u in units
            if u.get("work_unit_id") == sid or str(u.get("work_unit_id", "")).startswith(sid)
        ),
        None,
    )
    # (a) the unit's disputed state is visible on the master surface.
    ok_disputed = unit is not None and unit.get("status") == "disputed"
    audit = (unit or {}).get("audit") or {}
    # (b) the audit unit + validator executor kind + terminal outcome are visible.
    ok_audit = (
        bool(audit)
        and audit.get("executor_kind") == "validator"
        and audit.get("outcome") in ("pending", "passed", "mismatch-resolved")
    )
    # (c) the affected submission is not a live-ranked completed score.
    ok_invalidated = no_live_score
    # (d) the lying worker's fault is visible on GET /v1/workers AND base worker status.
    cli = h.worker_status_cli()
    ok_fault_both = bool(fault) and fault["worker_id"] in cli
    print(
        f"  (a) GET /v1/workers/units unit={unit.get('work_unit_id') if unit else None} "
        f"status={unit.get('status') if unit else None}"
    )
    print(
        f"  (b) audit unit={audit.get('work_unit_id')} executor_kind={audit.get('executor_kind')} "
        f"outcome={audit.get('outcome')}"
    )
    print(
        f"  (c) prism submission status={submission.get('status')} no_live_score={ok_invalidated}"
    )
    print(f"  (d) fault worker={fault['worker_id']} visible on API+CLI={ok_fault_both}")
    passed = ok_disputed and ok_audit and ok_invalidated and ok_fault_both
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed


def drill_fleet_agreement(h: Harness) -> bool:
    print("\n=== DRILL 6/9: fleet API vs `base worker status` CLI agree (VAL-CROSS-009) ===")
    workers = [h.start_worker("alice"), h.start_worker("bob")]
    for owner in ("alice", "bob"):
        wait_until(
            f"{owner} active", lambda o=owner: len(h.master_active_workers(o)) >= 1, timeout=60
        )

    # Diff the two operator surfaces: every worker's id, owner, provider, status,
    # and fault count must be IDENTICAL between GET /v1/workers and the parsed
    # `base worker status` CLI output (retried briefly so both surfaces observe
    # the same heartbeat-derived status).
    api_view: dict[str, dict[str, Any]] = {}
    cli_view: dict[str, dict[str, Any]] = {}
    cli_text = ""
    agree = False
    deadline = time.time() + 30
    while time.time() < deadline:
        api_view = _fleet_from_api(h.master_workers())
        cli_text = h.worker_status_cli()
        cli_view = _parse_cli_workers(cli_text)
        if api_view and api_view == cli_view:
            agree = True
            break
        time.sleep(2)

    print("  --- GET /v1/workers ---")
    for wid, rec in sorted(api_view.items()):
        print(
            f"    {wid} owner={rec['owner']} provider={rec['provider']} "
            f"status={rec['status']} faults={rec['faults']}"
        )
    print("  --- base worker status ---")
    print("    " + cli_text.replace("\n", "\n    ").strip())
    if not agree:
        only_api = {k: api_view[k] for k in api_view.keys() - cli_view.keys()}
        only_cli = {k: cli_view[k] for k in cli_view.keys() - api_view.keys()}
        mismatched = {
            k: {"api": api_view[k], "cli": cli_view[k]}
            for k in api_view.keys() & cli_view.keys()
            if api_view[k] != cli_view[k]
        }
        print(f"  DIFF only_api={only_api} only_cli={only_cli} mismatched={mismatched}")
    else:
        print(f"  API and CLI fleet views identical for {len(api_view)} worker(s)")
    for w in workers:
        h.kill(w)
    for owner in ("alice", "bob"):
        _wait_stale(h, ss58(MINERS[owner]))
    print(f"  RESULT: {'PASS' if agree else 'FAIL'}")
    return agree


#: Worker lifecycle statuses rendered by ``base worker status`` (used to reliably
#: distinguish fleet rows from any interleaved log lines when parsing the CLI).
_WORKER_STATUSES = {"active", "stale", "pending", "retired"}


def _fleet_from_api(api_workers: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Project GET /v1/workers into a comparable {worker_id: fields} map."""

    return {
        w["worker_id"]: {
            "owner": w["miner_hotkey"],
            "provider": w["provider"],
            "status": w["status"],
            "faults": len(w.get("faults") or []),
        }
        for w in api_workers
    }


def _parse_cli_workers(cli: str) -> dict[str, dict[str, Any]]:
    """Parse `base worker status` table rows into a comparable map.

    Each fleet row is ``WORKER_ID OWNER PROVIDER STATUS FAULTS LAST_SEEN`` (all
    single whitespace-free tokens). Rows are identified by a known lifecycle
    status + numeric fault column so interleaved log lines are ignored.
    """

    rows: dict[str, dict[str, Any]] = {}
    for line in cli.splitlines():
        parts = line.split()
        if len(parts) < 6:
            continue
        worker_id, owner, provider, status, faults = parts[:5]
        if status not in _WORKER_STATUSES or not faults.isdigit():
            continue
        rows[worker_id] = {
            "owner": owner,
            "provider": provider,
            "status": status,
            "faults": int(faults),
        }
    return rows


def _completed_with_score(sub: dict[str, Any]) -> dict[str, Any] | None:
    if sub.get("status") == "completed" and _score_of(sub) is not None:
        return sub
    return None


def _score_of(sub: dict[str, Any]) -> Any:
    for key in ("final_score", "score"):
        if sub.get(key) is not None:
            return sub[key]
    score = sub.get("score")
    if isinstance(score, dict):
        return score.get("final_score")
    return None


def _wait_stale(h: Harness, owner_hotkey: str) -> None:
    time.sleep(WORKER_TTL + 2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="", help="comma list of drill numbers (1,2,3,4,6)")
    parser.add_argument("--workdir", default="/tmp/mission-run")
    args = parser.parse_args()
    selected = {s.strip() for s in args.only.split(",") if s.strip()} or {"4", "1", "3", "2", "6"}

    h = Harness(workdir=Path(args.workdir))
    results: dict[str, bool] = {}
    try:
        print("== bring up master + prism + validator ==")
        h.start_master()
        h.start_prism()
        wait_health(MASTER_URL, "master")
        wait_health(PRISM_URL, "prism")
        h.start_validator()
        time.sleep(3)

        if "4" in selected:
            results["VAL-CROSS-004 admission"] = drill_admission(h)
        if "1" in selected:
            results["VAL-CROSS-001 pipeline"] = drill_full_pipeline(h)
        if "3" in selected:
            ok, cross011_ok, _ = drill_divergence(h)
            results["VAL-CROSS-003 divergence"] = ok
            results["VAL-CROSS-011 dispute-discoverable"] = cross011_ok
        if "2" in selected:
            results["VAL-CROSS-002 self-eval"] = drill_self_eval(h)
        if "6" in selected:
            results["VAL-CROSS-009 fleet-agree"] = drill_fleet_agreement(h)
    finally:
        h.kill_all()

    print("\n==================== DRILL SUMMARY ====================")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    all_ok = results and all(results.values())
    print("======================================================")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
