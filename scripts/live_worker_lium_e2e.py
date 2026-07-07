#!/usr/bin/env python3
"""Live Lium worker-plane end-to-end: real pod enrolls with a LOCAL master
(VAL-CROSS-005), opt-in and billable.

Gated behind ``BASE_LIVE_PROVIDER_TESTS=1`` (real Lium money; AGENTS.md money
rules + architecture.md sec 3.1 cost guardrails). ONE batched Lium session
bounded to <= $2:

1. bring up a LOCAL mission master (worker plane ON, mock metagraph) + a tiny
   in-process fake challenge exposing exactly one gpu work unit;
2. provision a real Lium pod running the worker image via the REAL CLI
   ``base worker deploy --provider lium`` (cheapest suitable offer,
   ``termination_hours=1``);
3. reach RUNNING, install the torch-free ``bittensor_wallet`` keypair + copy the
   mission ``base`` source + the in-pod agent runner into the pod over SSH;
4. run :mod:`pod_worker_agent` INSIDE the pod, reaching the local master over a
   REVERSE SSH tunnel; the agent enrolls (miner-signed binding pre-signed on the
   host), heartbeats to ``active``, pulls the gpu unit, executes the CPU stub, and
   posts an ExecutionProof stamped with the LIUM provider + the REAL pod id;
5. verify enrollment (fleet ``GET /v1/workers``) + the forwarded ExecutionProof
   (provider names lium + the real pod id, worker signature verifies);
6. DELETE the pod, verify it is gone, assert ``GET /pods`` empty + the balance
   delta <= $2, and kill every local process by PID.

The pod is terminated in a ``finally`` on EVERY path so a failure never leaks a
billable pod. A JSON trace is written to ``$LIVE_WORKER_TRACE`` (default
``/tmp/live_worker_e2e_trace.json``). Secrets are never printed. NOT for
production.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import bittensor as bt
import httpx

from base.compute import LiumClient
from base.compute.lium import LiumError
from base.security.validator_auth import canonical_validator_request
from base.validator.agent.signing import KeypairRequestSigner
from base.worker.deploy import build_signed_binding
from base.worker.proof import (
    MANIFEST_SHA256_PAYLOAD_KEY,
    PROOF_PAYLOAD_KEY,
    execution_proof_signing_payload,
)

# Reuse the proven VAL-PROV-014 helpers (credentials, ssh target parsing, ...).
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
import live_lium_e2e as prov  # noqa: E402

REPO_ROOT = SCRIPTS_DIR.parent
BASE_SRC = REPO_ROOT / "src"
BASE_PY = REPO_ROOT / ".venv" / "bin" / "python"
MASTER_SCRIPT = SCRIPTS_DIR / "mission" / "mission_master.py"
POD_AGENT_SCRIPT = SCRIPTS_DIR / "mission" / "pod_worker_agent.py"

MASTER_PORT = 3140
FAKE_PORT = 3141
MASTER_URL = f"http://127.0.0.1:{MASTER_PORT}"
FAKE_URL = f"http://127.0.0.1:{FAKE_PORT}"
POD_MASTER_PORT = 3400  # reverse-tunnel listener port INSIDE the pod (pod-local)
TOKEN = "mission-live-worker-token"
NETUID = 100
WORKER_TTL = 120
CHALLENGE_SLUG = "prism"

OWNER_URI = "//LiumWorkerOwner"
SUBMITTER_URI = "//LiumSubmitter"
VALIDATOR_URI = "//LiumFleetValidator"
WORKER_URI = "//LiumPodWorker"

MAX_PRICE_PER_GPU_HR = 1.50
BALANCE_GUARDRAIL = 2.00
POD_RUN_SECONDS = 300
POD_POLL_BUDGET = 15 * 60
POD_POLL_INTERVAL = 20.0

# The M1 placeholder worker image (WORKER_IMAGE = ghcr.io/.../prism-evaluator) is a
# large private-namespace GHCR image; Lium executors return CREATION_FAILED trying
# to pull it (confirmed live). The pod only needs to run the CPU worker agent, so
# this live check rents a small, reliably-pullable Docker Hub base image and
# installs the worker runtime in-pod. The deploy still runs through the real
# ``base worker deploy --provider lium`` CLI (which reuses this pre-created
# template), and the enroll/execute/prove/terminate cycle is unchanged.
POD_IMAGE = "python"
POD_IMAGE_TAG = "3.12-slim"
POD_PIP_DEPS = (
    "httpx pydantic pydantic-settings sqlalchemy PyYAML fastapi typer "
    "docker bittensor-wallet"
)


def ss58(uri: str) -> str:
    return str(bt.Keypair.create_from_uri(uri).ss58_address)


class RetryLiumClient(LiumClient):
    """LiumClient that retries the throttled Lium edge (HTTP 429) with backoff.

    Lium's API edge enforces a strict short-window request cap; a burst of calls
    (offer list + template + rent + status polls) can transiently return 429. Only
    429 is retried -- every other status (including a WAF 403) surfaces unchanged.
    """

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        last: Exception | None = None
        for attempt in range(5):
            try:
                return await super()._request(method, path, **kwargs)
            except LiumError as exc:
                if getattr(exc, "status_code", None) != 429:
                    raise
                last = exc
                await asyncio.sleep(6.0 * (attempt + 1))
        raise last if last else LiumError(f"Lium {method} {path} ret/exhausted")


# --------------------------------------------------------------------------- #
# In-process fake challenge (the master's bridged work source + result sink).   #
# --------------------------------------------------------------------------- #
@dataclass
class ChallengeState:
    unit: dict[str, Any] | None = None
    results: list[dict[str, Any]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def arm(self, submission_id: str, submission_ref: str) -> None:
        with self.lock:
            self.unit = {
                "submission_id": submission_id,
                "submission_ref": submission_ref,
            }

    def disarm(self) -> None:
        with self.lock:
            self.unit = None

    def work_units(self) -> list[dict[str, Any]]:
        with self.lock:
            return [dict(self.unit)] if self.unit else []

    def record_result(self, body: dict[str, Any]) -> None:
        with self.lock:
            self.results.append(body)

    def captured(self) -> list[dict[str, Any]]:
        with self.lock:
            return list(self.results)


_STATE = ChallengeState()


class _ChallengeHandler(BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:  # silence stdlib access log
        return

    def _json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib API
        if self.path == "/health":
            self._json(200, {"status": "ok"})
            return
        if self.path.rstrip("/") == "/internal/v1/work_units":
            self._json(200, {"work_units": _STATE.work_units()})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - stdlib API
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode() or "{}")
        except ValueError:
            body = {"_raw": raw.decode(errors="replace")}
        path = self.path.rstrip("/")
        if path == "/internal/v1/work_units/result":
            _STATE.record_result(body)
            self._json(200, {"status": "accepted"})
            return
        if path == "/internal/v1/work_units/fold":
            self._json(200, {"status": "folded"})
            return
        self._json(404, {"error": "not found"})


def _start_fake_challenge() -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", FAKE_PORT), _ChallengeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


# --------------------------------------------------------------------------- #
# Local master process.                                                         #
# --------------------------------------------------------------------------- #
def _master_config(workdir: Path) -> dict[str, Any]:
    entries = [
        {
            "hotkey": ss58(OWNER_URI),
            "uid": 0,
            "validator_permit": False,
            "stake": 1000.0,
        },
        {
            "hotkey": ss58(SUBMITTER_URI),
            "uid": 1,
            "validator_permit": False,
            "stake": 1000.0,
        },
        {
            "hotkey": ss58(VALIDATOR_URI),
            "uid": 99,
            "validator_permit": True,
            "stake": 5000.0,
        },
    ]
    return {
        "port": MASTER_PORT,
        "host": "127.0.0.1",
        "db_url": f"sqlite+aiosqlite:///{workdir / 'master.sqlite3'}",
        "netuid": NETUID,
        "metagraph": entries,
        "prism": {
            "slug": CHALLENGE_SLUG,
            "internal_base_url": FAKE_URL,
            "token": TOKEN,
        },
        "orchestration_interval_seconds": 1.0,
        "worker_heartbeat_ttl_seconds": WORKER_TTL,
        "health_interval_seconds": 2.0,
        "replication_factor": 1,
    }


def _spawn_master(workdir: Path) -> subprocess.Popen:
    cfg_path = workdir / "master.json"
    cfg_path.write_text(json.dumps(_master_config(workdir), indent=2), encoding="utf-8")
    log = (workdir / "master.log").open("w", encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(BASE_SRC)
    return subprocess.Popen(
        [str(BASE_PY), str(MASTER_SCRIPT), str(cfg_path)],
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(REPO_ROOT),
    )


def _wait_health(url: str, name: str, *, timeout: float = 45.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(f"{url}/health", timeout=3).status_code == 200:
                print(f"  {name} healthy at {url}")
                return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1.0)
    raise TimeoutError(f"{name} did not become healthy at {url}")


# --------------------------------------------------------------------------- #
# Fleet reads (signed as the permitted mock validator).                          #
# --------------------------------------------------------------------------- #
def _validator_headers(method: str, path: str) -> dict[str, str]:
    keypair = bt.Keypair.create_from_uri(VALIDATOR_URI)
    nonce = uuid.uuid4().hex
    ts = str(int(time.time()))
    canonical = canonical_validator_request(
        method=method, path=path, query_string="", timestamp=ts, nonce=nonce, body=b""
    )
    sig = keypair.sign(canonical.encode())
    sig_hex = (
        "0x" + bytes(sig).hex() if isinstance(sig, (bytes, bytearray)) else str(sig)
    )
    return {
        "X-Hotkey": ss58(VALIDATOR_URI),
        "X-Signature": sig_hex,
        "X-Nonce": nonce,
        "X-Timestamp": ts,
    }


def _fleet_workers() -> list[dict[str, Any]]:
    resp = httpx.get(
        f"{MASTER_URL}/v1/workers",
        headers=_validator_headers("GET", "/v1/workers"),
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json().get("workers", [])


# --------------------------------------------------------------------------- #
# SSH helpers.                                                                   #
# --------------------------------------------------------------------------- #
def _ssh_base(target: dict[str, Any], private_key: str) -> list[str]:
    argv = [
        "-i",
        private_key,
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "ConnectTimeout=20",
        "-o",
        "BatchMode=yes",
        "-o",
        "ServerAliveInterval=15",
    ]
    if target.get("port"):
        argv += ["-p", str(target["port"])]
    return argv


def _ssh_run(
    target: dict[str, Any], private_key: str, command: str, *, timeout: int = 240
) -> subprocess.CompletedProcess:
    argv = [
        "ssh",
        *_ssh_base(target, private_key),
        f"{target['user']}@{target['host']}",
        command,
    ]
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


def _scp(
    target: dict[str, Any],
    private_key: str,
    local: str,
    remote: str,
    *,
    timeout: int = 240,
) -> subprocess.CompletedProcess:
    argv = ["scp", *_ssh_base(target, private_key)]
    # scp uses -P for port
    if "-p" in argv:
        idx = argv.index("-p")
        argv[idx] = "-P"
    argv += [local, f"{target['user']}@{target['host']}:{remote}"]
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


def _detect_pod_python(target: dict[str, Any], private_key: str) -> str:
    probe = _ssh_run(
        target,
        private_key,
        "test -x /opt/venv/bin/python && echo /opt/venv/bin/python "
        "|| command -v python3",
    )
    out = probe.stdout.strip()
    py = out.splitlines()[-1].strip() if out else ""
    if not py:
        raise SystemExit(f"could not locate python in pod: {probe.stderr[:300]}")
    return py


# --------------------------------------------------------------------------- #
# Deploy via the real CLI.                                                       #
# --------------------------------------------------------------------------- #
def _write_deploy_config(workdir: Path, template_name: str, ssh_pub_file: str) -> Path:
    cfg = f"""\
network:
  name: base
  netuid: {NETUID}
  chain_endpoint: null
  wallet_name: default
  wallet_hotkey: default
  wallet_path: null
  master_uid: 0
compute:
  worker_plane_enabled: true
worker:
  agent:
    master_url: {MASTER_URL}
    gateway_url: null
    capabilities:
      - gpu
    poll_interval_seconds: 2.0
    request_timeout_seconds: 15.0
    broker_url: http://127.0.0.1:8082
    broker_token_file: null
  deploy:
    provider: lium
    gpu_count: 1
    max_price_per_hour: {MAX_PRICE_PER_GPU_HR}
    max_lifetime_hours: 1.0
    startup_commands: tail -f /dev/null
    ready_timeout_seconds: 60.0
    ssh_public_key_file: {ssh_pub_file}
    ssh_key_name: {prov.SSH_KEY_NAME}
    template_name: {template_name}
  identity:
    key_uri: {WORKER_URI}
    key_mnemonic: null
    wallet_name: null
    wallet_hotkey: null
    miner_key_uri: {OWNER_URI}
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
    path = workdir / "deploy.yaml"
    path.write_text(cfg, encoding="utf-8")
    return path


async def _deploy_pod(
    lium: LiumClient,
    workdir: Path,
    api_key: str,
    ssh_pub_file: str,
    trace: dict[str, Any],
) -> str:
    template_name = f"prism-worker-live-{int(time.time())}"
    cfg_path = _write_deploy_config(workdir, template_name, ssh_pub_file)

    # Pre-create the worker template with an EMPTY environment. The real CLI would
    # otherwise bake the pod env (which carries a loopback master_url) into the
    # template body, and the Lium API edge WAF blocks any request body containing
    # an ``http://127.0.0.1`` URL (403 "Request blocked"). Pre-creating by name
    # means the CLI's idempotent ensure_template finds it and skips that blocked
    # POST; the pod runs ``tail -f /dev/null`` so the baked env is unused anyway.
    template_id = await lium.ensure_template(
        name=template_name,
        docker_image=POD_IMAGE,
        docker_image_tag=POD_IMAGE_TAG,
        internal_ports=(22,),
        startup_commands="tail -f /dev/null",
    )
    trace["template"] = {"name": template_name, "id": template_id}
    print(f"[deploy] pre-created template {template_name} id={template_id}")
    await asyncio.sleep(6)  # let the Lium edge rate-limit window cool down

    env = dict(os.environ)
    env["PYTHONPATH"] = str(BASE_SRC)
    env["LIUM_API_KEY"] = api_key
    argv = [
        str(REPO_ROOT / ".venv" / "bin" / "base"),
        "worker",
        "deploy",
        "--provider",
        "lium",
        "--max-price",
        str(MAX_PRICE_PER_GPU_HR),
        "--config",
        str(cfg_path),
    ]
    print(
        "[deploy] running real CLI: base worker deploy --provider lium "
        f"--max-price {MAX_PRICE_PER_GPU_HR}"
    )
    out = ""
    pod_id = None
    for attempt in range(1, 3):
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            env=env,
            cwd=str(REPO_ROOT),
            timeout=240,
        )
        out = proc.stdout + "\n" + proc.stderr
        trace.setdefault("deploy_cli_attempts", []).append(
            {
                "attempt": attempt,
                "returncode": proc.returncode,
                "stdout": proc.stdout.strip()[-1500:],
                "stderr": proc.stderr.strip()[-1000:],
            }
        )
        print(out.strip())
        match = re.search(r"Provisioned\s+lium\s+instance\s+(\S+)", out)
        if proc.returncode == 0 and match:
            pod_id = match.group(1)
            break
        # a failed attempt cleans up its own partially-rented pod (LiumClient
        # try/finally); guard anyway by terminating any leaked pod before retry.
        leaked = await lium.list_pods()
        for pod in leaked:
            pid = str(pod.get("id"))
            print(f"[deploy] terminating leaked pod {pid} before retry")
            await lium.terminate(pid)
        if attempt < 2:
            print("[deploy] deploy attempt failed; retrying in 12s")
            await asyncio.sleep(12)
    if pod_id is None:
        raise SystemExit("base worker deploy did not provision a pod")
    trace["deploy_cli"] = {"template_name": template_name}
    offer = re.search(r"Selected\s+lium\s+offer\s+(\S+)\s+\(([^)]*)\)", out)
    if offer:
        trace["deploy_cli"]["selected_offer"] = offer.group(1)
        trace["deploy_cli"]["selected_offer_desc"] = offer.group(2)
    print(f"[deploy] provisioned pod_id={pod_id}")
    return pod_id


# --------------------------------------------------------------------------- #
# Proof verification.                                                            #
# --------------------------------------------------------------------------- #
def _verify_proof(
    proof: dict[str, Any], pod_id: str, candidate_unit_ids: list[str]
) -> dict[str, Any]:
    import hashlib

    checks: dict[str, Any] = {}
    checks["version_is_1"] = proof.get("version") == 1
    checks["tier_ge_0"] = int(proof.get("tier", -1)) >= 0
    provider = proof.get("provider") or {}
    checks["provider_name_lium"] = provider.get("name") == "lium"
    checks["provider_pod_id_matches"] = str(provider.get("pod_id")) == str(pod_id)
    sig = proof.get("worker_signature") or {}
    worker_pubkey = sig.get("worker_pubkey")
    checks["worker_pubkey_is_pod_worker"] = worker_pubkey == ss58(WORKER_URI)
    manifest = proof.get(MANIFEST_SHA256_PAYLOAD_KEY)
    checks["manifest_present"] = bool(manifest)

    matched_unit = None
    sig_ok = False
    for uid in candidate_unit_ids:
        if not uid:
            continue
        if manifest and hashlib.sha256(uid.encode()).hexdigest() == manifest:
            matched_unit = uid
        try:
            payload = execution_proof_signing_payload(
                manifest_sha256=str(manifest), unit_id=uid
            )
            kp = bt.Keypair(ss58_address=str(worker_pubkey))
            if kp.verify(payload, sig.get("sig")):
                sig_ok = True
                matched_unit = uid
                break
        except Exception:  # noqa: BLE001
            continue
    checks["worker_signature_verifies"] = sig_ok
    checks["manifest_binds_unit"] = matched_unit is not None
    checks["matched_unit_id"] = matched_unit
    checks["all_passed"] = all(
        v for k, v in checks.items() if k not in ("matched_unit_id",)
    )
    return checks


# --------------------------------------------------------------------------- #
# Main sequence.                                                                 #
# --------------------------------------------------------------------------- #
async def _amain() -> int:
    if os.environ.get("BASE_LIVE_PROVIDER_TESTS") != "1":
        print("BASE_LIVE_PROVIDER_TESTS != 1; skipping live worker E2E")
        return 0

    creds = prov._load_credentials()
    api_key = creds["LIUM_API_KEY"]
    prov._load_ssh_public_key()  # validate the mission SSH public key exists
    private_key = os.environ.get("LIUM_SSH_PRIVATE_KEY", prov.DEFAULT_SSH_PRIVATE_KEY)
    ssh_pub_file = private_key + ".pub"

    workdir = Path(os.environ.get("LIVE_WORKER_WORKDIR", "/tmp/live-worker-run"))
    workdir.mkdir(parents=True, exist_ok=True)

    trace: dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pod_id": None,
    }

    lium = RetryLiumClient(api_key)
    balance_before = await lium.balance()
    trace["balance_before"] = balance_before
    print(f"[budget] Lium balance before: ${balance_before:.4f}")
    if balance_before < BALANCE_GUARDRAIL:
        raise SystemExit(
            f"balance ${balance_before:.4f} < ${BALANCE_GUARDRAIL} minimum; aborting"
        )

    master_proc: subprocess.Popen | None = None
    fake_server: ThreadingHTTPServer | None = None
    agent_proc: subprocess.Popen | None = None
    pod_id: str | None = None
    already_deleted = False
    exit_code = 0

    try:
        # 1. local control plane
        print("== bring up local master + fake challenge ==")
        fake_server = _start_fake_challenge()
        _wait_health(FAKE_URL, "fake-challenge")
        master_proc = _spawn_master(workdir)
        _wait_health(MASTER_URL, "master")

        # 2. arm exactly one gpu unit (submitted by a DISTINCT owner from the worker)
        submission_id = uuid.uuid4().hex
        _STATE.arm(submission_id, ss58(SUBMITTER_URI))
        print(
            f"[unit] armed gpu unit submission_id={submission_id} "
            f"owner={ss58(SUBMITTER_URI)}"
        )

        # 3. provision the pod via the real CLI
        pod_id = await _deploy_pod(lium, workdir, api_key, ssh_pub_file, trace)
        trace["pod_id"] = pod_id

        # 4. poll to RUNNING
        print("[pod] polling to RUNNING")
        deadline = time.monotonic() + POD_POLL_BUDGET
        instance = None
        last_status = ""
        while True:
            current = await lium.status(pod_id)
            status = (current.status or "").upper()
            if status != last_status:
                print(f"[pod] status={status}")
                trace.setdefault("status_transitions", []).append(status)
                last_status = status
            if status in prov.RUNNING_STATUSES:
                instance = current
                break
            if status in prov.TERMINAL_FAIL_STATUSES:
                raise SystemExit(f"pod reached terminal status {status}")
            if time.monotonic() >= deadline:
                raise SystemExit(f"pod did not reach RUNNING (last={status})")
            await asyncio.sleep(POD_POLL_INTERVAL)

        raw = dict(instance.raw)
        target = prov._parse_ssh_target(str(raw.get("ssh_connect_cmd", "")), raw)
        if not target.get("host") or not target.get("user"):
            raise SystemExit(
                f"could not parse ssh target: {raw.get('ssh_connect_cmd')!r}"
            )
        trace["ssh_target"] = {k: v for k, v in target.items()}
        print(f"[ssh] target {target['user']}@{target['host']}:{target.get('port')}")

        # 5. in-pod setup: python, keypair lib, base source, agent runner + config
        # SSH may take a moment after RUNNING; retry the first probe.
        pod_py = ""
        for attempt in range(1, 9):
            try:
                pod_py = _detect_pod_python(target, private_key)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"[ssh] probe attempt {attempt} failed: {exc}")
                time.sleep(10)
        if not pod_py:
            raise SystemExit("pod SSH never became reachable")
        print(f"[pod] python={pod_py}")

        print("[pod] installing worker runtime deps")
        pip = _ssh_run(
            target,
            private_key,
            f"{pod_py} -m pip install --no-cache-dir {POD_PIP_DEPS}",
            timeout=400,
        )
        trace["pip_install"] = {
            "rc": pip.returncode,
            "tail": (pip.stdout + pip.stderr).strip()[-400:],
        }
        if pip.returncode != 0:
            raise SystemExit(
                f"pod dependency install failed: {(pip.stdout + pip.stderr)[-500:]}"
            )

        print("[pod] copying base source + agent runner")
        tgz = workdir / "base-src.tgz"
        subprocess.run(
            ["tar", "-C", str(BASE_SRC), "-czf", str(tgz), "base"], check=True
        )
        _ssh_run(target, private_key, "mkdir -p /tmp/base-src")
        r1 = _scp(target, private_key, str(tgz), "/tmp/base-src.tgz")
        if r1.returncode != 0:
            raise SystemExit(f"scp base source failed: {r1.stderr[:300]}")
        rx = _ssh_run(
            target,
            private_key,
            "tar -C /tmp/base-src -xzf /tmp/base-src.tgz && echo OK",
        )
        if "OK" not in rx.stdout:
            raise SystemExit(f"extract base source failed: {rx.stderr[:300]}")
        r2 = _scp(
            target, private_key, str(POD_AGENT_SCRIPT), "/tmp/pod_worker_agent.py"
        )
        if r2.returncode != 0:
            raise SystemExit(f"scp agent runner failed: {r2.stderr[:300]}")

        # 6. pre-sign the miner binding on the host (pod never holds the miner key)
        worker_pubkey = ss58(WORKER_URI)
        miner_signer = KeypairRequestSigner(bt.Keypair.create_from_uri(OWNER_URI))
        binding = build_signed_binding(
            worker_pubkey=worker_pubkey, miner_signer=miner_signer
        )
        pod_cfg = {
            "master_url": f"http://127.0.0.1:{POD_MASTER_PORT}",
            "worker_uri": WORKER_URI,
            "miner_hotkey": binding.miner_hotkey,
            "binding_signature": binding.signature,
            "binding_nonce": binding.nonce,
            "provider": "lium",
            "pod_id": pod_id,
            "executor_id": pod_id,
            "capabilities": ["gpu"],
            "heartbeat_interval_seconds": 5,
            "poll_interval_seconds": 2.0,
            "run_seconds": POD_RUN_SECONDS,
        }
        cfg_local = workdir / "pod_cfg.json"
        cfg_local.write_text(json.dumps(pod_cfg, indent=2), encoding="utf-8")
        r3 = _scp(target, private_key, str(cfg_local), "/tmp/pod_cfg.json")
        if r3.returncode != 0:
            raise SystemExit(f"scp pod config failed: {r3.stderr[:300]}")

        # 7. run the in-pod agent over a reverse SSH tunnel to the local master
        agent_log = (workdir / "pod-agent.log").open("w", encoding="utf-8")
        run_cmd = (
            f"PYTHONPATH=/tmp/base-src {pod_py} -u "
            "/tmp/pod_worker_agent.py /tmp/pod_cfg.json"
        )
        agent_argv = [
            "ssh",
            *_ssh_base(target, private_key),
            "-o",
            "ExitOnForwardFailure=yes",
            "-R",
            f"127.0.0.1:{POD_MASTER_PORT}:127.0.0.1:{MASTER_PORT}",
            f"{target['user']}@{target['host']}",
            run_cmd,
        ]
        print("[pod] launching in-pod worker agent over reverse tunnel")
        agent_proc = subprocess.Popen(
            agent_argv, stdout=agent_log, stderr=subprocess.STDOUT
        )

        # 8. observe enrollment (fleet active)
        print("[observe] waiting for the pod worker to reach active in the fleet")
        active_worker = None
        deadline = time.time() + 120
        while time.time() < deadline:
            try:
                for w in _fleet_workers():
                    if (
                        w.get("provider_instance_ref") == pod_id
                        and w.get("status") == "active"
                    ):
                        active_worker = w
                        break
            except Exception:  # noqa: BLE001
                pass
            if active_worker:
                break
            if agent_proc.poll() is not None:
                raise SystemExit(
                    "in-pod agent exited before enrolling; see pod-agent.log"
                )
            time.sleep(3)
        if not active_worker:
            raise SystemExit("pod worker did not reach active within timeout")
        trace["active_worker"] = {
            "worker_id": active_worker.get("worker_id"),
            "worker_pubkey": active_worker.get("worker_pubkey"),
            "miner_hotkey": active_worker.get("miner_hotkey"),
            "provider": active_worker.get("provider"),
            "provider_instance_ref": active_worker.get("provider_instance_ref"),
            "status": active_worker.get("status"),
        }
        print(f"[observe] worker ACTIVE: {trace['active_worker']}")

        # 9. observe the forwarded ExecutionProof
        print(
            "[observe] waiting for the executed unit's ExecutionProof to be forwarded"
        )
        forwarded = None
        deadline = time.time() + 120
        while time.time() < deadline:
            for body in _STATE.captured():
                result = body.get("result") or {}
                if PROOF_PAYLOAD_KEY in result:
                    forwarded = body
                    break
            if forwarded:
                break
            if agent_proc.poll() is not None:
                # agent finished its run window; give a final read
                for body in _STATE.captured():
                    result = body.get("result") or {}
                    if PROOF_PAYLOAD_KEY in result:
                        forwarded = body
                        break
                if forwarded:
                    break
            time.sleep(3)
        if not forwarded:
            raise SystemExit("no ExecutionProof was forwarded for the executed unit")
        _STATE.disarm()
        proof = forwarded["result"][PROOF_PAYLOAD_KEY]
        candidate_units = [str(forwarded.get("work_unit_id") or ""), submission_id]
        checks = _verify_proof(proof, pod_id, candidate_units)
        trace["forwarded_result"] = {
            "work_unit_id": forwarded.get("work_unit_id"),
            "submission_ref": forwarded.get("submission_ref"),
            "proof": proof,
            "checks": checks,
        }
        print(f"[proof] checks: {json.dumps(checks)}")
        if not checks["all_passed"]:
            raise SystemExit(f"ExecutionProof verification failed: {checks}")
        print(
            "[proof] ExecutionProof verified: lium provider + real pod id + "
            "valid worker signature"
        )

    except SystemExit as exc:
        trace["result"] = "failure"
        trace["error"] = str(exc)
        print(f"FAILURE: {exc}", file=sys.stderr)
        exit_code = 1
    except Exception as exc:  # noqa: BLE001
        trace["result"] = "failure"
        trace["error"] = f"{type(exc).__name__}: {exc}"
        print(f"FAILURE: {type(exc).__name__}: {exc}", file=sys.stderr)
        exit_code = 1
    finally:
        # stop the in-pod agent + reverse tunnel
        if agent_proc is not None and agent_proc.poll() is None:
            agent_proc.terminate()
            try:
                agent_proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                agent_proc.kill()
        # DELETE the pod on EVERY path
        if pod_id is not None:
            print(f"[cleanup] DELETE pod {pod_id}")
            try:
                await lium.terminate(pod_id)
                already_deleted = await lium.verify_terminated(pod_id)
                if not already_deleted:
                    await asyncio.sleep(5)
                    already_deleted = await lium.verify_terminated(pod_id)
                trace["verified_terminated"] = already_deleted
                print(f"[cleanup] verified_terminated={already_deleted}")
            except Exception as exc:  # noqa: BLE001
                trace["cleanup_error"] = str(exc)
                print(f"[cleanup] terminate error: {exc}", file=sys.stderr)
        # tear down local processes by PID
        if master_proc is not None and master_proc.poll() is None:
            master_proc.terminate()
            try:
                master_proc.wait(timeout=8)
            except Exception:  # noqa: BLE001
                master_proc.kill()
        if fake_server is not None:
            fake_server.shutdown()

    # post-run accounting (best-effort; pod already deleted)
    try:
        balance_after = await lium.balance()
        pods_left = await lium.list_pods()
    except Exception as exc:  # noqa: BLE001
        balance_after = None
        pods_left = []
        trace["accounting_error"] = str(exc)
    if balance_after is not None:
        delta = balance_before - balance_after
        trace["balance_after"] = balance_after
        trace["balance_delta"] = delta
        trace["pods_remaining"] = len(pods_left)
        print(
            f"[done] balance_before=${balance_before:.4f} after=${balance_after:.4f} "
            f"delta=${delta:.4f} pods_remaining={len(pods_left)}"
        )
        if len(pods_left) != 0:
            print(
                f"WARNING: {len(pods_left)} pod(s) still listed after cleanup",
                file=sys.stderr,
            )
            exit_code = 1
        if delta > BALANCE_GUARDRAIL:
            print(
                f"WARNING: balance delta ${delta:.4f} exceeds "
                f"${BALANCE_GUARDRAIL} guardrail",
                file=sys.stderr,
            )
            exit_code = 1

    if exit_code == 0 and trace.get("result") != "failure":
        trace["result"] = "success"
    trace["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out = Path(os.environ.get("LIVE_WORKER_TRACE", "/tmp/live_worker_e2e_trace.json"))
    out.write_text(json.dumps(trace, indent=2, default=str))
    print(f"trace written to {out}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_amain()))
