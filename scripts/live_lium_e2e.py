#!/usr/bin/env python3
"""Live Lium/Targon validation driven through the M1 provider clients (opt-in).

Gated behind ``BASE_LIVE_PROVIDER_TESTS=1`` because it makes real, billable calls
to production Lium (architecture.md sec 3.1 cost guardrails + mission AGENTS.md
money rules). It exercises the SAME ``LiumClient`` / ``TargonClient`` code the
worker plane uses, never raw curl.

Two phases:

* Read-only reachability (no mutating call): Lium ``users/me`` balance,
  ``executors`` offers, ``watchtower/digest``; Targon ``inventory`` offers and an
  authenticated ``apps`` listing (VAL-PROV-012 / VAL-PROV-013).
* ONE batched Lium rental cycle (VAL-PROV-014): ensure the SSH key + template,
  rent the cheapest suitable executor (< $1.50/GPU/hr, ``termination_hours=1``,
  pod name ``mission-<ts>``), poll to RUNNING, SSH ``nvidia-smi -L``, pull logs,
  DELETE, verify the pod is gone, and record the balance delta (must be <= $1).

The pod is deleted in a ``finally`` block on EVERY path (including exceptions and
a partially provisioned pod) so a failure never leaks a billable pod. A JSON trace
of the whole sequence is written to ``$LIUM_E2E_TRACE`` (default
``/tmp/lium_e2e_trace.json``). Credentials/keys are read from the environment or
the mission credentials file; secret values are never printed.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from base.compute import InstanceSpec, LiumClient, Offer, TargonClient

MAX_PRICE_PER_GPU_HR = 1.50
TERMINATION_HOURS = 1
POLL_BUDGET_SECONDS = 15 * 60
POLL_INTERVAL_SECONDS = 25.0
SSH_ATTEMPTS = 8
SSH_RETRY_SECONDS = 15.0
LOG_LINE_CAP = 40
LOG_STREAM_TIMEOUT_SECONDS = 20.0

RUNNING_STATUSES = {"RUNNING"}
TERMINAL_FAIL_STATUSES = {"FAILED", "CREATION_FAILED", "BROKEN", "STOPPED"}

DEFAULT_CREDENTIALS_FILE = "/root/.config/prism-mission/credentials.env"
DEFAULT_SSH_PRIVATE_KEY = "/root/.config/prism-mission/lium_ssh_ed25519"
SSH_KEY_NAME = "prism-mission-readiness"

E2E_TEMPLATE_NAME = "prism-mission-e2e"
E2E_IMAGE = "nvidia/cuda"
E2E_IMAGE_TAG = "12.4.1-base-ubuntu22.04"
# Single, metachar-free command: Lium rejects startup commands containing shell
# metacharacters. It keeps the container alive; Lium's pod agent serves SSH.
E2E_STARTUP_COMMAND = "tail -f /dev/null"


class _TracingLiumClient(LiumClient):
    """LiumClient that records the raw ``POST .../rent`` response shape.

    The rent response body is what :meth:`LiumClient._extract_pod_id` parses (with
    a ``GET /pods`` lookup by ``pod_name`` fallback). Capturing it lets the live
    run confirm the real shape without changing client behavior.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.rent_capture: dict[str, Any] | None = None

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> Any:
        response = await super()._request(
            method, path, json_body=json_body, params=params
        )
        if method == "POST" and path.endswith("/rent"):
            body: Any
            try:
                body = response.json() if response.content else None
            except ValueError:
                body = response.text[:1000]
            self.rent_capture = {
                "status_code": response.status_code,
                "body_type": type(body).__name__,
                "body_keys": (sorted(body.keys()) if isinstance(body, dict) else None),
                "body_preview": _preview(body),
            }
        return response


def _preview(body: Any) -> Any:
    if isinstance(body, dict):
        return {k: _preview(v) for k, v in list(body.items())[:20]}
    if isinstance(body, list):
        return [_preview(v) for v in body[:3]]
    if isinstance(body, str):
        return body[:200]
    return body


def _load_credentials() -> dict[str, str]:
    creds: dict[str, str] = {}
    for name in ("LIUM_API_KEY", "TARGON_API_KEY"):
        value = os.environ.get(name)
        if value:
            creds[name] = value
    creds_file = Path(os.environ.get("PRISM_MISSION_CREDS", DEFAULT_CREDENTIALS_FILE))
    if creds_file.is_file():
        for raw in creds_file.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :]
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            creds.setdefault(key, val.strip().strip('"').strip("'"))
    missing = [n for n in ("LIUM_API_KEY", "TARGON_API_KEY") if not creds.get(n)]
    if missing:
        raise SystemExit(f"missing credentials: {', '.join(missing)}")
    return creds


def _load_ssh_public_key() -> str:
    pub = Path(os.environ.get("LIUM_SSH_PRIVATE_KEY", DEFAULT_SSH_PRIVATE_KEY) + ".pub")
    if not pub.is_file():
        raise SystemExit(f"missing SSH public key: {pub}")
    return pub.read_text().strip()


def _parse_ssh_target(ssh_connect_cmd: str, raw: Mapping[str, Any]) -> dict[str, Any]:
    user_host = re.search(r"([\w.-]+)@([\w.\-]+)", ssh_connect_cmd or "")
    port_match = re.search(r"-p\s+(\d+)", ssh_connect_cmd or "")
    target: dict[str, Any] = {}
    if user_host:
        target["user"] = user_host.group(1)
        target["host"] = user_host.group(2)
    if port_match:
        target["port"] = int(port_match.group(1))
    if "port" not in target:
        mapping = raw.get("ports_mapping")
        if isinstance(mapping, Mapping):
            external = mapping.get("22") or mapping.get(22)
            if external is not None:
                try:
                    target["port"] = int(external)
                except (TypeError, ValueError):
                    pass
    return target


def _run_ssh_nvidia_smi(target: Mapping[str, Any], private_key: str) -> dict[str, Any]:
    argv = [
        "ssh",
        "-i",
        private_key,
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "BatchMode=yes",
    ]
    if target.get("port"):
        argv += ["-p", str(target["port"])]
    argv.append(f"{target['user']}@{target['host']}")
    argv.append("nvidia-smi -L && echo SMOKE_OK")
    last: dict[str, Any] = {}
    for attempt in range(1, SSH_ATTEMPTS + 1):
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            last = {"attempt": attempt, "returncode": None, "error": "ssh timed out"}
            time.sleep(SSH_RETRY_SECONDS)
            continue
        last = {
            "attempt": attempt,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip()[:500],
        }
        if proc.returncode == 0 and "SMOKE_OK" in proc.stdout:
            return last
        time.sleep(SSH_RETRY_SECONDS)
    return last


async def _collect_logs(client: LiumClient, pod_id: str) -> list[str]:
    lines: list[str] = []

    async def _drain() -> None:
        async for line in client.stream_logs(pod_id):
            lines.append(line)
            if len(lines) >= LOG_LINE_CAP:
                break

    try:
        await asyncio.wait_for(_drain(), timeout=LOG_STREAM_TIMEOUT_SECONDS)
    except Exception as exc:  # noqa: BLE001 - logs are best-effort
        lines.append(f"<log stream ended: {type(exc).__name__}>")
    return lines


def _suitable_offers(offers: list[Offer]) -> list[Offer]:
    suitable = [
        o
        for o in offers
        if 0 < o.price_per_hour <= MAX_PRICE_PER_GPU_HR and o.gpu_count >= 1
    ]
    suitable.sort(key=lambda o: (o.price_per_hour, o.gpu_count))
    return suitable


async def _ensure_e2e_template(client: LiumClient) -> dict[str, Any]:
    """Ensure a dedicated E2E template with a rent-safe startup command exists.

    Lium rejects a rent whose template startup command contains shell
    metacharacters ("Malicious startup command detected"), so the many public
    templates that chain ``service ssh start && tail -f /dev/null`` cannot be
    rented. This template keeps the container alive with a single metachar-free
    command; Lium's pod agent provides SSH. Idempotent: reused if it already
    exists (the client's ``ensure_template`` then finds it by name at rent time).
    """
    response = await client._request("GET", "/templates")
    data = response.json()
    templates = data.get("templates", []) if isinstance(data, dict) else data
    for tmpl in templates:
        if isinstance(tmpl, dict) and tmpl.get("name") == E2E_TEMPLATE_NAME:
            return {
                "name": E2E_TEMPLATE_NAME,
                "id": str(tmpl.get("id")),
                "image": tmpl.get("docker_image"),
                "created": False,
            }
    body = {
        "name": E2E_TEMPLATE_NAME,
        "docker_image": E2E_IMAGE,
        "docker_image_tag": E2E_IMAGE_TAG,
        "startup_commands": E2E_STARTUP_COMMAND,
        "internal_ports": [22],
        "is_private": True,
        "container_start_immediately": True,
    }
    created = await client._request("POST", "/templates", json_body=body)
    payload = created.json()
    return {
        "name": E2E_TEMPLATE_NAME,
        "id": str(payload.get("id")) if isinstance(payload, dict) else None,
        "image": f"{E2E_IMAGE}:{E2E_IMAGE_TAG}",
        "created": True,
    }


async def _read_only_phase(
    lium: LiumClient, targon: TargonClient, trace: dict[str, Any]
) -> float:
    balance = await lium.balance()
    offers = await lium.list_offers()
    digest = await lium.watchtower_digest()
    targon_offers = await targon.list_offers()
    targon_apps = await targon.list_apps()

    if balance < 2:
        raise SystemExit(f"Lium balance ${balance:.4f} < $2 minimum; aborting")
    if not offers:
        raise SystemExit("Lium GET /executors returned no offers")
    if not re.match(r"^sha256:[0-9a-f]{64}$", digest):
        raise SystemExit(f"watchtower digest not sha256-shaped: {digest[:16]}...")
    if not targon_offers or any(o.price_per_hour <= 0 for o in targon_offers):
        raise SystemExit("Targon inventory parse produced no positive-priced offers")

    trace["read_only"] = {
        "lium_balance": balance,
        "lium_offers_count": len(offers),
        "lium_suitable_count": len(_suitable_offers(offers)),
        "watchtower_digest": digest,
        "targon_offers": [
            {
                "shape": o.id,
                "gpu": o.gpu_type,
                "count": o.gpu_count,
                "usd_hr": o.price_per_hour,
            }
            for o in targon_offers
        ],
        "targon_apps_count": len(targon_apps),
    }
    print(
        f"[read-only] lium_balance=${balance:.4f} offers={len(offers)} "
        f"digest_ok targon_offers={len(targon_offers)} targon_apps={len(targon_apps)}"
    )
    return balance


async def _rental_phase(
    lium: _TracingLiumClient,
    ssh_public_key: str,
    private_key: str,
    balance_before: float,
    trace: dict[str, Any],
) -> None:
    template = await _ensure_e2e_template(lium)
    offers = _suitable_offers(await lium.list_offers())
    if not offers:
        raise SystemExit("no suitable Lium offer <= $1.50/GPU/hr")

    pod_name = f"mission-{int(time.time())}"
    spec = InstanceSpec(
        name=pod_name,
        template_ref=template["name"],
        ssh_public_keys=(ssh_public_key,),
        ssh_key_name=SSH_KEY_NAME,
        max_lifetime_hours=TERMINATION_HOURS,
        max_price_per_hour=MAX_PRICE_PER_GPU_HR,
        gpu_count=1,
    )
    rental: dict[str, Any] = {
        "pod_name": pod_name,
        "template": template,
        "attempts": [],
        "status_transitions": [],
    }
    trace["rental"] = rental

    pod_id: str | None = None
    instance = None
    for offer in offers[:5]:
        attempt = {
            "executor_id": offer.id,
            "gpu": offer.gpu_type,
            "gpu_count": offer.gpu_count,
            "price_per_gpu_hr": offer.price_per_hour,
        }
        try:
            print(
                f"[rent] trying {offer.gpu_type} ({offer.id}) "
                f"@ ${offer.price_per_hour}/gpu/hr"
            )
            instance = await lium.provision(spec, offer=offer)
            pod_id = instance.id
            attempt["result"] = "rented"
            attempt["pod_id"] = pod_id
            rental["attempts"].append(attempt)
            rental["selected"] = attempt
            break
        except Exception as exc:  # noqa: BLE001 - fallback to next-cheapest offer
            attempt["result"] = f"failed: {type(exc).__name__}: {exc}"
            rental["attempts"].append(attempt)
            print(f"[rent] failed: {type(exc).__name__}: {exc}")

    rental["rent_response_shape"] = lium.rent_capture

    if pod_id is None or instance is None:
        raise SystemExit("could not rent any suitable executor")

    already_deleted = False
    try:
        print(f"[rent] pod_id={pod_id} rented; polling to RUNNING")
        deadline = time.monotonic() + POLL_BUDGET_SECONDS
        last_status = ""
        final_status = ""
        while True:
            current = await lium.status(pod_id)
            status = (current.status or "").upper()
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            if status != last_status:
                rental["status_transitions"].append({"at": now, "status": status})
                print(f"[poll] {now} status={status}")
                last_status = status
            if status in RUNNING_STATUSES:
                final_status = status
                instance = current
                break
            if status in TERMINAL_FAIL_STATUSES:
                raise SystemExit(f"pod reached terminal failure status {status}")
            if time.monotonic() >= deadline:
                raise SystemExit(
                    f"pod did not reach RUNNING within {POLL_BUDGET_SECONDS}s "
                    f"(last status {status})"
                )
            await asyncio.sleep(POLL_INTERVAL_SECONDS)

        rental["running"] = True
        raw = dict(instance.raw)
        ssh_connect_cmd = str(raw.get("ssh_connect_cmd", ""))
        target = _parse_ssh_target(ssh_connect_cmd, raw)
        rental["ssh_target"] = {k: v for k, v in target.items() if k != "user" or v}
        if not target.get("host") or not target.get("user"):
            raise SystemExit(f"could not parse ssh target from: {ssh_connect_cmd!r}")

        target_desc = f"{target['user']}@{target['host']}:{target.get('port')}"
        print(f"[ssh] connecting to {target_desc}")
        ssh_result = _run_ssh_nvidia_smi(target, private_key)
        rental["nvidia_smi"] = ssh_result
        print(f"[ssh] returncode={ssh_result.get('returncode')}")
        print(ssh_result.get("stdout", ""))
        if ssh_result.get("returncode") != 0 or "SMOKE_OK" not in ssh_result.get(
            "stdout", ""
        ):
            raise SystemExit(f"nvidia-smi over SSH failed: {ssh_result}")

        print("[logs] pulling pod logs")
        logs = await _collect_logs(lium, pod_id)
        rental["logs_excerpt"] = logs[:LOG_LINE_CAP]
        rental["logs_line_count"] = len(logs)
        print(f"[logs] collected {len(logs)} line(s)")
        rental["final_status"] = final_status
    finally:
        if pod_id is not None:
            print(f"[cleanup] DELETE pod {pod_id}")
            await lium.terminate(pod_id)
            gone = await lium.verify_terminated(pod_id)
            rental["deleted"] = True
            rental["verified_terminated"] = gone
            already_deleted = gone
            print(f"[cleanup] verify_terminated={gone}")
            if not gone:
                # Second confirmation attempt: a billing leak is critical.
                await asyncio.sleep(5)
                gone = await lium.verify_terminated(pod_id)
                rental["verified_terminated"] = gone
                print(f"[cleanup] second verify_terminated={gone}")

    balance_after = await lium.balance()
    delta = balance_before - balance_after
    pods_left = await lium.list_pods()
    rental["balance_before"] = balance_before
    rental["balance_after"] = balance_after
    rental["balance_delta"] = delta
    rental["pods_remaining"] = len(pods_left)
    print(
        f"[done] balance_before=${balance_before:.4f} after=${balance_after:.4f} "
        f"delta=${delta:.4f} pods_remaining={len(pods_left)}"
    )
    if not already_deleted:
        raise SystemExit("pod not verified terminated; potential billing leak")
    if len(pods_left) != 0:
        raise SystemExit(f"{len(pods_left)} pod(s) still listed after cleanup")
    if delta > 1.0:
        raise SystemExit(f"balance delta ${delta:.4f} exceeds $1.00 guardrail")


async def _amain() -> int:
    if os.environ.get("BASE_LIVE_PROVIDER_TESTS") != "1":
        print("BASE_LIVE_PROVIDER_TESTS != 1; skipping live provider E2E")
        return 0

    creds = _load_credentials()
    ssh_public_key = _load_ssh_public_key()
    private_key = os.environ.get("LIUM_SSH_PRIVATE_KEY", DEFAULT_SSH_PRIVATE_KEY)

    trace: dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    lium = _TracingLiumClient(creds["LIUM_API_KEY"])
    targon = TargonClient(creds["TARGON_API_KEY"])

    exit_code = 0
    try:
        balance_before = await _read_only_phase(lium, targon, trace)
        await _rental_phase(lium, ssh_public_key, private_key, balance_before, trace)
        trace["result"] = "success"
    except SystemExit as exc:
        trace["result"] = "failure"
        trace["error"] = str(exc)
        print(f"FAILURE: {exc}", file=sys.stderr)
        exit_code = 1
    finally:
        trace["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        out = Path(os.environ.get("LIUM_E2E_TRACE", "/tmp/lium_e2e_trace.json"))
        out.write_text(json.dumps(trace, indent=2, default=str))
        print(f"trace written to {out}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_amain()))
