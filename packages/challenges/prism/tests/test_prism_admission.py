"""Admission rule + flag-OFF regression tests (VAL-PRISM-014/015/020/021).

A tiny threaded HTTP stub stands in for the base master's ``GET /v1/workers/active?hotkey=`` so the
tests observe exactly what prism queries and control the answer (zero/one worker, 5xx, slow, or a
closed port). Everything binds to loopback ports in the mission range 3100-3199.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import anyio
from conftest import VALID_CODE, signed_headers
from fastapi.testclient import TestClient

from prism_challenge.admission import NO_ACTIVE_WORKER_CODE
from prism_challenge.app import create_app
from prism_challenge.config import PrismSettings, WorkerPlaneConfig


def _reserve_port() -> int:
    """Return a free loopback port in the mission range 3100-3199."""
    for port in range(3100, 3200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("no free port available in 3100-3199")


class _StubMaster:
    """Threaded stub of the master's ``GET /v1/workers/active`` recording every request."""

    def __init__(self, *, active_count: int = 0, status_code: int = 200, delay: float = 0.0):
        self.active_count = active_count
        self.status_code = status_code
        self.delay = delay
        self.requests: list[dict[str, Any]] = []
        self.port = _reserve_port()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> _StubMaster:
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: Any) -> None:  # silence stub access logs
                pass

            def do_GET(self) -> None:  # noqa: N802 (http.server API)
                parsed = urlparse(self.path)
                query = parse_qs(parsed.query)
                stub.requests.append(
                    {"path": parsed.path, "hotkey": query.get("hotkey", [None])[0]}
                )
                if stub.delay:
                    time.sleep(stub.delay)
                hotkey = query.get("hotkey", [""])[0]
                body = json.dumps(
                    {
                        "workers": [
                            {
                                "worker_id": f"w{i}",
                                "worker_pubkey": f"pk{i}",
                                "miner_hotkey": hotkey,
                                "provider": "local",
                                "status": "active",
                                "created_at": "2026-01-01T00:00:00Z",
                            }
                            for i in range(stub.active_count)
                        ]
                    }
                ).encode()
                try:
                    self.send_response(stub.status_code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass  # client (prism) already timed out and closed the connection

        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()


def _settings(
    tmp_path: Path,
    *,
    admission_requires_worker: bool = False,
    master_base_url: str | None = None,
    admission_timeout: float = 5.0,
    enabled: bool = False,
    validator_hotkeys: tuple[str, ...] = (),
    max_code_bytes: int = 7_500_000,
) -> PrismSettings:
    return PrismSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'prism.sqlite3'}",
        shared_token="secret",
        allow_insecure_signatures=True,
        fineweb_sample_count=4,
        distributed_contract_policy="off",
        max_code_bytes=max_code_bytes,
        validator_hotkeys=validator_hotkeys,
        worker_plane=WorkerPlaneConfig(
            enabled=enabled,
            admission_requires_worker=admission_requires_worker,
            master_base_url=master_base_url,
            admission_timeout_seconds=admission_timeout,
        ),
    )


def _json_body(code: str = VALID_CODE) -> bytes:
    return json.dumps({"code": code, "filename": "model.py"}, separators=(",", ":")).encode()


def _submission_count(client: TestClient) -> int:
    repo = client.app.state.repository

    async def _count() -> int:
        async with repo.database.connect() as conn:
            rows = await conn.execute_fetchall("SELECT COUNT(*) AS c FROM submissions")
        return int(dict(rows[0])["c"])

    return anyio.run(_count)


def _post_direct(client: TestClient, *, nonce: str, body: bytes | None = None) -> Any:
    payload = body if body is not None else _json_body()
    return client.post(
        "/v1/submissions",
        content=payload,
        headers={
            **signed_headers("secret", payload, nonce=nonce),
            "Content-Type": "application/json",
        },
    )


def _post_bridge(
    client: TestClient, *, hotkey: str = "hk-bridge", body: bytes | None = None
) -> Any:
    payload = body if body is not None else _json_body()
    return client.post(
        "/internal/v1/bridge/submissions",
        content=payload,
        headers={
            "Authorization": "Bearer secret",
            "X-Base-Verified-Hotkey": hotkey,
            "Content-Type": "application/json",
        },
    )


# --- VAL-PRISM-014: admission ON, direct route -----------------------------------------------


def test_admission_on_rejects_without_active_worker(tmp_path):
    with _StubMaster(active_count=0) as stub:
        settings = _settings(
            tmp_path, admission_requires_worker=True, master_base_url=stub.base_url
        )
        with TestClient(create_app(settings)) as client:
            response = _post_direct(client, nonce="n1")
            assert response.status_code == 403, response.text
            assert response.json()["detail"]["code"] == NO_ACTIVE_WORKER_CODE
            assert _submission_count(client) == 0
        # prism queried the master admission surface for the submitting hotkey
        assert any(
            req["path"] == "/v1/workers/active" and req["hotkey"] == "hk" for req in stub.requests
        )


def test_admission_on_accepts_after_worker_active(tmp_path):
    with _StubMaster(active_count=0) as stub:
        settings = _settings(
            tmp_path, admission_requires_worker=True, master_base_url=stub.base_url
        )
        with TestClient(create_app(settings)) as client:
            rejected = _post_direct(client, nonce="n1")
            assert rejected.status_code == 403, rejected.text
            assert _submission_count(client) == 0

            # a worker becomes active; an identically-constructed re-signed submission is accepted
            stub.active_count = 1
            accepted = _post_direct(client, nonce="n2")
            assert accepted.status_code == 200, accepted.text
            assert accepted.json()["hotkey"] == "hk"
            assert _submission_count(client) == 1
            submission_id = accepted.json()["id"]
            assert client.get(f"/v1/submissions/{submission_id}").status_code == 200


# --- VAL-PRISM-015: admission flag OFF is byte-identical to legacy ----------------------------


def test_admission_off_accepts_without_master_call(tmp_path):
    with _StubMaster(active_count=0) as stub:
        settings = _settings(
            tmp_path, admission_requires_worker=False, master_base_url=stub.base_url
        )
        with TestClient(create_app(settings)) as client:
            response = _post_direct(client, nonce="n1")
            assert response.status_code == 200, response.text
            assert response.json()["hotkey"] == "hk"
            assert _submission_count(client) == 1
        assert stub.requests == []  # flag OFF => zero master calls


def test_admission_off_legacy_error_paths_untouched(tmp_path):
    settings = _settings(
        tmp_path,
        admission_requires_worker=False,
        validator_hotkeys=("val-hk",),
        max_code_bytes=2_000,
    )
    with TestClient(create_app(settings)) as client:
        # invalid signature => 401
        body = _json_body()
        bad_sig = client.post(
            "/v1/submissions",
            content=body,
            headers={
                "X-Hotkey": "hk",
                "X-Signature": "deadbeef",
                "X-Nonce": "n-bad",
                "X-Timestamp": signed_headers("secret", body)["X-Timestamp"],
                "Content-Type": "application/json",
            },
        )
        assert bad_sig.status_code == 401, bad_sig.text

        # validator hotkey => 403 with the LEGACY message (not NO_ACTIVE_WORKER)
        val_body = _json_body()
        val = client.post(
            "/v1/submissions",
            content=val_body,
            headers={
                **signed_headers("secret", val_body, hotkey="val-hk", nonce="n-val"),
                "Content-Type": "application/json",
            },
        )
        assert val.status_code == 403, val.text
        assert val.json()["detail"] == "validator hotkey is not allowed to submit"

        # oversized code => 413
        big = _json_body("A" * 4_000)
        oversized = _post_direct(client, nonce="n-big", body=big)
        assert oversized.status_code == 413, oversized.text
        assert oversized.json()["detail"] == "submission too large"


# --- VAL-PRISM-020: admission is fail-closed and bounded when the master is unreachable -------


def test_admission_fail_closed_connection_refused(tmp_path):
    closed_port = _reserve_port()  # nothing is listening here
    settings = _settings(
        tmp_path,
        admission_requires_worker=True,
        master_base_url=f"http://127.0.0.1:{closed_port}",
        admission_timeout=2.0,
    )
    with TestClient(create_app(settings)) as client:
        start = time.monotonic()
        response = _post_direct(client, nonce="n1")
        elapsed = time.monotonic() - start
        assert response.status_code == 403, response.text
        assert response.json()["detail"]["code"] == NO_ACTIVE_WORKER_CODE
        assert _submission_count(client) == 0
        assert elapsed < 3.0  # connection refused is immediate, well under the bound


def test_admission_fail_closed_timeout_is_bounded(tmp_path):
    with _StubMaster(active_count=1, delay=5.0) as stub:  # accepts but never answers in time
        settings = _settings(
            tmp_path,
            admission_requires_worker=True,
            master_base_url=stub.base_url,
            admission_timeout=0.5,
        )
        with TestClient(create_app(settings)) as client:
            start = time.monotonic()
            response = _post_direct(client, nonce="n1")
            elapsed = time.monotonic() - start
            assert response.status_code == 403, response.text
            assert response.json()["detail"]["code"] == NO_ACTIVE_WORKER_CODE
            assert _submission_count(client) == 0
            assert elapsed < 3.0  # bounded by admission_timeout + margin, not the 5s stub delay


def test_admission_fail_closed_master_5xx(tmp_path):
    with _StubMaster(active_count=1, status_code=503) as stub:
        settings = _settings(
            tmp_path, admission_requires_worker=True, master_base_url=stub.base_url
        )
        with TestClient(create_app(settings)) as client:
            response = _post_direct(client, nonce="n1")
            assert response.status_code == 403, response.text
            assert response.json()["detail"]["code"] == NO_ACTIVE_WORKER_CODE
            assert _submission_count(client) == 0


def test_admission_fail_closed_shape_is_consistent(tmp_path):
    """Every fail-closed mode uses the SAME 403 NO_ACTIVE_WORKER shape (VAL-PRISM-020)."""
    outcomes: list[tuple[int, str]] = []

    # zero workers
    with _StubMaster(active_count=0) as stub:
        settings = _settings(
            tmp_path / "zero", admission_requires_worker=True, master_base_url=stub.base_url
        )
        with TestClient(create_app(settings)) as client:
            r = _post_direct(client, nonce="n1")
            outcomes.append((r.status_code, r.json()["detail"]["code"]))

    # 5xx
    with _StubMaster(active_count=1, status_code=500) as stub:
        settings = _settings(
            tmp_path / "err", admission_requires_worker=True, master_base_url=stub.base_url
        )
        with TestClient(create_app(settings)) as client:
            r = _post_direct(client, nonce="n1")
            outcomes.append((r.status_code, r.json()["detail"]["code"]))

    # connection refused
    closed_port = _reserve_port()
    settings = _settings(
        tmp_path / "refused",
        admission_requires_worker=True,
        master_base_url=f"http://127.0.0.1:{closed_port}",
        admission_timeout=1.0,
    )
    with TestClient(create_app(settings)) as client:
        r = _post_direct(client, nonce="n1")
        outcomes.append((r.status_code, r.json()["detail"]["code"]))

    assert outcomes == [(403, NO_ACTIVE_WORKER_CODE)] * 3


def test_admission_off_inert_when_master_unreachable(tmp_path):
    """Flag OFF => the three unreachable master states cause zero difference and zero calls."""
    closed_port = _reserve_port()
    settings = _settings(
        tmp_path,
        admission_requires_worker=False,
        master_base_url=f"http://127.0.0.1:{closed_port}",
    )
    with TestClient(create_app(settings)) as client:
        response = _post_direct(client, nonce="n1")
        assert response.status_code == 200, response.text
        assert _submission_count(client) == 1


# --- VAL-PRISM-021: the BASE bridge path enforces admission identically -----------------------


def test_bridge_admission_on_rejects_without_active_worker(tmp_path):
    with _StubMaster(active_count=0) as stub:
        settings = _settings(
            tmp_path, admission_requires_worker=True, master_base_url=stub.base_url
        )
        with TestClient(create_app(settings)) as client:
            response = _post_bridge(client, hotkey="H")
            assert response.status_code == 403, response.text
            assert response.json()["detail"]["code"] == NO_ACTIVE_WORKER_CODE
            assert _submission_count(client) == 0
        assert any(
            req["path"] == "/v1/workers/active" and req["hotkey"] == "H" for req in stub.requests
        )


def test_bridge_admission_on_accepts_with_active_worker(tmp_path):
    with _StubMaster(active_count=1) as stub:
        settings = _settings(
            tmp_path, admission_requires_worker=True, master_base_url=stub.base_url
        )
        with TestClient(create_app(settings)) as client:
            response = _post_bridge(client, hotkey="H")
            assert response.status_code == 200, response.text
            assert response.json()["hotkey"] == "H"
            assert _submission_count(client) == 1


def test_bridge_admission_off_is_legacy(tmp_path):
    with _StubMaster(active_count=0) as stub:
        settings = _settings(
            tmp_path, admission_requires_worker=False, master_base_url=stub.base_url
        )
        with TestClient(create_app(settings)) as client:
            response = _post_bridge(client, hotkey="H")
            assert response.status_code == 200, response.text
            assert _submission_count(client) == 1
        assert stub.requests == []  # flag OFF => zero master calls on the bridge path
