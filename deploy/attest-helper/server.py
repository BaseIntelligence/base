#!/usr/bin/env python3
"""base-attest-helper — GET /v1/quote for miner CVM certify path."""
from __future__ import annotations

import binascii
import hashlib
import json
import os
import socket
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

SOCK = os.environ.get("BASE_DSTACK_SOCK", "/var/run/dstack.sock")
PORT = int(os.environ.get("BASE_ATTEST_PORT", "8081"))
DOMAIN = b"base-attest-v1"


def scale_bytes(b: bytes) -> bytes:
    """parity-scale-codec Encode for Vec/&[u8]: Compact(len) ‖ bytes (single-byte compact for len<64)."""
    n = len(b)
    if n < 64:
        return bytes([n << 2]) + b
    if n < (1 << 14):
        v = (n << 2) | 0x01
        return bytes([v & 0xFF, (v >> 8) & 0xFF]) + b
    if n < (1 << 30):
        v = (n << 2) | 0x02
        return bytes([(v >> i) & 0xFF for i in (0, 8, 16, 24)]) + b
    raise ValueError("bytes too long for compact")


def scale_u16(v: int) -> bytes:
    return int(v).to_bytes(2, "little")


def scale_u64(v: int) -> bytes:
    return int(v).to_bytes(8, "little")


def compute_report_data(
    netuid: int,
    epoch: int,
    miner_pk: bytes,
    nonce: bytes,
    validator_hk: bytes,
) -> bytes:
    if not (len(miner_pk) == 32 and len(nonce) == 32 and len(validator_hk) == 32):
        raise ValueError("miner/nonce/validator must be 32 bytes")
    pre = b"".join(
        [
            scale_bytes(DOMAIN),
            scale_u16(netuid),
            scale_u64(epoch),
            miner_pk,  # fixed [u8;32] raw
            nonce,
            validator_hk,
        ]
    )
    return hashlib.sha512(pre).digest()  # 64 bytes


def dstack_post(path: str, body: dict) -> dict:
    payload = json.dumps(body).encode()
    req = (
        f"POST {path} HTTP/1.1\r\n"
        f"Host: dstack\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(payload)}\r\n"
        f"Connection: close\r\n\r\n"
    ).encode() + payload
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.settimeout(60)
        s.connect(SOCK)
        s.sendall(req)
        data = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            data += chunk
    finally:
        s.close()
    if b"\r\n\r\n" not in data:
        raise RuntimeError(f"bad dstack response: {data[:200]!r}")
    header, raw = data.split(b"\r\n\r\n", 1)
    status = header.split(b"\r\n", 1)[0].decode(errors="replace")
    hl = header.lower()
    if b"transfer-encoding: chunked" in hl:
        out = b""
        i = 0
        while i < len(raw):
            nl = raw.find(b"\r\n", i)
            if nl < 0:
                break
            size = int(raw[i:nl], 16)
            i = nl + 2
            if size == 0:
                break
            out += raw[i : i + size]
            i += size + 2
        raw = out
    if not status.startswith("HTTP/1.1 200") and not status.startswith("HTTP/1.0 200"):
        raise RuntimeError(f"dstack {path} failed: {status} body={raw[:300]!r}")
    return json.loads(raw.decode())


def hex32(s: str, name: str) -> bytes:
    t = (s or "").strip().lower().removeprefix("0x")
    b = bytes.fromhex(t)
    if len(b) != 32:
        raise ValueError(f"{name} must be 32 bytes hex, got {len(b)}")
    return b


class H(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code: int, obj: dict):
        body = (json.dumps(obj) + "\n").encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/healthz", "/readyz", "/"):
            self._send(200, {"ok": True, "service": "base-attest-helper"})
            return
        if u.path != "/v1/quote":
            self._send(404, {"error": "not found"})
            return
        qs = parse_qs(u.query)
        try:
            netuid = int((qs.get("netuid") or ["0"])[0])
            epoch = int((qs.get("epoch") or ["0"])[0])
            nonce = hex32((qs.get("nonce_hex") or [""])[0], "nonce_hex")
            vhk = hex32((qs.get("validator_hotkey_hex") or [""])[0], "validator_hotkey_hex")
            mhk = hex32((qs.get("miner_hotkey_hex") or [""])[0], "miner_hotkey_hex")
            rd = compute_report_data(netuid, epoch, mhk, nonce, vhk)
            hex_rd = binascii.hexlify(rd).decode()
            result = dstack_post("/GetQuote", {"report_data": hex_rd})
            # Normalize to certify.rs QuoteJson shape
            quote = result.get("quote") or result.get("quote_hex") or result.get("quoteHex")
            if isinstance(quote, str):
                qhex = quote.removeprefix("0x")
            elif isinstance(quote, (bytes, bytearray)):
                qhex = binascii.hexlify(quote).decode()
            else:
                # some APIs nest
                qhex = binascii.hexlify(bytes(quote)).decode() if quote is not None else ""
                if not qhex and "quote" in result:
                    raise RuntimeError(f"unrecognized quote field: {type(result.get('quote'))}")
            el = result.get("event_log") or result.get("event_log_json") or result.get("eventLog")
            if isinstance(el, (dict, list)):
                el_json = json.dumps(el, separators=(",", ":"))
            elif isinstance(el, str):
                # keep as-is if already JSON text
                el_json = el
            else:
                el_json = "[]"
            self._send(
                200,
                {
                    "quote_hex": qhex,
                    "event_log_json": el_json,
                    "report_data_hex": hex_rd,
                },
            )
        except Exception as e:
            self._send(500, {"error": str(e)})


def main() -> None:
    print(f"base-attest-helper listening on 0.0.0.0:{PORT} sock={SOCK}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()


if __name__ == "__main__":
    main()
