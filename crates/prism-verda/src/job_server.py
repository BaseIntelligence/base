"""Operator-owned Verda batch entry (not miner-supplied). Health + one job then exit."""
import json
import os
import subprocess
import sys
import tarfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("PRISM_VERDA_PORT", "8000"))
WORKDIR = "/tmp/prism_eval"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("%s\n" % (fmt % args))

    def do_GET(self):
        if self.path.split("?", 1)[0] in ("/health", "/"):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")
            return
        self.send_error(404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n)
        os.makedirs(WORKDIR, exist_ok=True)
        tar = os.path.join(WORKDIR, "upload.tar")
        with open(tar, "wb") as f:
            f.write(body)
        try:
            with tarfile.open(tar) as tf:
                tf.extractall(WORKDIR)
        except (tarfile.TarError, OSError) as e:
            self._json(500, {"ok": False, "error": "extract: %s" % e})
            os._exit(1)
        timeout = int(os.environ.get("PRISM_VERDA_TIMEOUT_SECS", "18000"))
        env = os.environ.copy()
        try:
            subprocess.check_call(
                [
                    sys.executable,
                    "-c",
                    "import transformers",
                ],
                cwd=WORKDIR,
                env=env,
            )
        except subprocess.CalledProcessError:
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--break-system-packages",
                    "--root-user-action=ignore",
                    "transformers==4.44.2",
                    "datasets==3.0.2",
                    "pyarrow==17.0.0",
                ],
                cwd=WORKDIR,
                env=env,
            )
        try:
            proc = subprocess.run(
                [sys.executable, "main.py"],
                cwd=WORKDIR,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            code = proc.returncode
            captured = (proc.stdout or "") + "\n" + (proc.stderr or "")
        except subprocess.TimeoutExpired as e:
            code = 124
            captured = "%s\n" % e
        log = ""
        try:
            with open(os.path.join(WORKDIR, "harness.log"), "r", errors="replace") as f:
                log = f.read()
        except OSError:
            log = captured
        metrics = ""
        mp = os.path.join(WORKDIR, "metrics.json")
        if os.path.isfile(mp):
            with open(mp, "r", errors="replace") as f:
                metrics = f.read()
        payload = {
            "ok": code == 0,
            "code": code,
            "log": log[-65536:],
            "metrics": metrics,
        }
        self._json(200 if code == 0 else 500, payload)
        os._exit(0 if code == 0 else 1)

    def _json(self, code, obj):
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
