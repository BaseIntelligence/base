"""Parent-side miner subprocess runner (recipe 1.3.0).

Spawns `python3 -m prismlib.miner_entry <ctx.json>` — wrapped in
`unshare --net --` when available so the miner subprocess is born inside an
empty network namespace (intra-container isolation, no gVisor). The
`unshare` probe runs once and is cached; `PRISM_DISABLE_NETNS=1` forces the
isolation off; when unavailable the harness falls back to a plain
subprocess with a loud log warning + manifest record (Sim/CI keep working).

Result channel: the child writes exactly one JSON line to fd
`PRISM_RESULT_FD` (default 3, passed via `pass_fds`). stdout/stderr are
streamed into the harness log; `PRISM_PHASE=<build|train|score>` marker
lines drive per-phase timeout budgets — a phase that exceeds its budget is
killed.
"""

import collections
import json
import os
import shutil
import subprocess
import sys
import threading
import time

from .envutil import log

PHASE_MARK = "PRISM_PHASE="
_UNSHARE_CACHE = None


def probe_unshare():
    """Probe once whether `unshare --net` works in this container (cached)."""
    global _UNSHARE_CACHE
    if _UNSHARE_CACHE is not None:
        return dict(_UNSHARE_CACHE)
    if os.environ.get("PRISM_DISABLE_NETNS") == "1":
        _UNSHARE_CACHE = {"available": False, "detail": "disabled via PRISM_DISABLE_NETNS=1"}
        return dict(_UNSHARE_CACHE)
    exe = shutil.which("unshare")
    if exe is None:
        _UNSHARE_CACHE = {"available": False, "detail": "unshare not in PATH"}
        return dict(_UNSHARE_CACHE)
    try:
        r = subprocess.run([exe, "--net", "--", "true"], capture_output=True, text=True, timeout=30)
        if r.returncode == 0:
            _UNSHARE_CACHE = {"available": True, "detail": "probe ok"}
        else:
            _UNSHARE_CACHE = {
                "available": False,
                "detail": f"probe rc={r.returncode}: {(r.stderr or '')[:200]}",
            }
    except Exception as exc:  # noqa: BLE001
        _UNSHARE_CACHE = {"available": False, "detail": f"probe error: {exc}"}
    return dict(_UNSHARE_CACHE)


def _open_result_channel(env):
    """Create the result pipe; returns (read_fd, pass_fd).

    Arranges the child-visible write end as fd 3 when numbering allows
    (the literal contract); otherwise the actual fd is communicated via
    `PRISM_RESULT_FD`, which the child honors first.
    """
    r_fd, w_fd = os.pipe()
    pass_fd = w_fd
    alias_fd = None
    if w_fd != 3 and r_fd != 3:
        try:
            os.dup2(w_fd, 3)
            pass_fd = 3
            alias_fd = 3
        except OSError:
            pass_fd = w_fd
    env["PRISM_RESULT_FD"] = str(pass_fd)
    return r_fd, w_fd, pass_fd, alias_fd


def run_miner_subprocess(
    workdir,
    ctx_json_path,
    *,
    train_cap_s,
    build_timeout_s,
    score_timeout_s,
):
    """Spawn + supervise the miner subprocess.

    Returns `{"payload": dict|None, "netns": bool, "unshare": {...},
    "rc": int|None, "killed_phase": str|None, "tail": [str]}`.
    """
    unshare = probe_unshare()
    netns = bool(unshare["available"])
    if not netns:
        log(
            "WARNING: network isolation OFF ("
            + unshare["detail"]
            + "); miner subprocess shares the pod network (fallback)"
        )

    cmd = []
    if netns:
        cmd.extend(["unshare", "--net", "--"])
    cmd.extend([sys.executable, "-m", "prismlib.miner_entry", ctx_json_path])

    env = dict(os.environ)
    # The tokenizer cache is warmed by the parent (which has network); the
    # child must resolve it offline — mandatory when netns is on.
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    # Resolve prismlib from the harness root (this file's grandparent), not
    # from the child cwd — robust against workdir layout changes.
    harness_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prev_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = harness_root + (os.pathsep + prev_pp if prev_pp else "")
    r_fd, w_fd, pass_fd, alias_fd = _open_result_channel(env)

    proc = subprocess.Popen(
        cmd,
        cwd=workdir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        pass_fds=(pass_fd,),
    )
    os.close(w_fd)
    if alias_fd is not None:
        os.close(alias_fd)

    state = {"phase": "build", "phase_start": time.monotonic()}
    tail = collections.deque(maxlen=40)

    def _reader():
        try:
            for line in proc.stdout:
                line = line.rstrip("\n")
                if line.startswith(PHASE_MARK):
                    ph = line[len(PHASE_MARK) :].strip()
                    if ph:
                        state["phase"] = ph
                        state["phase_start"] = time.monotonic()
                    log(f"miner phase -> {state['phase']}")
                else:
                    tail.append(line)
                    log(f"miner| {line[:500]}")
        except ValueError:
            pass

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()

    # Drain the result channel concurrently: the one-line JSON payload can
    # exceed the 64 KiB pipe buffer (bounded telemetry series), so reading
    # only after child exit would deadlock — the child would block on
    # write(2) while the parent blocks on wait(2). Draining until EOF also
    # covers payloads larger than we retain for parsing.
    result_buf = bytearray()

    def _result_reader():
        try:
            while True:
                chunk = os.read(r_fd, 1 << 16)
                if not chunk:
                    break
                if len(result_buf) < 8 * 1024 * 1024:
                    result_buf.extend(chunk)
        except OSError:
            pass

    result_reader = threading.Thread(target=_result_reader, daemon=True)
    result_reader.start()

    budgets = {
        "build": float(build_timeout_s),
        "train": float(train_cap_s) + 120.0,
        "score": float(score_timeout_s),
    }
    killed_phase = None
    while True:
        if proc.poll() is not None:
            break
        phase = state["phase"]
        budget = budgets.get(phase, max(budgets.values()))
        if time.monotonic() - state["phase_start"] > budget:
            killed_phase = phase
            log(f"miner phase '{phase}' exceeded budget {budget:.0f}s — killing subprocess")
            proc.kill()
            break
        time.sleep(0.5)

    try:
        rc = proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        rc = proc.wait(timeout=30)
    reader.join(timeout=30)
    # EOF arrives when every write end closes (child exit). If a stray
    # grandchild keeps the fd open we still parse whatever was buffered.
    result_reader.join(timeout=30)
    os.close(r_fd)

    payload = None
    line = bytes(result_buf).split(b"\n", 1)[0].strip()
    if line:
        try:
            payload = json.loads(line.decode("utf-8", errors="replace"))
        except ValueError:
            payload = None

    return {
        "payload": payload,
        "netns": netns,
        "unshare": unshare,
        "rc": rc,
        "killed_phase": killed_phase,
        "tail": list(tail),
    }
