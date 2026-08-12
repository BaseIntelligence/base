"""Parent-side v3 two-phase subprocess runner (recipe 1.3.0, E2).

Genericized sibling of `prismlib.runner` for the v3 flow: spawns an
arbitrary `python3 -m <module> <ctx.json>` child under `unshare --net`
(when available) in its **own process group**, applies per-phase timeout
budgets driven by `PRISM_PHASE=` marker lines, drains the fd result
channel concurrently, then **hard-kills the process group** on exit and
checks for survivors (the post-train gate: no miner process may outlive
the train phase — private eval assets only land after this check).

Returns the same dict shape as runner.run_miner_subprocess plus
`survivors` (None when the check was not possible).
"""

import collections
import json
import os
import signal
import subprocess
import sys
import threading
import time

from .envutil import log
from .runner import PHASE_MARK, _open_result_channel, probe_unshare


def _kill_group(pgid):
    try:
        os.killpg(pgid, signal.SIGKILL)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return False


def _group_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by someone else — treat as alive


def run_phase(
    module,
    workdir,
    ctx_json_path,
    *,
    budgets,
    extra_env=None,
):
    """Spawn + supervise one v3 phase child.

    `budgets`: {phase_name: seconds} — the child announces phases via
    `PRISM_PHASE=` lines; unknown phases get the largest budget.
    Returns {"payload", "netns", "unshare", "rc", "killed_phase",
             "tail", "survivors"}.
    """
    unshare = probe_unshare()
    netns = bool(unshare["available"])
    if not netns:
        log(
            "WARNING: network isolation OFF ("
            + unshare["detail"]
            + "); v3 phase subprocess shares the pod network (fallback)"
        )

    cmd = []
    if netns:
        cmd.extend(["unshare", "--net", "--"])
    cmd.extend([sys.executable, "-m", module, ctx_json_path])

    env = dict(os.environ)
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    harness_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prev_pp = env.get("PYTHONPATH", "")
    parts = [harness_root]
    automodel = os.path.join(workdir, "automodel")
    if os.path.isdir(automodel):
        parts.append(automodel)
    submission = os.path.join(workdir, "submission")
    if os.path.isdir(submission):
        parts.append(submission)  # last among harness roots: no shadowing
    if prev_pp:
        parts.append(prev_pp)
    env["PYTHONPATH"] = os.pathsep.join(parts)
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
        start_new_session=True,  # own process group: killable as a unit
    )
    pgid = proc.pid
    os.close(w_fd)
    if alias_fd is not None:
        os.close(alias_fd)

    state = {"phase": next(iter(budgets), "run"), "phase_start": time.monotonic()}
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
                    log(f"v3 phase -> {state['phase']}")
                else:
                    tail.append(line)
                    log(f"v3| {line[:500]}")
        except ValueError:
            pass

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()

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

    killed_phase = None
    while True:
        if proc.poll() is not None:
            break
        phase = state["phase"]
        budget = float(budgets.get(phase, max(budgets.values())))
        if time.monotonic() - state["phase_start"] > budget:
            killed_phase = phase
            log(f"v3 phase '{phase}' exceeded budget {budget:.0f}s — killing process group")
            _kill_group(pgid)
            break
        time.sleep(0.5)

    try:
        rc = proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        rc = proc.wait(timeout=30)
    reader.join(timeout=30)
    result_reader.join(timeout=30)
    os.close(r_fd)

    # Post-exit gate: the whole group must be dead before we return.
    _kill_group(pgid)
    survivors = _group_alive(pgid)
    if survivors:
        log("WARNING: survivor processes in v3 phase group after SIGKILL")

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
        "survivors": survivors,
    }
