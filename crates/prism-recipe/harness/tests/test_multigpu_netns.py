"""Multi-GPU netns wrapper: shape, shlex quoting, and `lo` actually UP.

`unshare --net` leaves loopback DOWN, which breaks single-node multi-GPU
rendezvous (torch.distributed `env://` on 127.0.0.1). `netns_child_cmd` brings
`lo` up inside the namespace without widening the isolation boundary.
"""

import os
import shlex
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prismlib.runner import netns_child_cmd  # noqa: E402


def test_passthrough_when_not_isolating():
    argv = [sys.executable, "-m", "prismlib.miner_entry", "/tmp/ctx.json"]
    assert netns_child_cmd(False, argv) == argv
    assert netns_child_cmd(None, argv) == argv
    # Returns a copy, never the caller's list object.
    out = netns_child_cmd(False, argv)
    out.append("mutated")
    assert "mutated" not in argv


def test_wrapper_shape():
    argv = [sys.executable, "-m", "prismlib.miner_entry", "/tmp/ctx.json"]
    cmd = netns_child_cmd(True, argv)
    assert cmd[:5] == ["unshare", "--net", "--", "sh", "-c"], cmd
    assert len(cmd) == 6, cmd
    script = cmd[5]
    # Loopback is brought up, guarded on `ip` being present, and failures are
    # non-fatal so a missing iproute2 degrades to plain isolation.
    assert "command -v ip >/dev/null 2>&1 &&" in script, script
    assert "ip link set lo up 2>/dev/null;" in script, script
    # The child replaces the shell — no extra process in the supervision tree.
    assert "exec " in script, script
    assert script.index("exec ") > script.index("ip link set lo up"), script


def test_shlex_quoting():
    # Paths with spaces / quotes / shell metacharacters must survive intact.
    nasty = "/tmp/a b/ctx';touch /tmp/pwned;'.json"
    argv = [sys.executable, "-m", "prismlib.miner_entry", nasty]
    script = netns_child_cmd(True, argv)[5]
    inner = script.split("exec ", 1)[1]
    # Round-trip: the shell would rebuild exactly the argv we handed in.
    assert shlex.split(inner) == argv, inner
    assert shlex.quote(nasty) in script, script


def test_lo_is_up_inside_namespace():
    """When unshare is usable, `lo` must really be UP inside the child."""
    if shutil.which("unshare") is None:
        print("SKIP lo-up check: unshare not in PATH")
        return
    probe = subprocess.run(
        ["unshare", "--net", "--", "true"], capture_output=True, text=True, timeout=30
    )
    if probe.returncode != 0:
        print(f"SKIP lo-up check: unshare probe rc={probe.returncode}")
        return
    if shutil.which("ip") is None:
        print("SKIP lo-up check: iproute2 (`ip`) not installed")
        return

    # Baseline: without the wrapper, loopback is DOWN in a fresh netns.
    bare = subprocess.run(
        ["unshare", "--net", "--", "ip", "link", "show", "lo"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert bare.returncode == 0, bare.stderr
    assert "state DOWN" in bare.stdout, f"expected bare netns lo DOWN: {bare.stdout}"

    # With the wrapper, loopback is UP and 127.0.0.1 is bindable/connectable,
    # which is what torch.distributed env:// rendezvous needs.
    cmd = netns_child_cmd(True, ["ip", "link", "show", "lo"])
    up = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    assert up.returncode == 0, up.stderr
    assert "state UNKNOWN" in up.stdout or "state UP" in up.stdout, up.stdout
    assert "UP" in up.stdout.split(":", 2)[2].split(">")[0], up.stdout

    # Real rendezvous smoke: bind + connect on 127.0.0.1 inside the namespace.
    py = (
        "import socket;"
        "s=socket.socket();s.bind(('127.0.0.1',0));s.listen(1);"
        "c=socket.socket();c.connect(s.getsockname());"
        "a,_=s.accept();c.sendall(b'ok');"
        "print('LOOPBACK_OK' if a.recv(2)==b'ok' else 'BAD')"
    )
    rdv = subprocess.run(
        netns_child_cmd(True, [sys.executable, "-c", py]),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert rdv.returncode == 0, f"rc={rdv.returncode} err={rdv.stderr}"
    assert "LOOPBACK_OK" in rdv.stdout, rdv.stdout

    # Isolation is unchanged: the namespace still has no route off-host.
    routes = subprocess.run(
        netns_child_cmd(True, ["ip", "route", "show"]),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert routes.returncode == 0, routes.stderr
    assert routes.stdout.strip() == "", f"expected no routes: {routes.stdout!r}"


def test_both_spawn_sites_use_the_wrapper():
    """runner.run_miner_subprocess and v3flow.run_phase must both wrap."""
    here = os.path.dirname(__file__)
    for rel in ("prismlib/runner.py", "prismlib/v3flow.py"):
        src = open(os.path.join(here, "..", rel)).read()
        assert "netns_child_cmd(" in src, f"{rel} does not use netns_child_cmd"
        # No open-coded `unshare` argv left behind at the spawn sites.
        assert '["unshare", "--net", "--"]' not in src, rel
        assert 'cmd.extend(["unshare"' not in src, rel


def main():
    test_passthrough_when_not_isolating()
    test_wrapper_shape()
    test_shlex_quoting()
    test_lo_is_up_inside_namespace()
    test_both_spawn_sites_use_the_wrapper()
    print("MULTIGPU NETNS OK")


if __name__ == "__main__":
    main()
