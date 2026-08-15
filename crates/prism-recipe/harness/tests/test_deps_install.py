"""Miner dependency-install phase: manifest discovery + pip argv (pure)."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prismlib import deps  # noqa: E402


def main():
    # No manifest → no-op discovery.
    with tempfile.TemporaryDirectory() as d:
        assert deps.find_manifest(d) is None
        assert deps.install_miner_deps(d) is None

    # requirements.txt discovered and wins over pyproject.toml.
    with tempfile.TemporaryDirectory() as d:
        req = os.path.join(d, "requirements.txt")
        open(req, "w").write("flash-attn==2.6.3\n")
        open(os.path.join(d, "pyproject.toml"), "w").write("[project]\nname='x'\n")
        kind, path = deps.find_manifest(d)
        assert kind == "requirements" and path == req, (kind, path)
        cmd = deps.build_install_cmd(kind, path)
        assert cmd[:4] == [sys.executable, "-m", "pip", "install"], cmd
        assert "--break-system-packages" in cmd
        assert cmd[-2:] == ["-r", req], cmd

    # pyproject-only → pip install <dir>.
    with tempfile.TemporaryDirectory() as d:
        proj = os.path.join(d, "pyproject.toml")
        open(proj, "w").write("[project]\nname='x'\nversion='0'\n")
        kind, path = deps.find_manifest(d)
        assert kind == "pyproject", kind
        cmd = deps.build_install_cmd(kind, path)
        assert cmd[-1] == d, cmd
        assert "-r" not in cmd

    # Staged-tree layout: a patch-added manifest lands under submission/.
    with tempfile.TemporaryDirectory() as d:
        sub = os.path.join(d, "submission")
        os.makedirs(sub)
        open(os.path.join(sub, "requirements.txt"), "w").write("einops\n")
        kind, path = deps.find_manifest(d)
        assert kind == "requirements" and path.startswith(sub), (kind, path)

    # Unknown kind is a hard error.
    try:
        deps.build_install_cmd("wheel", "/x")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

    # A failing install raises RuntimeError (routes to install_deps class).
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "requirements.txt"), "w").write(
            "this-package-does-not-exist-prism-xyzzy==9.9.9\n"
        )
        try:
            deps.install_miner_deps(d, timeout_s=120)
            raise AssertionError("expected RuntimeError on bad requirement")
        except RuntimeError:
            pass

    print("deps install phase OK")


if __name__ == "__main__":
    main()
