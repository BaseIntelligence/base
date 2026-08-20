#!/usr/bin/env python3
"""Focused smoke: recipe 2.0 AutoModel staging + fixture entry (no NVIDIA clone).

Exercises FIXTURE_PIN + happy.patch apply, prism.toml entry knobs, and the
fixture markers. Does **not** claim a real AutoModel train (no EVAL_OK).
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HARNESS_ROOT = Path(__file__).resolve().parents[1]
# harness/ -> prism-recipe/ -> crates/prism-automodel/fixtures
FIXTURE_ROOT = (HARNESS_ROOT.parents[1] / "prism-automodel" / "fixtures").resolve()


def main():
    assert FIXTURE_ROOT.is_dir(), f"missing fixtures at {FIXTURE_ROOT}"
    sys.path.insert(0, str(HARNESS_ROOT))
    os.environ["PRISM_AUTOMODEL_FIXTURE"] = "1"
    os.environ["PRISM_AUTOMODEL_FIXTURE_ROOT"] = str(FIXTURE_ROOT)

    from prismlib import automodel as am

    # --- unit: stage + invoke ---
    out = am.fixture_smoke()
    manifest = out["manifest"]
    assert manifest["stub"] is True
    assert manifest["real_train"] is False
    assert manifest["base"] == am.FIXTURE_PIN_ID
    assert manifest["entry"] == am.DEFAULT_TRAIN_ENTRY
    assert out["result"]["via"] == "main"

    applied = Path(manifest["applied_dir"])
    assert (applied / "nemo_automodel/components/models/toy/layers.py").is_file()
    model_py = (applied / "nemo_automodel/components/models/toy/model.py").read_text(
        encoding="utf-8"
    )
    assert "width: int = 16" in model_py

    # --- prism.toml entry override ---
    with tempfile.TemporaryDirectory(prefix="prism-am-toml-") as td:
        td = Path(td)
        toml = td / "prism.toml"
        toml.write_text(
            '[prism]\nentry = "nemo_automodel/recipes/llm/train_ft.py"\n'
            'model_config = "nemo_automodel/components/models/toy/model.py"\n',
            encoding="utf-8",
        )
        os.environ[am.ENV_PRISM_TOML] = str(toml)
        os.environ[am.ENV_FIXTURE] = "1"
        os.environ[am.ENV_FIXTURE_ROOT] = str(FIXTURE_ROOT)
        m2 = am.stage(td / "w", force_fixture=True)
        assert m2["entry"].endswith("train_ft.py")
        assert m2["model_config"].endswith("model.py")
        seams = am.write_seams(td / "w", m2)
        assert Path(seams["arch_path"]).is_file()
        assert Path(seams["train_path"]).is_file()

    # --- main.py fixture short-circuit (no FineWeb / no EVAL_OK) ---
    with tempfile.TemporaryDirectory(prefix="prism-am-main-") as td:
        env = os.environ.copy()
        env["PRISM_WORKDIR"] = td
        env["PRISM_AUTOMODEL_FIXTURE"] = "1"
        env["PRISM_AUTOMODEL_FIXTURE_ROOT"] = str(FIXTURE_ROOT)
        # Ensure live dataset vars are absent / ignored on fixture path.
        env.pop("PRISM_DATASET_URL", None)
        env.pop("PRISM_DATASET_SHA256", None)
        r = subprocess.run(
            [sys.executable, str(HARNESS_ROOT / "main.py")],
            cwd=str(HARNESS_ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        combined = (r.stdout or "") + "\n" + (r.stderr or "")
        assert r.returncode == 0, combined[-2000:]
        assert "AUTOMODEL_FIXTURE_STAGED" in combined
        assert "AUTOMODEL_FIXTURE_ENTRY_OK" in combined
        assert "AUTOMODEL_FIXTURE_OK" in combined
        assert "EVAL_OK" not in combined, "fixture stub must not emit EVAL_OK"
        metrics_line = [
            ln for ln in combined.splitlines() if ln.startswith("METRICS_JSON=")
        ]
        assert metrics_line, combined[-1000:]
        m = json.loads(metrics_line[-1].split("=", 1)[1])
        assert m.get("automodel_stub") is True
        assert m.get("real_train") is False
        assert m.get("recipe") == "2.1.0"

    # --- pin+patch env path (non-fixture applied dir simulation) ---
    with tempfile.TemporaryDirectory(prefix="prism-am-pin-") as td:
        td = Path(td)
        # Copy pin + apply via stage with explicit dirs (fixture flag off,
        # but pin is still the tiny fixture tree — live NVIDIA not required).
        pin = FIXTURE_ROOT / "automodel-pin"
        patch = FIXTURE_ROOT / "patches" / "happy.patch"
        os.environ.pop(am.ENV_FIXTURE, None)
        os.environ[am.ENV_PIN_DIR] = str(pin)
        os.environ[am.ENV_PATCH_PATH] = str(patch)
        os.environ[am.ENV_BASE] = am.LIVE_PIN_ID
        os.environ.pop(am.ENV_PRISM_TOML, None)
        os.environ.pop(am.ENV_APPLIED_DIR, None)
        m3 = am.stage(td)
        assert m3["stub"] is False
        assert m3["real_train"] is True
        assert m3["source"] == "pin_patch"
        assert (Path(m3["applied_dir"]) / "nemo_automodel/components/models/toy/layers.py").is_file()

    # --- intake tree_blob layout: submission/nemo_automodel + .prism/ ---
    with tempfile.TemporaryDirectory(prefix="prism-am-sub-") as td:
        td = Path(td)
        for k in (am.ENV_PIN_DIR, am.ENV_PATCH_PATH, am.ENV_APPLIED_DIR, am.ENV_FIXTURE):
            os.environ.pop(k, None)
        # Re-stage via pin+patch into a temp applied tree, then reshape as submission/.
        mid = td / "mid"
        os.environ[am.ENV_PIN_DIR] = str(FIXTURE_ROOT / "automodel-pin")
        os.environ[am.ENV_PATCH_PATH] = str(FIXTURE_ROOT / "patches" / "happy.patch")
        os.environ[am.ENV_BASE] = am.FIXTURE_PIN_ID
        m_mid = am.stage(mid)
        os.environ.pop(am.ENV_PIN_DIR, None)
        os.environ.pop(am.ENV_PATCH_PATH, None)
        work = td / "pod"
        sub = work / "submission"
        import shutil

        shutil.copytree(m_mid["applied_dir"], sub)
        (sub / ".prism").mkdir(exist_ok=True)
        (sub / ".prism" / "automodel.base").write_text(
            am.FIXTURE_PIN_ID + "\n", encoding="utf-8"
        )
        (sub / ".prism" / "automodel.patch").write_text(
            (FIXTURE_ROOT / "patches" / "happy.patch").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        assert am.layout_requested(work)
        m4 = am.stage(work)
        assert m4["source"] == "submission_tree_blob"
        assert m4["stub"] is True  # fixture pin id
        assert (Path(m4["applied_dir"]) / "nemo_automodel").is_dir()

    print("automodel stage smoke OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
