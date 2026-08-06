"""PRISM eval battery registry (G1–G8).

Loader convention: each module `gN_<topic>.py` in this package exposes

    run(model, ctx) -> dict[str, float]

pure stdlib + torch, CPU-testable with tiny caps. Battery modules land in a
later recipe step (E2); this registry already tolerates their absence: a
group with no module is "missing", a module that fails to import is
"error", a module without a callable `run` (or whose `run` raises
NotImplementedError) is "stub". One battery module must never kill the
harness — errors are recorded per group.
"""

import importlib
import pkgutil

EXPECTED_GROUPS = ("g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8")


def discover():
    """Map group id -> {"status", "module", ...} for present gN_* modules."""
    found = {}
    for info in pkgutil.iter_modules(__path__):
        name = info.name
        group = name.split("_", 1)[0]
        if group not in EXPECTED_GROUPS or group in found:
            continue
        try:
            mod = importlib.import_module(f".{name}", __name__)
        except Exception as exc:  # noqa: BLE001
            found[group] = {
                "status": "error",
                "module": name,
                "detail": f"import: {str(exc)[:200]}",
            }
            continue
        run = getattr(mod, "run", None)
        if not callable(run):
            found[group] = {"status": "stub", "module": name, "detail": "no run()"}
        else:
            found[group] = {"status": "ok", "module": name, "run": run}
    return found


def status_summary():
    """{group: "ok"|"stub"|"error"|"missing"} — cheap, no battery execution."""
    disc = discover()
    return {g: ("missing" if g not in disc else disc[g]["status"]) for g in EXPECTED_GROUPS}


def run_battery(model, ctx):
    """Run every available group; per-group failures are captured, not raised."""
    results = {}
    disc = discover()
    for group in EXPECTED_GROUPS:
        entry = disc.get(group)
        if entry is None:
            results[group] = {"status": "missing"}
            continue
        if entry["status"] != "ok":
            results[group] = {k: v for k, v in entry.items() if k != "run"}
            continue
        try:
            metrics = entry["run"](model, ctx)
            results[group] = {
                "status": "ok",
                "module": entry["module"],
                "metrics": {str(k): float(v) for k, v in dict(metrics).items()},
            }
        except NotImplementedError:
            results[group] = {"status": "stub", "module": entry["module"]}
        except Exception as exc:  # noqa: BLE001
            results[group] = {
                "status": "error",
                "module": entry["module"],
                "detail": str(exc)[:200],
            }
    return results
