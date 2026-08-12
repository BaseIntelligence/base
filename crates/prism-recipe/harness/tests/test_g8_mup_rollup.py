"""Fail-closed org.g8.mup_lr_stability when the µP sweep runs and diverges."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval import rollup  # noqa: E402


def _g8_groups(metrics):
    return {
        "g8": {
            "status": "ok",
            "module": "g8_stability",
            "metrics": metrics,
            "partial": False,
        }
    }


def main():
    # Sweep succeeded: stability from g8.mup.stability.
    flat = rollup.flatten_metrics(
        _g8_groups(
            {
                "g8.divergence.series_nan_frac": 0.0,
                "g8.divergence.probe_nan_frac": 0.0,
                "g8.spikes.per_1k_steps": 0.0,
                "g8.mup.stub": 0.0,
                "g8.mup.lr_ratio_log2_abs": 1.0,
                "g8.mup.stability": 0.5,
            }
        ),
        [],
    )
    assert flat["org.g8.mup_lr_stability"] == 0.5, flat
    assert "org.g8.loss_spike_score" in flat

    # Sweep diverged: fail-closed floor 0.0 (must not omit the org key).
    flat = rollup.flatten_metrics(
        _g8_groups(
            {
                "g8.divergence.series_nan_frac": 0.0,
                "g8.divergence.probe_nan_frac": 0.0,
                "g8.mup.stub": 1.0,
                "g8.mup.stub_reason_sweep_diverged": 1.0,
                "g8.mup.stability": 0.0,
            }
        ),
        [],
    )
    assert flat.get("org.g8.mup_lr_stability") == 0.0, flat

    # Tiny-caps skip: no stability key → org key omitted.
    flat = rollup.flatten_metrics(
        _g8_groups(
            {
                "g8.divergence.series_nan_frac": 0.0,
                "g8.divergence.probe_nan_frac": 0.0,
                "g8.mup.stub": 1.0,
                "g8.mup.stub_reason_tiny_caps": 1.0,
            }
        ),
        [],
    )
    assert "org.g8.mup_lr_stability" not in flat, flat

    # Legacy success path without g8.mup.stability (ratio only).
    flat = rollup.flatten_metrics(
        _g8_groups(
            {
                "g8.divergence.series_nan_frac": 0.0,
                "g8.divergence.probe_nan_frac": 0.0,
                "g8.mup.stub": 0.0,
                "g8.mup.lr_ratio_log2_abs": 0.0,
            }
        ),
        [],
    )
    assert flat["org.g8.mup_lr_stability"] == 1.0, flat
    print("g8 mup rollup OK")


if __name__ == "__main__":
    main()
