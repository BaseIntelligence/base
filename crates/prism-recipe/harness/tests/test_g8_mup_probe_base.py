"""G8 µP sweep uses a reduced fixed probe base, not full production geometry."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval import g8_stability  # noqa: E402


def main():
    production = {
        "d_model": 1024,
        "n_layer": 24,
        "n_head": 16,
        "mlp_hidden": 2560,
        "vocab_size": 50257,
        "device": "cuda",
        "seed": 7,
        "arch": {"d_model": 1024, "n_layer": 24, "extra_ignored": True},
        "prism_width_multiplier": 99.0,  # must be stripped before sweep
        "tokenizer": object(),
    }
    probe = g8_stability.mup_probe_base_ctx(production)

    # Width/depth replaced by fixed probe; vocab/device/seed preserved.
    assert probe["d_model"] == 128, probe
    assert probe["n_layer"] == 4, probe
    assert probe["n_head"] == 4, probe
    assert probe["mlp_hidden"] == 320, probe
    assert probe["vocab_size"] == 50257, probe
    assert probe["device"] == "cuda", probe
    assert probe["seed"] == 7, probe
    assert "prism_width_multiplier" not in probe, probe
    assert probe["arch"]["d_model"] == 128, probe["arch"]
    assert probe["arch"]["n_layer"] == 4, probe["arch"]
    # Non-geometry arch keys survive.
    assert probe["arch"].get("extra_ignored") is True, probe["arch"]

    # Production build_ctx must not be mutated.
    assert production["d_model"] == 1024
    assert production["arch"]["d_model"] == 1024
    assert production["prism_width_multiplier"] == 99.0

    # Empty / None build_ctx still yields a usable probe.
    bare = g8_stability.mup_probe_base_ctx(None)
    assert bare["d_model"] == 128
    assert bare["arch"]["d_model"] == 128

    # Reference baseline honors probe overrides + width multiplier without OOM
    # geometry (CPU, no train): 4× params must exceed 1.5× of 1×.
    sys.path.insert(
        0,
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "baselines",
            "transformer_pp",
        ),
    )
    import architecture as tpp  # noqa: E402

    b1 = dict(probe)
    b1["prism_width_multiplier"] = 1.0
    b1["vocab_size"] = 256
    m1 = tpp.build_model(b1)
    n1 = sum(p.numel() for p in m1.parameters())

    b4 = dict(probe)
    b4["prism_width_multiplier"] = 4.0
    b4["vocab_size"] = 256
    m4 = tpp.build_model(b4)
    n4 = sum(p.numel() for p in m4.parameters())
    assert n4 > int(1.5 * n1), (n1, n4)
    # Sanity: probe 4× stays far under the 350M submission cap.
    assert n4 < 50_000_000, n4

    print(f"g8 mup probe base OK (1x={n1} 4x={n4})")


if __name__ == "__main__":
    main()
