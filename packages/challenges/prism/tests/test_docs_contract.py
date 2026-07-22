"""Documentation contract for the PRISM v2 product.

This suite pins the public docs (README + ``docs/**``) to the ACTUAL v2 system: the two-script
submission contract (``architecture.py``/``build_model`` + ``training.py``/``train``), the locked
FineWeb-Edu data plane (read-only, no network), the forced random-init validator re-execution,
**research-lab identity** (norm = try new architectures; goal = more performant ones), the
**dual param ladder** (124M explore provisional → 350M promote confirm/revoke), emission scoring
that is **held-out / generalization primary** with prequential bits-per-byte **secondary**
(Official-like), multimetric/Complete View as **published scientific research grade** (not the
emission scalar), **deterministic admission** (LLM hard gate and gateway removed), the single-node
multi-GPU contract, two-tier **0.50/0.50** ownership, and dry-run weights. It also asserts the docs
no longer reference the decommissioned v1-NAS machinery (component-review holds, ownership events,
the retired ``prism_run_manifest.v1.json``, or the removed ``local_cpu_smoke`` execution mode).

Assertions are anchored on real code constants so the docs cannot drift from the implementation.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from prism_challenge.evaluator.interface import (
    ARCHITECTURE_FACTORY_NAME,
    DEFAULT_ARCHITECTURE_ENTRYPOINT,
    DEFAULT_TRAINING_ENTRYPOINT,
    TRAINING_ENTRYPOINT_NAME,
)
from prism_challenge.evaluator.schemas import RUN_MANIFEST_V2_FILENAME, ExecutionMode

PUBLIC_DOCS = (
    "README.md",
    "docs/overview.md",
    "docs/architecture.md",
    "docs/submissions.md",
    "docs/scoring.md",
    "docs/official-comparison.md",
    "docs/scaling.md",
    "docs/security.md",
    "docs/api.md",
    "docs/operators.md",
    "docs/miner/README.md",
    "docs/miner/getting-started.md",
    "docs/miner/concepts.md",
    "docs/miner/troubleshooting.md",
    "docs/validator/README.md",
)


def read_doc(relative_path: str) -> str:
    return Path(relative_path).read_text(encoding="utf-8")


def all_docs_text() -> str:
    return "\n".join(read_doc(path) for path in PUBLIC_DOCS)


def test_two_script_contract_is_documented() -> None:
    readme = read_doc("README.md")
    submissions = read_doc("docs/submissions.md")
    miner = read_doc("docs/miner/README.md")
    getting_started = read_doc("docs/miner/getting-started.md")
    combined = f"{readme}\n{submissions}\n{miner}\n{getting_started}"

    assert "two-script" in combined.lower()
    assert DEFAULT_ARCHITECTURE_ENTRYPOINT in combined  # architecture.py
    assert DEFAULT_TRAINING_ENTRYPOINT in combined  # training.py
    assert f"{ARCHITECTURE_FACTORY_NAME}(ctx)" in combined  # build_model(ctx)
    assert f"{TRAINING_ENTRYPOINT_NAME}(ctx)" in combined  # train(ctx)
    # The miner owns the training loop; the challenge owns the data + evaluation.
    assert "the miner owns" in combined.lower()
    submissions_lower = submissions.lower()
    for expected in (
        "the challenge owns the dataset",
        "single combined module no longer satisfies",
    ):
        assert expected in submissions_lower


def test_miner_day1_getting_started_joinbase_bridge() -> None:
    """VAL-EMDOC-005: Phala-style day-1 miner path on joinbase + troubleshooting."""
    getting_started = read_doc("docs/miner/getting-started.md")
    troubleshooting = read_doc("docs/miner/troubleshooting.md")
    concepts = read_doc("docs/miner/concepts.md")
    miner_hub = read_doc("docs/miner/README.md")
    readme = read_doc("README.md")
    day1 = f"{getting_started}\n{miner_hub}\n{readme}"
    day1_lower = day1.lower()

    # Canonical public hosts (never platform.network as a shipping https URL).
    assert "https://joinbase.ai" in day1
    assert "https://chain.joinbase.ai" in day1
    assert "https://chain.platform.network" not in day1_lower
    assert "http://chain.platform.network" not in day1_lower

    # Bridge submit + leaderboard GET.
    assert "https://chain.joinbase.ai/v1/challenges/prism/submissions" in day1
    assert "https://chain.joinbase.ai/challenges/prism/leaderboard" in day1

    # Hotkey + pack seed + sign headers spine.
    assert "hotkey" in day1_lower
    assert "pack_seed_family" in day1
    assert "transformer-tiny-1m" in day1
    assert "X-Hotkey" in day1 and "X-Signature" in day1
    assert "X-Nonce" in day1 and "X-Timestamp" in day1
    assert (
        "prism:{hotkey}:{nonce}:{timestamp:" in day1
        or "prism:{hotkey}:{nonce}:{timestamp}" in day1
    )

    # Checklist present on day-1 page.
    assert "checklist" in getting_started.lower()

    # Troubleshooting is first-class (401/429/502).
    t_lower = troubleshooting.lower()
    assert "401" in troubleshooting and "429" in troubleshooting and "502" in troubleshooting
    assert "troubleshooting" in t_lower or "unauthorized" in t_lower

    # Science / scoring deep links live as Concepts, not only buried narrative.
    c_lower = concepts.lower()
    assert "held-out" in c_lower and "primary" in c_lower
    assert "official comparison" in c_lower or "multimetric" in c_lower
    assert "no-tee" in c_lower or "tee verifier" in c_lower
    assert "getting started" in concepts.lower() or "getting-started" in concepts.lower()

    # README badges / miner pointers to joinbase day-1.
    assert "joinbase" in readme.lower()
    assert "docs/miner/getting-started.md" in readme


def test_locked_fineweb_data_plane_is_documented() -> None:
    submissions = read_doc("docs/submissions.md")
    miner = read_doc("docs/miner/README.md")
    security = read_doc("docs/security.md")
    scaling = read_doc("docs/scaling.md")
    combined = f"{submissions}\n{miner}\n{security}\n{scaling}"

    assert "FineWeb-Edu" in combined
    assert "read-only" in combined
    assert "network=none" in combined
    assert "no network" in combined.lower()
    assert "HF_HUB_OFFLINE" in combined
    # Only the train split is miner-visible; val/test stay secret.
    assert "held-out" in combined.lower()
    assert "never exposed" in combined.lower() or "not exposed" in combined.lower()


def test_forced_init_re_execution_is_documented() -> None:
    architecture = read_doc("docs/architecture.md")
    security = read_doc("docs/security.md")
    submissions = read_doc("docs/submissions.md")
    combined = f"{architecture}\n{security}\n{submissions}"

    assert "re-execut" in combined.lower()
    assert "forced" in combined.lower()
    assert "random init" in combined.lower()
    assert "fixed seed" in combined.lower() or "forced seed" in combined.lower()
    # The challenge computes the score itself and ignores miner-reported numbers.
    assert "ignores" in combined.lower() or "ignored" in combined.lower()
    assert "miner-reported" in combined.lower()


def test_prequential_bpb_scoring_is_documented() -> None:
    scoring = read_doc("docs/scoring.md")
    lower = scoring.lower()

    assert "prequential" in lower
    assert "bits-per-byte" in lower
    assert "bpb" in lower
    assert "held-out" in lower
    assert "memorization" in lower
    assert "tokenizer-agnostic" in lower
    # Compute-normalized, never wall-clock.
    assert "compute-normalized" in lower
    assert "wall-clock" in lower
    assert "lower" in lower and "better" in lower
    # Emission is Official-like: held-out primary, bpb secondary (not bpb-primary emission).
    assert "held-out" in lower and "primary" in lower
    assert "secondary" in lower and "bpb" in lower
    assert "official comparison" in lower
    assert "leaderboard" in lower
    assert "bpb-primary" not in lower
    # Multimetric / Complete View is research grade, not silent emission replacement.
    assert "multimetric" in lower or "complete view" in lower
    assert "research grade" in lower or "scientific" in lower


def test_research_lab_identity_and_ladder_are_documented() -> None:
    """VAL-RESLAB-001: research-lab identity, norm/goal, small-first ladder 124M→350M."""
    readme = read_doc("README.md")
    overview = read_doc("docs/overview.md")
    scoring = read_doc("docs/scoring.md")
    submissions = read_doc("docs/submissions.md")
    miner = read_doc("docs/miner/README.md")
    combined = f"{readme}\n{overview}\n{scoring}\n{submissions}\n{miner}"
    lower = combined.lower()

    assert "research lab" in lower
    assert "new architecture" in lower or "new architectures" in lower
    assert "norm" in lower
    assert "goal" in lower
    assert "performant" in lower or "outperform" in lower or "more performant" in lower
    # Dual ladder numbers locked.
    assert "124" in combined and "350" in combined
    assert "124m" in lower or "124_000_000" in combined or "124000000" in combined
    assert "350m" in lower or "350_000_000" in combined or "350000000" in combined
    assert "provisional" in lower
    assert "promote" in lower
    assert "confirm" in lower or "confirms" in lower
    assert "revoke" in lower or "revokes" in lower
    # Emission held-out primary language on miner-facing surfaces.
    assert "held-out" in lower and "primary" in lower
    assert "emission" in lower
    # Default exploration seeds under 124M.
    assert "tiny-1m" in lower or "transformer-tiny-1m" in lower
    assert "mamba-tiny" in lower or "mamba-tiny-1m" in lower
    # Two-tier ownership 0.50/0.50.
    assert "0.50" in combined


def test_emission_vs_research_multimetric_surfaces_are_honest() -> None:
    """VAL-RESLAB-002: emission held-out primary; multimetric is research grade."""
    scoring = read_doc("docs/scoring.md")
    protocol = read_doc("docs/official-comparison.md")
    overview = read_doc("docs/overview.md")
    submissions = read_doc("docs/submissions.md")
    combined = f"{scoring}\n{protocol}\n{overview}\n{submissions}"
    lower = combined.lower()

    # Emission crown locked.
    assert "emission" in lower
    assert "held-out" in lower and "primary" in lower
    assert "secondary" in lower and "bpb" in lower
    # Multimetric / Complete View = published scientific / research grade, not silent emission.
    assert "multimetric.v1.1" in combined or "multimetric" in lower
    assert "complete view" in lower or "complete_view" in lower
    assert "research grade" in lower or "scientific" in lower or "published research" in lower
    assert (
        "silently replace" in lower
        or "do not silently" in lower
        or "does not silently" in lower
        or "not the emission" in lower
        or "emission scalar" in lower
        or "not silent" in lower
    )
    # Must not claim multimetric is the emission crown path via bpb-primary emission language.
    assert "bpb-primary" not in scoring.lower()
    assert "bpb-primary" not in overview.lower()


def test_scale_p3_full_scale_readiness_and_k_ge_3_docs_lock() -> None:
    """VAL-SCALE-015/016/017: full_scale readiness, public K≥3, research annex non-emission."""
    scaling = read_doc("docs/scaling.md")
    protocol = read_doc("docs/official-comparison.md")
    combined = f"{scaling}\n{protocol}"
    lower = combined.lower()

    # Full-scale / sample-100BT mode path documented (VAL-SCALE-015).
    assert "full_scale_eval" in combined or "full_scale" in lower
    assert "sample-100bt" in lower or "sample-100BT" in combined or "100bt" in lower
    assert "sample-10bt" in lower or "sample-10BT" in combined
    assert "probe_full_scale_readiness" in combined
    assert "blocked_with_reason" in lower or "blocked" in lower
    assert "dry-run" in lower or "dry run" in lower
    assert "100bt spend" in lower or "without requiring" in lower or "no 100bt" in lower

    # Public multi-seed K≥3 lock + ladder explore→promote→full_scale (VAL-SCALE-016).
    assert "k≥3" in lower or "k>=3" in lower or "k≥3" in combined
    assert "provisional" in lower
    assert "k=1" in lower or "k = 1" in lower
    assert "explore" in lower and "promote" in lower and "full_scale" in lower
    assert "scale_ladder" in lower or "scale ladder" in lower or "explore → promote" in lower

    # Research protocol annex explicitly non-emission (VAL-SCALE-017).
    assert "research_protocol_annex" in combined or "research protocol annex" in lower
    assert "non_emission" in lower or "non-emission" in lower
    assert "emission_weight_crown" in lower or "not silent emission" in lower
    assert "research grade" in lower
    assert "scale_p3_protocol_pin" in combined


def test_open_arch_norm_is_documented() -> None:
    """VAL-RESLAB-010: novel nn.Module expected; seeds pack; no family emission shortcuts."""
    readme = read_doc("README.md")
    overview = read_doc("docs/overview.md")
    submissions = read_doc("docs/submissions.md")
    scoring = read_doc("docs/scoring.md")
    miner = read_doc("docs/miner/README.md")
    combined = f"{readme}\n{overview}\n{submissions}\n{scoring}\n{miner}"
    lower = combined.lower()

    # Open-arch expected, not second-class.
    assert "nn.module" in lower or "torch.nn.module" in lower
    assert (
        "expected" in lower
        or "first-class" in lower
        or "welcome" in lower
        or "not second-class" in lower
    )
    assert "new architecture" in lower or "new architectures" in lower or "novel" in lower
    assert "deeploop" in lower or "looped" in lower
    assert "ast" in lower
    # Default seeds remain a first stop under explore 124M.
    assert "tiny-1m" in lower or "transformer-tiny-1m" in lower
    assert "mamba-tiny" in lower or "mamba-tiny-1m" in lower
    # Architecture-agnostic emission path; no family-specific shortcuts.
    assert "architecture-agnostic" in lower or "architecture agnostic" in lower
    scoring_lower = scoring.lower()
    assert "architecture-agnostic" in scoring_lower or "architecture agnostic" in scoring_lower
    assert "family-specific" in scoring_lower or "no family" in scoring_lower


def test_official_comparison_protocol_v1_is_documented() -> None:
    """Official Comparison Protocol v1: held-out primary, bpb secondary,
    honest hooks, GPU deferred."""
    protocol = read_doc("docs/official-comparison.md")
    lower = protocol.lower()

    assert "prism official comparison protocol v1" in lower
    assert "prism_official_compare.v1" in lower or "protocol v1" in lower
    # Ranking: held-out/generalization PRIMARY, prequential bpb SECONDARY.
    assert "primary" in lower and "held-out" in lower
    assert "secondary" in lower and "bpb" in lower
    assert "wall-clock never" in lower or "wall-clock" in lower
    # Dual ladder present on compare surface (124M explore default).
    assert "124" in protocol and ("param" in lower or "parameter" in lower or "ladder" in lower)
    assert "matched" in lower and ("token" in lower or "budget" in lower)
    assert "build_model" in protocol
    assert "train" in lower
    assert "iter_train_batches" in protocol
    assert "never authoritative" in lower or "non-authoritative" in lower
    assert "miner" in lower and (
        "self-report" in lower or "self report" in lower or "reported" in lower
    )
    assert "real-provider" in lower
    assert "orthogonal" in lower
    assert "deferred" in lower and ("nvidia" in lower or "gpu" in lower)
    # Multi-seed residual rule present.
    assert "multi-seed" in lower or "multi seed" in lower or "seeds" in lower
    # Scientific grade honesty: multimetric is not emission scalar.
    assert "research grade" in lower or "scientific" in lower
    assert "emission" in lower


def test_official_comparison_scorecard_v1_1_is_documented() -> None:
    """Official Comparison multimetric scorecard v1.1 (VAL-SCORE-001, VAL-SCORE-012)."""
    protocol = read_doc("docs/official-comparison.md")
    operators = read_doc("docs/operators.md")
    scoring = read_doc("docs/scoring.md")
    lower = protocol.lower()
    op_lower = operators.lower()
    scoring_lower = scoring.lower()

    # Annex identity: additive multimetric.v1.1 on prism_official_compare.v1 (not sole v2 crown).
    assert "multimetric.v1.1" in protocol
    assert "scorecard_id" in lower
    assert "prism_official_compare.v1" in lower
    assert (
        "multi-metric scorecard" in lower
        or "multimetric scorecard" in lower
        or "scorecard annex" in lower
    )
    assert "not a full v2" in lower or "sole weighted crown" in lower
    # Tiers V/P/S/R + A→Z metric catalogue anchors.
    assert "validity" in lower
    assert "short-gen" in lower or "heldout_delta" in lower
    assert "needle" in lower
    assert "mqar" in lower or "associative recall" in lower
    assert "induction" in lower or "copy" in lower
    assert "lag" in lower and ("nll" in lower or "lag_nll" in lower or "lag-nll" in lower)
    assert "sample-efficiency" in lower or "sample efficiency" in lower or "sample_eff" in lower
    assert "memorization" in lower or "memo_gap" in lower
    assert "vram" in lower
    assert "tokens" in lower and ("per" in lower or "tok/s" in lower or "tokens_per_s" in lower)
    assert "stability" in lower or "nan" in lower
    assert "multi-seed" in lower
    assert "k≥3" in lower or "k>=3" in lower or "k≥3" in protocol
    # Default v1 rank preserved + mandatory TIE_POLAR / crown_allowed=false.
    assert "primary_heldout" in lower or "held-out primary" in lower or "v1 preserved" in lower
    assert "tie_polar" in lower
    assert "crown_allowed" in lower and "false" in lower
    assert "polar" in lower

    # Honesty: prior LAB-GPU K=1 short-ctx mamba heldout lead is provisional only.
    assert "provisional" in lower
    assert "k=1" in lower or "k = 1" in lower
    assert "mamba" in lower
    assert "architecture superiority" in lower or "architecture-superiority" in lower
    assert "insufficient" in lower or "provisional only" in lower

    # Non-claim: no REAL-PROVIDER TEE unlock from scorecard / LAB-GPU.
    assert "real-provider" in lower
    assert "blocked" in lower
    assert "never" in lower or "not" in lower

    # Operator + scoring callouts cover the annex (VAL-SCORE-001 evidence surface).
    assert "multimetric.v1.1" in operators or "scorecard" in op_lower
    assert "tie_polar" in op_lower
    assert "provisional" in op_lower
    assert "multimetric.v1.1" in scoring or "scorecard" in scoring_lower
    assert "tie_polar" in scoring_lower
    assert "provisional" in scoring_lower


def test_official_comparison_complete_view_v1_2_is_documented() -> None:
    """Complete View v1.2 protocol history still documented (VAL-COMPLETE-001)."""
    protocol = read_doc("docs/official-comparison.md")
    operators = read_doc("docs/operators.md")
    lower = protocol.lower()
    op_lower = operators.lower()

    assert "complete view" in lower
    assert "complete_view.v1.2" in protocol
    assert "multimetric.complete.v1.2" in protocol
    assert "scorecard_id" in lower
    assert "multimetric.v1.1" in protocol  # historical relation
    assert "multi-axis" in lower or "multi axis" in lower or "per-axis" in lower
    assert "tie_polar" in lower
    assert "crown_allowed" in lower
    assert "opaque" in lower and ("weighted" in lower or "sole crown" in lower)
    assert "real-provider" in lower
    assert "blocked" in lower
    # Must-have matrix anchors (docs catalogue, not suite fills).
    assert "val_bpb" in lower or "val_bpb_trained" in protocol
    assert "needle" in lower
    assert "mqar" in lower
    assert "length" in lower and "extrap" in lower
    assert "disagreement" in lower or "per-axis" in lower

    # Operators pointer covers Complete View (v1.3 primary, v1.2 history ok).
    assert "complete view" in op_lower or "complete_view" in op_lower
    assert (
        "multimetric.complete.v1.2" in operators
        or "complete_view.v1.2" in operators
        or "multimetric.complete.v1.3" in operators
        or "complete_view.v1.3" in operators
    )
    assert "tie_polar" in op_lower
    assert "blocked" in op_lower


def test_official_comparison_complete_view_v1_3_reasoning_is_documented() -> None:
    """Complete View v1.3 P10 reasoning/logic identity + honesty (VAL-REASON-001/012)."""
    protocol = read_doc("docs/official-comparison.md")
    operators = read_doc("docs/operators.md")
    lower = protocol.lower()
    op_lower = operators.lower()

    assert "complete_view.v1.3" in protocol
    assert "multimetric.complete.v1.3" in protocol
    assert "p10_reasoning_logic" in lower
    assert "multimetric.complete.v1.2" in protocol  # history preserved
    assert "logic_synthetic" in lower or "synthetic" in lower
    assert "closed" in lower and ("forced" in lower or "ce" in lower)
    assert "chance" in lower
    assert "reasoning" in lower
    # Honesty non-claims for seed-scale logic (VAL-REASON-012).
    assert "gsm8k" in lower or "mmlu" in lower
    assert "human" in lower or "agi" in lower
    assert "lab" in lower or "diagnostic" in lower
    assert "seed-scale" in lower or "seed scale" in lower or "~7m" in lower
    assert "real-provider" in lower and "blocked" in lower

    assert "complete_view.v1.3" in operators or "multimetric.complete.v1.3" in operators
    assert "reasoning" in op_lower or "p10" in op_lower
    assert "blocked" in op_lower


def test_train_series_telemetry_protocol_is_documented() -> None:
    """Challenge-owned prism_train_series.v1 protocol docs (VAL-TELE-001)."""
    protocol = read_doc("docs/official-comparison.md")
    operators = read_doc("docs/operators.md")
    submissions = read_doc("docs/submissions.md")
    lower = protocol.lower()
    op_lower = operators.lower()
    sub_lower = submissions.lower()

    # Schema identity + ownership (VAL-TELE-001).
    assert "prism_train_series.v1" in protocol
    assert "challenge-owned" in lower or "challenge owned" in lower
    assert "authority" in lower
    assert "miner" in lower and (
        "non-authoritative" in lower
        or "never authoritative" in lower
        or "ignored" in lower
        or "self-report" in lower
    )

    # Point / channel catalogue anchors.
    assert "grad_norm" in lower
    assert "clip" in lower
    assert "tokens_seen" in lower or "tokens" in lower
    assert "wall" in lower
    assert "online" in lower or "train_ce" in lower or "bpb" in lower

    # Fail-closed when Official grade requires series.
    assert "fail-closed" in lower or "fail closed" in lower
    assert "require_train_series" in lower or "require train series" in lower

    # Operators + submissions honesty surfaces.
    assert "prism_train_series.v1" in operators
    assert "challenge-owned" in op_lower or "challenge owned" in op_lower
    assert "grad_norm" in op_lower
    assert "fail-closed" in op_lower or "fail closed" in op_lower
    assert "prism_train_series.v1" in submissions
    assert "grad_norm" in sub_lower
    assert "clip" in sub_lower


def test_train_series_scientific_vs_emission_grade_is_documented() -> None:
    """Scientific multi-axis Official grade vs emission held-out primary + residual series
    (VAL-TELE-012, VAL-TELE-010; research-lab emission invert)."""
    protocol = read_doc("docs/official-comparison.md")
    scoring = read_doc("docs/scoring.md")
    operators = read_doc("docs/operators.md")
    submissions = read_doc("docs/submissions.md")
    lower = protocol.lower()
    scoring_lower = scoring.lower()
    op_lower = operators.lower()
    sub_lower = submissions.lower()

    # VAL-TELE-012: multi-axis Official/Complete View = scientific miner grade (research);
    # emission is held-out primary + bpb secondary (not bpb-primary emission).
    assert "scientific" in lower
    assert "official comparison" in lower or "complete view" in lower
    assert "multi-axis" in lower or "multi axis" in lower
    assert "emission" in lower
    assert "held-out" in lower and "primary" in lower
    assert "bpb" in lower and "secondary" in lower
    assert "leaderboard" in lower
    # Legacy bpb-primary emission language must not remain as an emission claim.
    assert "bpb-primary" not in lower

    assert "scientific" in scoring_lower
    assert "prism_train_series.v1" in scoring
    assert "held-out" in scoring_lower and "primary" in scoring_lower
    assert "bpb" in scoring_lower and "secondary" in scoring_lower
    assert "leaderboard" in scoring_lower
    assert "bpb-primary" not in scoring_lower

    assert "scientific" in op_lower or "research grade" in op_lower or "emission" in op_lower
    assert "emission" in op_lower
    assert "held-out" in op_lower or "heldout" in op_lower
    assert "bpb-primary" not in op_lower

    assert "scientific" in sub_lower or "research" in sub_lower
    assert "emission" in sub_lower
    assert "held-out" in sub_lower and "primary" in sub_lower
    assert "bpb-primary" not in sub_lower

    # VAL-TELE-010: series residual only — visibility + sample-eff/stability densify;
    # never sole primary.
    assert "sample-eff" in lower or "sample efficiency" in lower or "sample_eff" in lower
    assert "stability" in lower
    assert "residual" in lower
    assert "never sole" in lower or "sole primary" in lower
    assert "not" in lower or "never" in lower

    assert "residual" in scoring_lower or "never sole" in scoring_lower
    assert (
        "sample-eff" in scoring_lower
        or "sample-efficiency" in scoring_lower
        or "stability" in scoring_lower
    )
    assert "residual" in op_lower or "never sole" in op_lower
    assert "sample-eff" in op_lower or "sample-efficiency" in op_lower or "stability" in op_lower
    assert "residual" in sub_lower or "never sole" in sub_lower


def test_llm_hard_gate_is_documented() -> None:
    """Docs say the LLM hard gate/gateway are removed; admission is deterministic."""
    security = read_doc("docs/security.md")
    operators = read_doc("docs/operators.md")
    combined = f"{security}\n{operators}"
    combined_lower = combined.lower()

    assert "deterministic admission" in combined_lower
    assert "llm hard gate" in combined_lower or "llm gateway" in combined_lower
    assert "removed" in combined_lower
    # Residual gateway knobs are fail-closed; no live provider/gateway path remains.
    assert "/llm/v1" not in combined
    assert "X-Gateway-Token" not in combined
    # A reject is terminal and stops the pipeline before any GPU work.
    assert "reject" in combined_lower
    assert "before any gpu" in combined_lower or "before gpu" in combined_lower


def test_no_tee_provider_trust_and_image_pin_are_documented() -> None:
    """VAL-NOTEE-011: provider trust + IMAGE_PIN; no TEE-required production scoring path."""
    readme = read_doc("README.md")
    architecture = read_doc("docs/architecture.md")
    security = read_doc("docs/security.md")
    scoring = read_doc("docs/scoring.md")
    operators = read_doc("docs/operators.md")
    overview = read_doc("docs/overview.md")
    api = read_doc("docs/api.md")
    validator = read_doc("docs/validator/README.md")
    protocol = read_doc("docs/official-comparison.md")
    config_example = Path("config.example.yaml").read_text(encoding="utf-8")

    product_docs = (
        f"{readme}\n{architecture}\n{security}\n{scoring}\n{operators}\n"
        f"{overview}\n{api}\n{validator}"
    )
    product_lower = product_docs.lower()
    all_lower = f"{product_docs}\n{protocol}".lower()

    # Provider trust + IMAGE_PIN narrative on shipping surfaces.
    assert "provider_trust" in product_lower or "provider trust" in product_lower
    assert (
        "image_pin" in product_lower
        or "image pin" in product_lower
        or "pinned_image_digest" in product_docs
    )
    assert "lium" in product_lower and "targon" in product_lower
    assert "pinned_image_digest" in config_example
    # config.example.yaml must not reintroduce a product tee: block.
    assert "\ntee:" not in config_example
    assert not any(line.strip().startswith("tee:") for line in config_example.splitlines())

    # Production path must NOT instruct TEE-required scoring / active TeeVerifier.
    forbidden_production = (
        "tee-required scoring",
        "tee required mode",
        "enable tee-required",
        "prism-only, fail-closed local tee fixture verifier",
        "prism is the only tee-attestation verifier",
    )
    for phrase in forbidden_production:
        assert phrase not in product_lower, f"production TEE path still instructed: {phrase}"

    # Explicit no-TEE product stance on core surfaces.
    assert "does not" in product_lower and (
        "tee verifier" in product_lower or "tee-attestation verifier" in product_lower
    )
    assert "retired" in all_lower and "real-provider" in all_lower

    # Historical REAL-PROVIDER BLOCKED honesty may remain on lab/protocol surfaces.
    assert "real-provider" in all_lower
    assert "blocked" in all_lower


def test_multi_gpu_contract_is_documented() -> None:
    submissions = read_doc("docs/submissions.md")
    scaling = read_doc("docs/scaling.md")
    miner = read_doc("docs/miner/README.md")
    combined = f"{submissions}\n{scaling}\n{miner}"

    assert "torchrun --standalone --nnodes=1 --nproc-per-node=1" in combined
    assert "single-node" in combined.lower()
    assert "up to 8" in combined.lower() or "1-8" in combined
    assert "nproc=1" in combined
    assert "gloo" in combined.lower()
    assert "ddp" in combined.lower()


def test_weight_push_and_validator_submission_are_documented() -> None:
    combined = all_docs_text()
    combined_lower = combined.lower()

    assert "get_weights" in combined
    assert "raw" in combined_lower and "weight" in combined_lower
    # Challenge/master never write on-chain; validators submit under own wallets.
    assert "on-chain" in combined_lower
    assert "validator" in combined_lower


def test_v2_manifest_filename_is_documented() -> None:
    submissions = read_doc("docs/submissions.md")
    miner = read_doc("docs/miner/README.md")
    combined = f"{submissions}\n{miner}"

    assert RUN_MANIFEST_V2_FILENAME in combined  # prism_run_manifest.v2.json
    # The challenge authors it; any miner-written manifest is discarded.
    assert "challenge-authored" in combined.lower() or "challenge authors" in combined.lower()


def test_execution_modes_match_code() -> None:
    validator = read_doc("docs/validator/README.md")

    for mode in ExecutionMode:
        assert mode.value in validator
    # The retired local CPU smoke mode is gone from the enum and the docs.
    assert "local_cpu_smoke" not in validator


def test_docs_do_not_reference_decommissioned_machinery() -> None:
    combined = all_docs_text()
    combined_lower = combined.lower()

    forbidden = (
        "prism_run_manifest.v1",
        "component_review_holds",
        "ownership_events",
        "component_agent_reviews",
        "local_cpu_smoke",
        "run_local_cpu_smoke",
        "/internal/v1/component-review",
        "/internal/v1/worker/poll",
        "/internal/v1/validators/assignments",
        "neural architecture search",
        ".omo",
        "prometheus",
        "metis",
        "workflow artifacts",
    )
    for phrase in forbidden:
        assert phrase.lower() not in combined_lower, (
            f"decommissioned reference still present: {phrase}"
        )


def test_readme_describes_the_v2_product() -> None:
    readme = read_doc("README.md")
    readme_lower = readme.lower()

    for expected in (
        "research lab",
        "new architecture",
        "two-script",
        "FineWeb-Edu",
        "prequential",
        "bits-per-byte",
        "held-out",
        "LLM gateway",
        "deterministic",
        "validator",
        "124",
        "350",
        "0.50",
    ):
        assert expected.lower() in readme_lower, f"README missing v2 concept: {expected}"
    assert "tiny-1m" in readme_lower or "transformer-tiny-1m" in readme_lower
    assert "mamba-tiny" in readme_lower


def test_api_doc_only_lists_live_internal_routes() -> None:
    api = read_doc("docs/api.md")

    for live_route in (
        "/internal/v1/get_weights",
        "/internal/v1/bridge/submissions",
        "/internal/v1/worker/process-next",
    ):
        assert live_route in api
    for removed_route in (
        "/internal/v1/worker/poll",
        "/internal/v1/component-review/holds",
        "/internal/v1/validators/assignments",
    ):
        assert removed_route not in api


def test_architecture_lab_routes_replace_dead_nas_routes(client: TestClient) -> None:
    """The architecture-lab serving layer revives ``/v1/architectures`` and re-homes variants.

    The original v1-NAS standalone ``/v1/training-variants`` listing stayed dead (its writers were
    removed in the NAS decommission, commit ``12376e6``); training variants are now nested under
    ``/v1/architectures/{id}/variants``. ``/v1/architectures`` is a LIVE grouped leaderboard again
    (the lab producer repopulates ``architecture_families`` / ``training_variants``), so this
    contract pins the new reality: the grouped listing is served (200 with the documented shape),
    while the retired flat ``/v1/training-variants`` path remains absent (404, no route registered).
    """

    architectures = client.get("/v1/architectures")
    assert architectures.status_code == 200, architectures.text
    body = architectures.json()
    assert set(body.keys()) == {"epoch_id", "architectures"}
    assert isinstance(body["architectures"], list)

    # The old flat variant listing is gone; variants now hang off an architecture.
    assert client.get("/v1/training-variants").status_code == 404

    # The nested variant route is registered (an unknown architecture is a 404 *from the handler*,
    # never a 405/route-missing): a missing architecture yields the handler's not-found detail.
    nested = client.get("/v1/architectures/does-not-exist/variants")
    assert nested.status_code == 404, nested.text
    assert nested.json()["detail"] == "architecture not found"
