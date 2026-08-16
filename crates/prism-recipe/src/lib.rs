//! Normative PRISM recipe (`docs/PRISM_RECIPE.md`).
//!
//! PRISM is a **recipe-contract challenge**: the miner submits custom code
//! (`architecture.py` defining `build_model(ctx)` and `training.py` defining
//! `train(model, ctx)`) that runs **inside the operator-owned harness** on a
//! Lium pod. The harness supplies everything the contract promises — "all the
//! data" the miner can request:
//!
//! - the pinned pretraining shard (fineweb-edu sample, pinned URL + SHA-256),
//! - seed, tokenizer, device, step/time caps,
//! - a frozen val stream and its tokenization,
//! - the exact BPB lattice (val CE / ln 2).
//!
//! The same objects are exposed to miners over `GET /v1/recipe` and
//! `GET /v1/recipe/baseline` so they can run the harness locally before
//! submitting. The v3 flow adds a **staged eval pack** (default
//! `eval_tier=public`): after the train phase completes, the operator
//! stages held-out eval assets (built from public HF datasets) plus a
//! generator seed onto the pod (post-train, over SSH) and the battery runs
//! against them. Staging is fail-closed — if the assets cannot be staged
//! or verified, the eval terminates rather than falling back to the tiny
//! embedded `public_dev` fixtures. An optional `private` tier remains for
//! secret contamination mirrors. Tiny public anchors (`eval/public_dev/`)
//! remain published for local miner reproduction.
//!
//! The harness itself is a multi-file Python package embedded at build time
//! ([`HARNESS_FILES`]) and streamed to the pod over SSH by `prism-lium` as
//! one deterministic tar on stdin. No `PyPI` packages at runtime: it only uses stdlib +
//! `torch` + `transformers` + `datasets` preinstalled in the pinned pod
//! image. [`HARNESS_PY`] remains as a legacy single-file view (the package
//! entrypoint `main.py`).

#![forbid(unsafe_code)]

mod review_tree;
mod zip_submit;

pub mod anchors;
pub mod attribution;
pub mod baselines;

pub use attribution::{build_attribution_runs, AttributionCell, AttributionError, AttributionRun};
pub use review_tree::materialize_review_sources;
pub use zip_submit::{
    probe_zip_kind, sources_from_zip, training_from_zip, tree_from_zip, SourceTree, ZipKind,
    ZipSubmitError, DEFAULT_TREE_ENTRY, MAX_TOKENIZER_FILES, MAX_TOKENIZER_TOTAL_BYTES,
    MAX_TREE_FILES, MAX_TREE_FILE_BYTES, MAX_TREE_TOTAL_BYTES, TOKENIZER_DIR, TOKENIZER_EXTENSIONS,
    TREE_MANIFEST_FILE, VENDOR_LOCK_FILE,
};

/// Embedded harness package: pod-relative path → file contents, uploaded
/// file-by-file by `prism-lium` into the pod workdir. Layout: `main.py`
/// (parent entrypoint), `prismlib/*.py` (library modules: miner subprocess
/// runner, seeded train stream, G6 probes, harness-owned scoring, pod
/// manifest, v3 two-phase entries, cheatguard screening), plus the
/// `cheatguard_patterns.json` banned-pattern list, and `eval/` (G1–G8
/// battery registry + battery modules + procedural generators + the
/// vendored community protocols (`vendor_*.py` + `VENDOR.md` provenance)
/// + the public dev family).
///
/// Keep sorted by path — [`harness_files_sha256`] and [`recipe_pin_hex`]
/// hash the set in sorted-path order.
pub const HARNESS_FILES: &[(&str, &str)] = &[
    (
        "cheatguard_patterns.json",
        include_str!("../harness/cheatguard_patterns.json"),
    ),
    ("eval/VENDOR.md", include_str!("../harness/eval/VENDOR.md")),
    (
        "eval/__init__.py",
        include_str!("../harness/eval/__init__.py"),
    ),
    ("eval/common.py", include_str!("../harness/eval/common.py")),
    (
        "eval/g1_intrinsic.py",
        include_str!("../harness/eval/g1_intrinsic.py"),
    ),
    (
        "eval/g2_downstream.py",
        include_str!("../harness/eval/g2_downstream.py"),
    ),
    (
        "eval/g3_recall.py",
        include_str!("../harness/eval/g3_recall.py"),
    ),
    (
        "eval/g4_reasoning.py",
        include_str!("../harness/eval/g4_reasoning.py"),
    ),
    (
        "eval/g5_babilong.py",
        include_str!("../harness/eval/g5_babilong.py"),
    ),
    (
        "eval/g5_longctx.py",
        include_str!("../harness/eval/g5_longctx.py"),
    ),
    (
        "eval/g5_ruler.py",
        include_str!("../harness/eval/g5_ruler.py"),
    ),
    (
        "eval/g6_curve.py",
        include_str!("../harness/eval/g6_curve.py"),
    ),
    (
        "eval/g7_inference.py",
        include_str!("../harness/eval/g7_inference.py"),
    ),
    (
        "eval/g8_stability.py",
        include_str!("../harness/eval/g8_stability.py"),
    ),
    (
        "eval/gen_longctx.py",
        include_str!("../harness/eval/gen_longctx.py"),
    ),
    (
        "eval/gen_reasoning.py",
        include_str!("../harness/eval/gen_reasoning.py"),
    ),
    (
        "eval/generators.py",
        include_str!("../harness/eval/generators.py"),
    ),
    (
        "eval/natural_docs.py",
        include_str!("../harness/eval/natural_docs.py"),
    ),
    (
        "eval/public_dev/README.md",
        include_str!("../harness/eval/public_dev/README.md"),
    ),
    (
        "eval/public_dev/g1/domains/code.jsonl",
        include_str!("../harness/eval/public_dev/g1/domains/code.jsonl"),
    ),
    (
        "eval/public_dev/g1/domains/news.jsonl",
        include_str!("../harness/eval/public_dev/g1/domains/news.jsonl"),
    ),
    (
        "eval/public_dev/g2/arc_challenge.jsonl",
        include_str!("../harness/eval/public_dev/g2/arc_challenge.jsonl"),
    ),
    (
        "eval/public_dev/g2/arc_easy.jsonl",
        include_str!("../harness/eval/public_dev/g2/arc_easy.jsonl"),
    ),
    (
        "eval/public_dev/g2/boolq.jsonl",
        include_str!("../harness/eval/public_dev/g2/boolq.jsonl"),
    ),
    (
        "eval/public_dev/g2/hellaswag.jsonl",
        include_str!("../harness/eval/public_dev/g2/hellaswag.jsonl"),
    ),
    (
        "eval/public_dev/g2/lambada.jsonl",
        include_str!("../harness/eval/public_dev/g2/lambada.jsonl"),
    ),
    (
        "eval/public_dev/g2/openbookqa.jsonl",
        include_str!("../harness/eval/public_dev/g2/openbookqa.jsonl"),
    ),
    (
        "eval/public_dev/g2/piqa.jsonl",
        include_str!("../harness/eval/public_dev/g2/piqa.jsonl"),
    ),
    (
        "eval/public_dev/g2/winogrande.jsonl",
        include_str!("../harness/eval/public_dev/g2/winogrande.jsonl"),
    ),
    (
        "eval/public_dev/g5/natural/README.md",
        include_str!("../harness/eval/public_dev/g5/natural/README.md"),
    ),
    (
        "eval/public_dev/g5/natural/helmet_rag.demos.jsonl",
        include_str!("../harness/eval/public_dev/g5/natural/helmet_rag.demos.jsonl"),
    ),
    (
        "eval/public_dev/g5/natural/helmet_rag.jsonl",
        include_str!("../harness/eval/public_dev/g5/natural/helmet_rag.jsonl"),
    ),
    (
        "eval/public_dev/g5/natural/natural_mcq.jsonl",
        include_str!("../harness/eval/public_dev/g5/natural/natural_mcq.jsonl"),
    ),
    (
        "eval/public_dev/seeds.json",
        include_str!("../harness/eval/public_dev/seeds.json"),
    ),
    ("eval/rollup.py", include_str!("../harness/eval/rollup.py")),
    ("eval/toklen.py", include_str!("../harness/eval/toklen.py")),
    (
        "eval/vendor_babilong.py",
        include_str!("../harness/eval/vendor_babilong.py"),
    ),
    (
        "eval/vendor_ruler.py",
        include_str!("../harness/eval/vendor_ruler.py"),
    ),
    ("main.py", include_str!("../harness/main.py")),
    (
        "prismlib/__init__.py",
        include_str!("../harness/prismlib/__init__.py"),
    ),
    (
        "prismlib/automodel.py",
        include_str!("../harness/prismlib/automodel.py"),
    ),
    (
        "prismlib/cheatguard.py",
        include_str!("../harness/prismlib/cheatguard.py"),
    ),
    (
        "prismlib/dataset.py",
        include_str!("../harness/prismlib/dataset.py"),
    ),
    (
        "prismlib/deps.py",
        include_str!("../harness/prismlib/deps.py"),
    ),
    (
        "prismlib/envutil.py",
        include_str!("../harness/prismlib/envutil.py"),
    ),
    (
        "prismlib/eval_v3.py",
        include_str!("../harness/prismlib/eval_v3.py"),
    ),
    (
        "prismlib/flops.py",
        include_str!("../harness/prismlib/flops.py"),
    ),
    (
        "prismlib/manifest.py",
        include_str!("../harness/prismlib/manifest.py"),
    ),
    (
        "prismlib/miner_entry.py",
        include_str!("../harness/prismlib/miner_entry.py"),
    ),
    (
        "prismlib/probes.py",
        include_str!("../harness/prismlib/probes.py"),
    ),
    (
        "prismlib/runner.py",
        include_str!("../harness/prismlib/runner.py"),
    ),
    (
        "prismlib/scoring.py",
        include_str!("../harness/prismlib/scoring.py"),
    ),
    (
        "prismlib/stream.py",
        include_str!("../harness/prismlib/stream.py"),
    ),
    (
        "prismlib/telemetry.py",
        include_str!("../harness/prismlib/telemetry.py"),
    ),
    (
        "prismlib/tokenizer.py",
        include_str!("../harness/prismlib/tokenizer.py"),
    ),
    (
        "prismlib/train_v3.py",
        include_str!("../harness/prismlib/train_v3.py"),
    ),
    (
        "prismlib/v3flow.py",
        include_str!("../harness/prismlib/v3flow.py"),
    ),
];

/// Legacy single-file view of the harness: the `main.py` parent entrypoint.
/// Kept for consumers written against the pre-1.3.0 single-file harness.
pub const HARNESS_PY: &str = include_str!("../harness/main.py");

/// Baseline submission: `architecture.py`.
pub const BASELINE_ARCHITECTURE_PY: &str = include_str!("../baseline/architecture.py");

/// Baseline submission: `training.py`.
pub const BASELINE_TRAINING_PY: &str = include_str!("../baseline/training.py");

/// Recipe semantic version (surfaced through the API). **2.0.0** requires the
/// `AutoModel` pin + miner unified-diff ZIP layout (`automodel.base` +
/// `automodel.patch`); legacy 1.x two-script / source-tree / `arch_id`
/// layouts are rejected on live (`unsupported_layout` / `recipe_version`).
/// Caps, `FineWeb` pin, telemetry, and the G1–G8 battery from 1.4 remain
/// unless a later bump says otherwise. See `docs/PRISM_RECIPE.md`.
pub const RECIPE_VERSION: &str = "2.0.0";

/// Maximum model parameters allowed after `build_model` (1B).
///
/// Raised 350M → 1B alongside the 4×RTX 5090 pod rental (recipe-v10): the
/// wall-clock budget is unchanged (6h), so the cap buys architectural
/// headroom rather than a longer run. Placeholder anchors and the public
/// GPT-2 Large reference row MUST be re-measured at this cap before any
/// `PRISM_ANCHOR_VERSION=2` / composite governance flip.
pub const MAX_PARAMS: u64 = 1_000_000_000;

/// Assets-dir–relative home of the G5 natural-document packs
/// (LongBench-v2 MCQ + HELMET RAG pools, their disjoint `public_dev`
/// mirror and the build manifest). Built operator-side by
/// `cargo run -p xtask -- natural-pack --out "$PRISM_EVAL_ASSETS_DIR"` and
/// carried to the pod by the existing post-train staging, which tars the
/// whole assets dir; `eval/natural_docs.py` resolves the same relative
/// path through `common.assets_path`.
pub const NATURAL_PACK_REL: &str = "g5/natural";

/// Cap on the packed eval-assets tar `prism-lium` streams to the pod over
/// ssh stdin. The natural-document pools at [`NATURAL_PACK_REL`] dominate
/// the assets dir — the G1/G2 mirrors are tiny by comparison — but raw
/// documents gzip well and a default pack lands around 10 MiB, so the cap
/// is unchanged and still refuses an assets dir that is obviously wrong.
/// Raised from 64 → 256 MiB so a full public pack (G1/G2 + LongBench-v2 /
/// HELMET natural pools) can stage without forcing tiny `public_dev`
/// natural fixtures. Gzipped packs still land well under this in practice.
pub const MAX_EVAL_ASSETS_PACKED_BYTES: usize = 256 * 1024 * 1024;

/// Pinned pretraining shard: fineweb-edu sample-10BT, single parquet shard.
///
/// Pin triple: URL + sha256 + license. The harness verifies the SHA-256
/// before training; a mismatch fails the eval `ChallengeInternal`.
pub const DATASET_URL: &str =
    "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/010_00000.parquet";

/// Canonical dataset ref (HF dataset id).
pub const DATASET_REF: &str = "HuggingFaceFW/fineweb-edu@sample/10BT";

/// Expected byte length of the pinned shard (operator-verified; the harness
/// still trust-verifies cryptographically via [`dataset_sha256`]).
#[must_use]
pub fn dataset_len_bytes() -> u64 {
    2_152_798_864
}

/// Expected SHA-256 of the pinned shard (hex lowercase). Operator-verified
/// by downloading the pin once and hashing; the upload path re-verifies.
///
/// See `docs/PRISM_RECIPE.md` for the update ceremony.
pub const DATASET_SHA256: &str = "e5a2eae25f057f0856a10bfae314c6ca8ea8bb08456d2131e9e89b2b8305e2f6";

/// Overrideable SHA-256 (`PRISM_DATASET_SHA256`); defaults to the build pin.
#[must_use]
pub fn dataset_sha256() -> String {
    std::env::var("PRISM_DATASET_SHA256")
        .ok()
        .map(|v| v.trim().to_owned())
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| DATASET_SHA256.into())
}

/// Train wall-clock cap per submission (**hours**) — a **safety bound**, not
/// the budget currency.
///
/// Lowered 6.0 → 5.0 when [`TRAIN_FLOPS_CAP`] became the currency. The pair
/// is chosen so FLOPs, not the clock, binds for essentially the whole field:
/// at `C_MAX = 3.0e18` on 4×RTX 5090 (838 TFLOPS peak bf16) the wall needed
/// is `C_MAX / (838e12 × MFU)` = 4.97 h at 20 % MFU, 3.98 h at 25 %, 3.31 h
/// at 30 %. So any implementation at ≥ 20 % MFU is FLOPs-bound and the
/// kernel lottery stops being a scored quantity. A slower implementation
/// still terminates — that is the anti-DoS job this cap keeps.
///
/// Asserted against the FLOPs cap in `tests::wall_bound_is_slack_at_target_mfu`.
pub const TRAIN_HOURS_CAP: f64 = 5.0;

/// Attested-FLOPs cap per submission — **the budget currency**.
///
/// Counted by the harness (`prismlib/flops.py`) with
/// `torch.utils.flop_counter.FlopCounterMode` over harness-driven
/// forward+backward passes on batches at secret stream indices, times the
/// harness-owned `stream.tokens_seen`. The miner never reports a number.
///
/// Because the meter is the *realized* dispatch graph, the budget is
/// class-adaptive for free: a model looped `r` times pays ~`r`× per token
/// (not exactly `r`× — `lm_head` and attention do not loop), a `MoE` pays only
/// for the experts that actually ran, and a big vocabulary pays for its head
/// (`6·d·V`). There is no size tier to declare and none to shop for.
pub const TRAIN_FLOPS_CAP: f64 = 3.0e18;

/// Eligibility floor: a run must attest at least this fraction of
/// [`TRAIN_FLOPS_CAP`] to be scored.
///
/// The underspend guard. Without it, an architecture that saturates early
/// profits by stopping early — it is then compared against a weaker
/// truncation reference (the compute-optimal frontier's slope is only
/// ≈ −0.05..−0.10 nats per e-fold, so buying less compute costs less score
/// than it saves). Below this floor the run is ineligible, not merely scaled.
pub const MIN_SPEND_FRACTION: f64 = 0.5;

/// Global eval-battery wall budget (seconds) the harness battery declares
/// (`eval.common.BATTERY_BUDGET_S`), mirrored here so the Rust pod
/// arithmetic and the Python battery cannot drift apart.
///
/// One global budget with per-group ceilings as fractional shares; the old
/// independent per-group ceilings summed to ≈ 3.92 h under a 3 h phase kill.
pub const EVAL_GLOBAL_BUDGET_S: f64 = 3600.0;

/// Number of `FlopCounterMode` probes at secret stream indices.
pub const FLOPS_PROBE_SAMPLES: u32 = 8;

/// Max coefficient of variation across the probe samples before the run is
/// flagged `flops_probe_unstable` (and the **max**, not the median, is used).
///
/// A high CV is the signature of input-dependent cost — an `MoE` that routes
/// to fewer experts on probe-shaped inputs, or an early-exit path. Probes
/// are drawn from the real train stream at secret indices precisely so they
/// are indistinguishable from training batches.
pub const FLOPS_PROBE_CV_MAX: f64 = 0.15;

/// Max relative disagreement between the dispatch counter and the analytic
/// FLOPs/token estimate before the run is flagged.
///
/// `FlopCounterMode` only sees what the `PyTorch` dispatcher sees, so a fused
/// Triton/CUDA kernel registered as one opaque op is invisible — and
/// recipe-v10 lets miners install their own dependencies, which makes that
/// reachable. The analytic model
/// (`6·N_body·r_eff + 6·d·V + 12·L·d·S`, `MoE` **active** experts only) is the
/// cross-check. This is the largest residual risk in the design, so the gap
/// is a visible metric (`org.diag.flops_analytic_ratio`), never a silent pass.
pub const FLOPS_ANALYTIC_GAP_MAX: f64 = 0.25;

/// Harness build-phase ceiling (`PRISM_BUILD_TIMEOUT_S` default, seconds).
pub const HARNESS_BUILD_TIMEOUT_S: f64 = 900.0;

/// Harness checkpoint/score-phase ceiling (`PRISM_SCORE_TIMEOUT_S`, seconds).
pub const HARNESS_SCORE_TIMEOUT_S: f64 = 1800.0;

/// Harness eval-phase ceiling (`PRISM_EVAL_TIMEOUT_S` default, seconds):
/// model load + G1-G8 battery (`eval.common.BATTERY_BUDGET_S` = 3600) +
/// rollup + scoring.
pub const HARNESS_EVAL_TIMEOUT_S: f64 = 5400.0;

/// Pod lifetime cap total (**hours**), sized to actually contain the
/// harness it rents rather than a round guess.
///
/// Derivation — the pod must strictly contain both children:
///
/// ```text
/// train child : build 900 + train (5 h = 18000) + grace 120 + checkpoint 1800 = 20820 s
/// eval child  : PRISM_EVAL_TIMEOUT_S 5400 (battery 3600 + load/rollup/score reserve)
/// worst case  : 26220 s = 7.28 h        ⇒ 7.5 h cap leaves 780 s of margin
/// ```
///
/// **History of the 7.0-vs-8.5 disagreement.** 7.0 was "6 h train + 1 h",
/// which the harness's own phase ceilings already broke (the train child
/// alone is 6.78 h at a 6 h train cap, and the eval child gets its whole
/// timeout on top) — a full-budget submission could be killed mid-eval and
/// lose the entire miner-funded rental, so a previous pass raised it to 8.5.
/// Both were right about their own arithmetic and wrong about each other's:
/// 8.5 is the correct ceiling **for a 6 h train cap**, and the design's 7.0
/// does not fit **any** train cap once the real phase ceilings are added
/// (7.0 h would need `PRISM_EVAL_TIMEOUT_S ≤ 4380 s`, i.e. 60 s of reserve
/// above the 3600 s battery — not workable). With [`TRAIN_HOURS_CAP`] now
/// 5.0 h the arithmetic closes at **7.5 h**, which is both below the old 8.5
/// and above the design's 7.0.
///
/// A higher ceiling never raises the bill for a run that finishes early
/// (pods are billed for time used); it only stops the orchestrator from
/// killing a run the recipe itself permits. `prism_lium_payer::sealed`
/// derives its TTL from these same constants, so the payer cannot drift.
/// Asserted in `tests::pod_lifetime_covers_train_plus_eval`.
pub const POD_LIFETIME_HOURS_CAP: f64 = 7.5;

/// Effective train wall-clock cap (hours). Production is always
/// [`TRAIN_HOURS_CAP`]; `PRISM_TEST_TRAIN_MINUTES` (staging/e2e only, works
/// for Sim and real Lium) shrinks it so a full eval fits in minutes.
#[must_use]
pub fn train_hours_cap() -> f64 {
    std::env::var("PRISM_TEST_TRAIN_MINUTES")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|m| *m > 0.0)
        .map_or(TRAIN_HOURS_CAP, |m| m / 60.0)
}

/// Effective attested-FLOPs cap. Production is always [`TRAIN_FLOPS_CAP`];
/// `PRISM_TEST_TRAIN_FLOPS` (staging/e2e only) shrinks it so a dual-cap run
/// completes in minutes, mirroring [`train_hours_cap`].
///
/// Non-finite / non-positive overrides are ignored rather than trusted, so a
/// malformed env var cannot silently disable the budget.
#[must_use]
pub fn train_flops_cap() -> f64 {
    std::env::var("PRISM_TEST_TRAIN_FLOPS")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|c| c.is_finite() && *c > 0.0)
        .unwrap_or(TRAIN_FLOPS_CAP)
}

/// Effective parameter cap. Production is always [`MAX_PARAMS`];
/// `PRISM_TEST_MAX_PARAMS` (staging/e2e only) selects a tiny-model profile.
#[must_use]
pub fn max_params() -> u64 {
    std::env::var("PRISM_TEST_MAX_PARAMS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|p| *p > 0)
        .unwrap_or(MAX_PARAMS)
}

/// Steps hard stop inside `train` (belt + clock). Config overrides only down.
pub const MAX_TRAIN_STEPS: u32 = 20_000;

/// Seed lattice (deterministic across miners; recipes share seed space).
pub const RECIPE_SEED: u64 = 0x0050_5249_534D;

/// Val stream rows frozen into evaluation (first N validated texts).
pub const VAL_ROWS: usize = 256;

/// Train rows from the pinned shard (`ctx.train_iter()`).
pub const TRAIN_ROWS: usize = 2_048;

/// Minimal sanity constraints every submission must satisfy **before** any
/// pod is rented — string-level only, the real check happens in-pod.
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum ContractError {
    /// architecture.py missing `def build_model(`.
    #[error("architecture_py must define build_model(ctx)")]
    MissingBuildModel,
    /// training.py missing `def train(`.
    #[error("training_py must define train(model, ctx)")]
    MissingTrain,
    /// `architecture_py` too large.
    #[error("architecture_py exceeds contract size ({0} bytes > {MAX_SOURCE_BYTES})")]
    ArchitectureTooLarge(usize),
    /// `training_py` too large.
    #[error("training_py exceeds contract size ({0} bytes > {MAX_SOURCE_BYTES})")]
    TrainingTooLarge(usize),
}

/// Per-source byte cap (schema only).
pub const MAX_SOURCE_BYTES: usize = 128 * 1024;

/// Validate contract shape (cheap local screen; in-pod check still authoritative).
///
/// # Errors
/// [`ContractError`] variants.
pub fn check_contract(architecture_py: &str, training_py: &str) -> Result<(), ContractError> {
    if architecture_py.len() > MAX_SOURCE_BYTES {
        return Err(ContractError::ArchitectureTooLarge(architecture_py.len()));
    }
    if training_py.len() > MAX_SOURCE_BYTES {
        return Err(ContractError::TrainingTooLarge(training_py.len()));
    }
    if !architecture_py.contains("def build_model(") {
        return Err(ContractError::MissingBuildModel);
    }
    if !training_py.contains("def train(") {
        return Err(ContractError::MissingTrain);
    }
    Ok(())
}

/// SHA-256 hex over the embedded harness file set ([`HARNESS_FILES`]),
/// sorted by pod-relative path: for each file, `path bytes || 0x00 ||
/// contents || 0xff` feed one SHA-256. Forwarded to the pod by `prism-lium`
/// as `PRISM_HARNESS_FILES_SHA256` and echoed back in `METRICS_JSON` v2 as
/// `harness_files_sha256` (the harness Python fallback recomputes it from
/// the uploaded files with the same algorithm).
#[must_use]
pub fn harness_files_sha256() -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    let mut files: Vec<&(&str, &str)> = HARNESS_FILES.iter().collect();
    files.sort_by(|a, b| a.0.cmp(b.0));
    for (path, contents) in files {
        h.update(path.as_bytes());
        h.update([0x00]);
        h.update(contents.as_bytes());
        h.update([0xff]);
    }
    hex::encode(h.finalize())
}

/// SHA-256 hex of the recipe contract tuple surfaced via the API. Covers
/// the full harness file set (via [`harness_files_sha256`]) since 1.3.0.
#[must_use]
pub fn recipe_pin_hex() -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(RECIPE_VERSION.as_bytes());
    h.update(MAX_PARAMS.to_le_bytes());
    h.update(DATASET_URL.as_bytes());
    h.update(dataset_sha256().as_bytes());
    h.update(harness_files_sha256().as_bytes());
    h.update(prism_automodel::AUTOMODEL_PIN.id.as_bytes());
    h.update(prism_automodel::AUTOMODEL_PIN.git_commit.as_bytes());
    h.update(prism_automodel::AUTOMODEL_PIN.content_sha256.as_bytes());
    hex::encode(h.finalize())
}

/// Public recipe descriptor (`GET /v1/recipe` body minus volatile bits).
#[derive(Debug, Clone, serde::Serialize)]
pub struct RecipeDescriptor {
    /// Semver of the recipe contract.
    pub version: &'static str,
    /// Pinned dataset URL (download+verify by the miner locally).
    pub dataset_url: &'static str,
    /// Canonical HF ref string.
    pub dataset_ref: &'static str,
    /// Pinned dataset SHA-256.
    pub dataset_sha256: String,
    /// Expected size hint (bytes).
    pub dataset_len_bytes: u64,
    /// Train wall-clock cap (hours) — safety bound, not the currency.
    pub train_hours_cap: f64,
    /// Attested-FLOPs cap (the budget currency).
    pub train_flops_cap: f64,
    /// Eligibility floor as a fraction of `train_flops_cap`.
    pub min_spend_fraction: f64,
    /// Global eval-battery wall budget (seconds).
    pub eval_global_budget_s: f64,
    /// `FlopCounterMode` probe count at secret stream indices.
    pub flops_probe_samples: u32,
    /// Max probe coefficient of variation before `flops_probe_unstable`.
    pub flops_probe_cv_max: f64,
    /// Max counter-vs-analytic relative gap before the run is flagged.
    pub flops_analytic_gap_max: f64,
    /// Pod lifetime cap hours.
    pub pod_lifetime_hours_cap: f64,
    /// Hard step stop.
    pub max_train_steps: u32,
    /// Seed lattice.
    pub seed: u64,
    /// Row caps.
    pub val_rows: usize,
    /// Train rows.
    pub train_rows: usize,
    /// Per-source byte cap.
    pub max_source_bytes: usize,
    /// Maximum model parameters after `build_model`.
    pub max_params: u64,
    /// Recipe contract pin (sha256 hex over the tuple).
    pub pin_hex: String,
    /// `AutoModel` pin id miners must echo in `automodel.base`.
    pub automodel_pin_id: &'static str,
    /// Upstream `AutoModel` repository URL.
    pub automodel_repo_url: &'static str,
    /// Frozen upstream tag (informational).
    pub automodel_git_ref: &'static str,
    /// Exact `AutoModel` commit the pin checkout must match.
    pub automodel_git_commit: &'static str,
    /// Content SHA-256 of the staged pin archive (empty until operator freeze).
    pub automodel_content_sha256: &'static str,
    /// Pod container image the harness executes in (CUDA/torch base). Miners
    /// build against this to know which wheels/toolchain are preinstalled.
    pub pod_image_ref: &'static str,
    /// Whether the pod runs a **network-on install phase** for miner
    /// dependency manifests before the netns-isolated train/eval.
    pub miner_install_supported: bool,
    /// ZIP/JSON members a miner may ship to install custom deps
    /// (`requirements.txt`, `pyproject.toml`).
    pub miner_deps_members: [&'static str; 2],
    /// Wall-clock cap (seconds) for the miner dependency-install phase.
    pub install_timeout_secs: u64,
}

/// Build the public descriptor (deterministic).
#[must_use]
pub fn descriptor() -> RecipeDescriptor {
    let pin = &prism_automodel::AUTOMODEL_PIN;
    RecipeDescriptor {
        version: RECIPE_VERSION,
        dataset_url: DATASET_URL,
        dataset_ref: DATASET_REF,
        dataset_sha256: dataset_sha256(),
        dataset_len_bytes: dataset_len_bytes(),
        train_hours_cap: TRAIN_HOURS_CAP,
        train_flops_cap: TRAIN_FLOPS_CAP,
        min_spend_fraction: MIN_SPEND_FRACTION,
        eval_global_budget_s: EVAL_GLOBAL_BUDGET_S,
        flops_probe_samples: FLOPS_PROBE_SAMPLES,
        flops_probe_cv_max: FLOPS_PROBE_CV_MAX,
        flops_analytic_gap_max: FLOPS_ANALYTIC_GAP_MAX,
        pod_lifetime_hours_cap: POD_LIFETIME_HOURS_CAP,
        max_train_steps: MAX_TRAIN_STEPS,
        seed: RECIPE_SEED,
        val_rows: VAL_ROWS,
        train_rows: TRAIN_ROWS,
        max_source_bytes: MAX_SOURCE_BYTES,
        max_params: MAX_PARAMS,
        pin_hex: recipe_pin_hex(),
        automodel_pin_id: pin.id,
        automodel_repo_url: pin.git_url,
        automodel_git_ref: pin.git_tag,
        automodel_git_commit: pin.git_commit,
        automodel_content_sha256: pin.content_sha256,
        pod_image_ref: POD_IMAGE_REF,
        miner_install_supported: MINER_INSTALL_SUPPORTED,
        miner_deps_members: [
            prism_automodel::MEMBER_REQUIREMENTS,
            prism_automodel::MEMBER_PYPROJECT,
        ],
        install_timeout_secs: INSTALL_TIMEOUT_SECS,
    }
}

/// Pod container image the harness runs in (advertised via `/v1/recipe`).
/// The recipe-v10 image ships CUDA 13, `PyTorch`, a full build toolchain,
/// and Transformer Engine so miners can NVFP4-train and `pip install`
/// extras (`FlashAttention`, `mamba-ssm`, …) from their own manifests.
pub const POD_IMAGE_REF: &str = "ghcr.io/baseintelligence/prism-pod:v10-cuda13-te";

/// The pod runs a network-on install phase for miner dependency manifests
/// before the netns-isolated train/eval (recipe-v10).
pub const MINER_INSTALL_SUPPORTED: bool = true;

/// Wall-clock cap (seconds) for the miner dependency-install phase.
pub const INSTALL_TIMEOUT_SECS: u64 = 1_800;

/// Validate the pinned dataset locally (download+hash) — no network in prod
/// callers; only run from tests or the operator CLI.
///
/// # Errors
/// Transport / hash mismatch.
pub fn verify_dataset_sha256(bytes: &[u8]) -> Result<(), String> {
    use sha2::{Digest, Sha256};
    let got = hex::encode(Sha256::digest(bytes));
    if got != dataset_sha256() {
        return Err(format!(
            "dataset sha256 mismatch: got {got}, want {}",
            dataset_sha256()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn harness_embedded_and_nonempty() {
        // HARNESS_PY is the multi-file package entrypoint (main.py).
        assert!(HARNESS_PY.contains("METRICS_JSON="));
        assert!(HARNESS_PY.contains("prismlib"));
        let all = harness_concat();
        assert!(all.contains("build_model"));
        assert!(all.contains("fineweb") || all.contains("DATASET_URL"));
    }

    fn harness_concat() -> String {
        HARNESS_FILES.iter().map(|(_, c)| *c).collect()
    }

    #[test]
    fn harness_files_layout_is_sorted_and_complete() {
        let paths: Vec<&str> = HARNESS_FILES.iter().map(|(p, _)| *p).collect();
        let mut sorted = paths.clone();
        sorted.sort_unstable();
        assert_eq!(paths, sorted, "HARNESS_FILES must stay sorted by path");
        let mut dedup = sorted.clone();
        dedup.dedup();
        assert_eq!(dedup.len(), sorted.len(), "duplicate harness paths");
        for required in [
            "main.py",
            "cheatguard_patterns.json",
            "eval/__init__.py",
            "eval/common.py",
            "eval/generators.py",
            "eval/gen_reasoning.py",
            "eval/gen_longctx.py",
            "eval/g1_intrinsic.py",
            "eval/g2_downstream.py",
            "eval/g3_recall.py",
            "eval/g4_reasoning.py",
            "eval/g5_babilong.py",
            "eval/g5_longctx.py",
            "eval/g5_ruler.py",
            "eval/toklen.py",
            "eval/vendor_babilong.py",
            "eval/vendor_ruler.py",
            "eval/VENDOR.md",
            "eval/g6_curve.py",
            "eval/g7_inference.py",
            "eval/g8_stability.py",
            "eval/natural_docs.py",
            "eval/rollup.py",
            "eval/public_dev/seeds.json",
            "eval/public_dev/README.md",
            "prismlib/__init__.py",
            "prismlib/automodel.py",
            "prismlib/cheatguard.py",
            "prismlib/runner.py",
            "prismlib/miner_entry.py",
            "prismlib/stream.py",
            "prismlib/probes.py",
            "prismlib/scoring.py",
            "prismlib/telemetry.py",
            "prismlib/tokenizer.py",
            "prismlib/manifest.py",
            "prismlib/dataset.py",
            "prismlib/envutil.py",
            "prismlib/train_v3.py",
            "prismlib/eval_v3.py",
            "prismlib/flops.py",
            "prismlib/deps.py",
            "prismlib/v3flow.py",
        ] {
            assert!(paths.contains(&required), "missing harness file {required}");
        }
        for required_g2 in [
            "lambada",
            "hellaswag",
            "piqa",
            "arc_easy",
            "arc_challenge",
            "winogrande",
            "boolq",
            "openbookqa",
        ] {
            let rel = format!("eval/public_dev/g2/{required_g2}.jsonl");
            assert!(
                paths.contains(&rel.as_str()),
                "missing public dev anchor {rel}"
            );
        }
        // G5 natural-document format fixtures: the code path must run (and
        // the mirror gap must degrade honestly) with no staged private pack.
        for required_natural in [
            "helmet_rag.demos.jsonl",
            "helmet_rag.jsonl",
            "natural_mcq.jsonl",
        ] {
            let rel = format!("eval/public_dev/{NATURAL_PACK_REL}/{required_natural}");
            assert!(
                paths.contains(&rel.as_str()),
                "missing natural fixture {rel}"
            );
        }
        let main = HARNESS_FILES
            .iter()
            .find(|(p, _)| *p == "main.py")
            .map(|(_, c)| *c);
        assert_eq!(main, Some(HARNESS_PY), "HARNESS_PY must alias main.py");
    }

    /// Every `prismlib` module the harness imports must be UPLOADED.
    ///
    /// `HARNESS_FILES` is the upload manifest: a module absent from it does
    /// not exist on the pod, however normal it looks on disk. This is a
    /// regression guard for a live break in which `prismlib/deps.py` was
    /// added and imported at module scope in `main.py` but never listed, so
    /// every real pod run died at import with `cannot import name 'deps'`
    /// while every local test passed — because local tests import from the
    /// source tree, where the file is present.
    ///
    /// The hardcoded list above cannot catch this class of bug: a new import
    /// with no matching entry is exactly what nobody remembers to add.
    #[test]
    fn every_imported_prismlib_module_is_uploaded() {
        let uploaded: Vec<&str> = HARNESS_FILES.iter().map(|(p, _)| *p).collect();
        let mut checked = 0_usize;
        for (path, source) in HARNESS_FILES {
            if !std::path::Path::new(path)
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("py"))
            {
                continue;
            }
            for line in source.lines() {
                let line = line.trim();
                // `from prismlib import x [as y]` / `from prismlib.x import y`
                let module = line
                    .strip_prefix("from prismlib import ")
                    .and_then(|rest| rest.split([',', ' ']).next())
                    .or_else(|| {
                        line.strip_prefix("from prismlib.")
                            .and_then(|rest| rest.split([' ', '.']).next())
                    });
                let Some(module) = module else { continue };
                let module = module.trim();
                // Skip non-module names re-exported by the package __init__
                // (constants like RECIPE_SEED, TRAIN_ROWS, VAL_ROWS).
                if module.is_empty() || module.chars().next().is_some_and(char::is_uppercase) {
                    continue;
                }
                let want = format!("prismlib/{module}.py");
                assert!(
                    uploaded.contains(&want.as_str()),
                    "{path} imports `prismlib.{module}` but {want} is NOT in \
                     HARNESS_FILES — the pod would die at import. Local tests \
                     pass because they read the source tree, not the upload."
                );
                checked += 1;
            }
        }
        assert!(
            checked >= 10,
            "import scan found only {checked} imports — the parser probably \
             stopped matching and this test is no longer guarding anything"
        );
    }

    #[test]
    fn harness_files_cover_v2_contract_markers() {
        let all = harness_concat();
        for marker in [
            "build_model",
            "train_stream",
            "tokens_seen_source",
            "prism_telemetry",
            "finish_evaluation",
            "report(",
            "PRISM_TEST_TRAIN_MINUTES",
            "PRISM_TEST_MAX_PARAMS",
            "PRISM_PROBE_EVERY",
            "PRISM_PROBE_TIME_BUDGET_S",
            "unshare",
            "PRISM_DISABLE_NETNS",
            "PRISM_RESULT_FD",
            "PRISM_PHASE=",
            "metrics_version",
            "pod_manifest",
            "harness_files_sha256",
            "parameter cap",
            "probe_curve",
            "PRISM_FLOW",
            "PRISM_EVAL_ASSETS_DIR",
            "PRISM_EVAL_SECRET_SEED",
            "PHASE_TRAIN_DONE",
            "CAP_EXCEEDED",
            "eval_tier",
            "checkpoint.pt",
            "run_battery",
            "metrics.json",
            "_emit_metrics",
            "cantor",
            "prism_width_multiplier",
            "cheatguard",
            "org.g1.bits_per_byte_code",
            "mirrors",
            "natural_mcq",
            "helmet_rag",
            "PRISM_TEST_TRAIN_ROWS",
            "PRISM_TEST_VAL_ROWS",
            "torch_seed",
            // Modular tokenizer contract: the tokenizer is part of the
            // submission, resolved once per phase and injected as
            // ctx["tokenizer"] / ctx["vocab_size"] (additive — a submission
            // that declares none still gets the pinned fallback).
            "build_tokenizer",
            "ctx[\"tokenizer\"]",
            "vocab_size",
            "fit_to_tokens",
            "bits_per_byte",
        ] {
            assert!(all.contains(marker), "harness package missing {marker}");
        }
        assert!(all.contains("1000000000") || all.contains("PRISM_MAX_PARAMS"));
    }

    #[test]
    fn harness_files_sha256_is_deterministic_hex() {
        let a = harness_files_sha256();
        let b = harness_files_sha256();
        assert_eq!(a, b);
        assert_eq!(a.len(), 64);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn baseline_satisfies_contract() {
        check_contract(BASELINE_ARCHITECTURE_PY, BASELINE_TRAINING_PY).expect("baseline");
    }

    #[test]
    fn contract_rejects_missing_train() {
        let err = check_contract(BASELINE_ARCHITECTURE_PY, "x = 1").unwrap_err();
        assert!(matches!(err, ContractError::MissingTrain));
    }

    /// The dual cap must be *dual*: FLOPs has to bind before the wall for a
    /// competent implementation, or the currency reverts to wall-clock and
    /// the kernel lottery is scored again. 4×RTX 5090 = 838 TFLOPS peak bf16.
    #[test]
    fn wall_bound_is_slack_at_target_mfu() {
        const PEAK_FLOPS: f64 = 838e12;
        let wall_needed_h = |mfu: f64| TRAIN_FLOPS_CAP / (PEAK_FLOPS * mfu) / 3600.0;
        // At 20% MFU the FLOPs cap is still reachable inside the wall bound.
        let at_20 = wall_needed_h(0.20);
        assert!(
            at_20 <= TRAIN_HOURS_CAP,
            "at 20% MFU a full-budget run needs {at_20:.2}h > {TRAIN_HOURS_CAP}h wall: \
             the wall would bind and FLOPs would stop being the currency"
        );
        // ...and it is not so slack that the anti-DoS bound is vacuous: a
        // very slow implementation must still be stopped by the clock.
        assert!(
            wall_needed_h(0.10) > TRAIN_HOURS_CAP,
            "wall bound must still bite for a pathologically slow run"
        );
        // The underspend floor has to be reachable well inside the wall.
        let floor_h = wall_needed_h(0.20) * MIN_SPEND_FRACTION;
        assert!(
            floor_h < TRAIN_HOURS_CAP,
            "MIN_SPEND_FRACTION floor ({floor_h:.2}h at 20% MFU) must be \
             reachable inside the wall bound, else honest runs fail the floor"
        );
        assert!((MIN_SPEND_FRACTION - 0.5).abs() < f64::EPSILON);
        assert!(TRAIN_FLOPS_CAP > 0.0 && TRAIN_FLOPS_CAP.is_finite());
    }

    /// The step cap and the FLOPs cap must be **mutually reachable**, and at
    /// the reference batch they are not.
    ///
    /// Measured in Phase 0 (`docs/evidence/prism-v3-phase0/`): the reference
    /// Transformer++ baseline stopped at 20 006 steps — its own
    /// `ctx["max_train_steps"]` budget — having attested `1.82e17` FLOPs, i.e.
    /// **6.1 %** of `TRAIN_FLOPS_CAP`. At `batch 8 x seq 512 = 4096`
    /// tokens/step, 20 000 steps buys `8.2e7` tokens, and the cap needs
    /// `1.35e9`. So a submission at the reference batch cannot reach
    /// `MIN_SPEND_FRACTION` no matter how efficient it is: the underspend gate
    /// would mark the *reference baseline* `Ineligible`.
    ///
    /// This is a real interaction between three constants, not a harness bug,
    /// and it is why the gate must not be switched on before miners are told
    /// the batch implication (~132 at seq 512, 16x the reference). Asserted
    /// here so the arithmetic is checked rather than rediscovered on a pod.
    #[test]
    fn step_cap_and_flops_cap_are_only_mutually_reachable_at_large_batch() {
        // Measured on 1xRTX 5090, d=1024 L=24 V=50257: dispatch-counted
        // FLOPs/token, cross-checked analytically to 1.1 %.
        const F_TOK_MEASURED: f64 = 2.221e9;
        const REF_BATCH: f64 = 8.0;
        const SEQ: f64 = 512.0;

        let tokens_at_step_cap = f64::from(MAX_TRAIN_STEPS) * REF_BATCH * SEQ;
        let spendable = tokens_at_step_cap * F_TOK_MEASURED;
        let fraction = spendable / TRAIN_FLOPS_CAP;
        assert!(
            fraction < MIN_SPEND_FRACTION,
            "the reference batch now reaches {pct:.1} % of the FLOPs cap, \
             which is at/above MIN_SPEND_FRACTION — if a constant changed to \
             make this true, update docs/PRISM_RECIPE.md and the miner docs, \
             because the batch guidance is no longer required",
            pct = fraction * 100.0
        );

        // The batch that DOES make the cap reachable inside the step cap.
        let needed_tokens = TRAIN_FLOPS_CAP / F_TOK_MEASURED;
        let needed_batch = needed_tokens / f64::from(MAX_TRAIN_STEPS) / SEQ;
        assert!(
            needed_batch > REF_BATCH,
            "sanity: reaching the cap must need a LARGER batch than the reference"
        );
        assert!(
            needed_batch < 4096.0,
            "batch {needed_batch:.0} at seq {seq} is not physically trainable \
             on the pod — the caps are irreconcilable, not merely demanding, \
             and TRAIN_FLOPS_CAP or MAX_TRAIN_STEPS must move",
            seq = SEQ
        );
    }

    /// Probe knobs must stay in the range the attestation reasons about.
    #[test]
    fn flops_probe_knobs_are_sane() {
        assert_eq!(FLOPS_PROBE_SAMPLES, 8);
        assert!((FLOPS_PROBE_CV_MAX - 0.15).abs() < f64::EPSILON);
        assert!((FLOPS_ANALYTIC_GAP_MAX - 0.25).abs() < f64::EPSILON);
        const {
            assert!(FLOPS_PROBE_SAMPLES >= 3, "median/CV need >= 3 samples");
            assert!(FLOPS_PROBE_CV_MAX > 0.0 && FLOPS_PROBE_CV_MAX < 1.0);
            assert!(FLOPS_ANALYTIC_GAP_MAX > 0.0 && FLOPS_ANALYTIC_GAP_MAX < 1.0);
        }
        // The harness must actually carry the attestation the caps describe.
        let all = harness_concat();
        for marker in [
            "FlopCounterMode",
            "flops_per_token",
            "BudgetExhausted",
            "analytic",
            "org.diag.flops_attested",
            "org.diag.binding_cap",
            "org.diag.spend_fraction",
        ] {
            assert!(
                all.contains(marker),
                "harness missing FLOPs marker {marker}"
            );
        }
    }

    /// `PRISM_TEST_TRAIN_FLOPS` must shrink the cap the same way
    /// `PRISM_TEST_TRAIN_MINUTES` shrinks the wall, and must ignore garbage.
    #[test]
    fn test_mode_env_overrides_flops_cap() {
        assert!((train_flops_cap() - TRAIN_FLOPS_CAP).abs() < f64::EPSILON);
        std::env::set_var("PRISM_TEST_TRAIN_FLOPS", "1e14");
        assert!((train_flops_cap() - 1e14).abs() < 1.0);
        for bad in ["0", "-1e14", "nan", "not-a-number", ""] {
            std::env::set_var("PRISM_TEST_TRAIN_FLOPS", bad);
            assert!(
                (train_flops_cap() - TRAIN_FLOPS_CAP).abs() < f64::EPSILON,
                "override {bad:?} must be ignored, not trusted"
            );
        }
        std::env::remove_var("PRISM_TEST_TRAIN_FLOPS");
    }

    /// The pod the orchestrator rents must outlast the harness it runs.
    /// Regression guard for the budget over-subscription: the train child's
    /// phase ceilings plus the eval child's ceiling must fit the pod cap.
    #[test]
    fn pod_lifetime_covers_train_plus_eval() {
        // Train child: build -> train (cap + 120 s grace) -> checkpoint.
        let train_child =
            HARNESS_BUILD_TIMEOUT_S + TRAIN_HOURS_CAP * 3600.0 + 120.0 + HARNESS_SCORE_TIMEOUT_S;
        // Eval child announces one phase, so its whole timeout applies.
        let worst_case_s = train_child + HARNESS_EVAL_TIMEOUT_S;
        let pod_s = POD_LIFETIME_HOURS_CAP * 3600.0;
        assert!(
            worst_case_s <= pod_s,
            "harness worst case {worst_case_s}s exceeds pod cap {pod_s}s \
             (train_child={train_child}s, eval={HARNESS_EVAL_TIMEOUT_S}s)"
        );
        // The eval ceiling must in turn contain the battery ceilings the
        // python side declares, with reserve for load/rollup/score.
        let harness = harness_concat();
        assert!(
            harness.contains("BATTERY_BUDGET_S = 3600.0"),
            "eval.common battery budget moved — re-check the pod arithmetic"
        );
        assert!(
            harness.contains("PRISM_EVAL_TIMEOUT_S\", 5400.0"),
            "harness eval timeout moved — re-check the pod arithmetic"
        );
        const {
            assert!(
                HARNESS_EVAL_TIMEOUT_S > EVAL_GLOBAL_BUDGET_S,
                "eval ceiling must leave reserve above the battery budget"
            );
        }
        // The Rust mirror of the battery budget must match the Python that
        // declares it, or the pod arithmetic above is computed on a fiction.
        assert!(
            harness.contains(&format!("BATTERY_BUDGET_S = {EVAL_GLOBAL_BUDGET_S:.1}")),
            "EVAL_GLOBAL_BUDGET_S must equal eval.common.BATTERY_BUDGET_S"
        );
        // Pod cap must be tight, not merely sufficient: an over-wide cap
        // hides a future over-subscription instead of failing this test.
        assert!(
            pod_s - worst_case_s < 3600.0,
            "pod cap {pod_s}s is more than 1h above the worst case \
             {worst_case_s}s — re-derive it rather than padding"
        );
    }

    #[test]
    fn caps_match_user_goal() {
        // 5.0h wall is the anti-DoS bound; TRAIN_FLOPS_CAP is the currency.
        assert!((TRAIN_HOURS_CAP - 5.0).abs() < f64::EPSILON);
        let a = POD_LIFETIME_HOURS_CAP;
        let b = TRAIN_HOURS_CAP;
        assert!(a > b);
        assert_eq!(MAX_PARAMS, 1_000_000_000);
        let all = harness_concat();
        assert!(all.contains("1000000000") || all.contains("PRISM_MAX_PARAMS"));
        assert!(all.contains("parameter cap"));
    }

    #[test]
    fn test_mode_env_overrides_caps_and_restores() {
        // Single test touches these process-global vars; no other test in
        // this binary reads them, so there is no parallel-test race.
        assert!((train_hours_cap() - TRAIN_HOURS_CAP).abs() < f64::EPSILON);
        assert_eq!(max_params(), MAX_PARAMS);
        std::env::set_var("PRISM_TEST_TRAIN_MINUTES", "15");
        std::env::set_var("PRISM_TEST_MAX_PARAMS", "2000000");
        assert!((train_hours_cap() - 0.25).abs() < f64::EPSILON);
        assert_eq!(max_params(), 2_000_000);
        std::env::set_var("PRISM_TEST_TRAIN_MINUTES", "nope");
        std::env::set_var("PRISM_TEST_MAX_PARAMS", "-5");
        assert!((train_hours_cap() - TRAIN_HOURS_CAP).abs() < f64::EPSILON);
        assert_eq!(max_params(), MAX_PARAMS);
        std::env::remove_var("PRISM_TEST_TRAIN_MINUTES");
        std::env::remove_var("PRISM_TEST_MAX_PARAMS");
    }

    #[test]
    fn harness_documents_telemetry_hook_contract() {
        let all = harness_concat();
        assert!(all.contains("prism_telemetry"));
        assert!(all.contains("finish_evaluation"));
        assert!(all.contains("report("));
        assert!(all.contains("PRISM_TEST_TRAIN_MINUTES"));
        assert!(all.contains("PRISM_TEST_MAX_PARAMS"));
    }

    #[test]
    fn descriptor_pin_stable() {
        let a = recipe_pin_hex();
        let b = recipe_pin_hex();
        assert_eq!(a, b);
        assert_eq!(a.len(), 64);
    }
}
