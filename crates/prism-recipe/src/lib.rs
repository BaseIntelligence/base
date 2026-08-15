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
        "prismlib/envutil.py",
        include_str!("../harness/prismlib/envutil.py"),
    ),
    (
        "prismlib/eval_v3.py",
        include_str!("../harness/prismlib/eval_v3.py"),
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

/// Maximum model parameters allowed after `build_model` (350M).
pub const MAX_PARAMS: u64 = 350_000_000;

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

/// Train wall clock cap per submission (seconds). **User goal: up to 6h.**
pub const TRAIN_HOURS_CAP: f64 = 6.0;

/// Pod lifetime cap total (seconds): train cap + bootstrap margin (1h).
pub const POD_LIFETIME_HOURS_CAP: f64 = 7.0;

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
    /// Train wall seconds cap.
    pub train_hours_cap: f64,
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
        assert!(all.contains("350000000") || all.contains("PRISM_MAX_PARAMS"));
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

    #[test]
    fn caps_match_user_goal() {
        assert!((TRAIN_HOURS_CAP - 6.0).abs() < f64::EPSILON);
        let a = POD_LIFETIME_HOURS_CAP;
        let b = TRAIN_HOURS_CAP;
        assert!(a > b);
        assert_eq!(MAX_PARAMS, 350_000_000);
        let all = harness_concat();
        assert!(all.contains("350000000") || all.contains("PRISM_MAX_PARAMS"));
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
