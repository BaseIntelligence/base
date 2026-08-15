//! Pod harness packaging and allowlisted environment helpers for Lium runs.

#![forbid(unsafe_code)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::redundant_closure_for_method_calls)]

mod detached;

use std::path::PathBuf;

use prism_lium_types::LiumError;

pub use detached::{
    classify_log, detach_launch_cmd, parse_harness_probe, parse_metrics_output, HarnessProbe,
    HarnessProgress, HARNESS_ABSENT, HARNESS_HARVEST_CMD, HARNESS_PROBE_CMD, TRAIN_DONE_MARKER,
};

/// Pod-side staging directory for eval assets.
pub const EVAL_ASSETS_POD_DIR: &str = "/tmp/prism_eval/eval-assets";

// Pod image for the harness. Default is the cu13.0.2 DinD base (sshd from
// image init, empty startup — other marketplace images lack a stable sshd
// under Lium's metachar-free startup rules).
//
// recipe-v10 target: the "complete" base built from deploy/prism-pod
// (ghcr.io/baseintelligence/prism-pod:v10-cuda13-te) ships Transformer
// Engine + build toolchain so miners can NVFP4-train and `pip install`
// extras from their own manifest (see prism-recipe/harness/prismlib/deps.py
// and prism_recipe::POD_IMAGE_REF). Opt-in via env until built+pushed and
// validated on a GPU node — live keeps the daturaai default.
const RECIPES_TEMPLATE_IMAGE: &str = "daturaai/pytorch";
const RECIPES_TEMPLATE_TAG: &str = "2.12.0-py3.12-cuda13.0.2-devel-ubuntu24.04-dind";
/// Default Lium template name (daturaai cu13 base).
pub const RECIPES_TEMPLATE_NAME: &str = "prism-recipe-v9";
/// Template name used automatically when the image is env-overridden
/// (template identity is name-based / reuse-if-exists on Lium: a new image
/// must ship under a new name or pods would keep the old template).
pub const RECIPES_TEMPLATE_NAME_V10: &str = "prism-recipe-v10";
/// Startup commands (empty: sshd comes from image init).
pub const RECIPES_TEMPLATE_STARTUP: &str = "";

/// Resolved pod `(image, tag, default_template_name)`. `PRISM_POD_IMAGE` /
/// `PRISM_POD_IMAGE_TAG` let ops stage the recipe-v10 TE image without a
/// code bump; unset falls back to the daturaai cu13 default (whose pinned
/// tag applies only to that image). An overridden image automatically flips
/// the template name to [`RECIPES_TEMPLATE_NAME_V10`].
#[must_use]
pub fn resolved_pod_image() -> (String, Option<String>, &'static str) {
    let env = |k: &str| std::env::var(k).ok().filter(|s| !s.trim().is_empty());
    match env("PRISM_POD_IMAGE") {
        Some(image) => (image, env("PRISM_POD_IMAGE_TAG"), RECIPES_TEMPLATE_NAME_V10),
        None => (
            RECIPES_TEMPLATE_IMAGE.to_owned(),
            Some(RECIPES_TEMPLATE_TAG.to_owned()),
            RECIPES_TEMPLATE_NAME,
        ),
    }
}

/// Pod bootstrap prepended to the run command. Payloads are staged separately.
pub const HARNESS_BOOTSTRAP: &str = "set -e\ncommand -v pip >/dev/null 2>&1 || apt-get update -q; command -v pip >/dev/null 2>&1 || DEBIAN_FRONTEND=noninteractive apt-get install -y -q python3-pip; python3 -c 'import torch' 2>/dev/null || echo 'torch stopping'; python3 -c 'import transformers' 2>/dev/null || pip install --break-system-packages --root-user-action=ignore 'transformers==4.44.2' 'datasets==3.0.2' 'pyarrow==17.0.0'; mkdir -p /tmp/prism_eval\n";

/// Fixed remote command for extracting the stdin-streamed harness archive.
pub const HARNESS_EXTRACT_CMD: &str = "set -e; mkdir -p /tmp/prism_eval; tar -x -C /tmp/prism_eval";

/// Build the deterministic harness + miner source archive streamed to a pod.
///
/// # Errors
/// Invalid or unexpandable source tree, or archive packing failure.
pub fn harness_upload_tar(
    architecture_py: &str,
    training_py: &str,
    tree_blob: Option<&[u8]>,
) -> Result<Vec<u8>, LiumError> {
    let harness: Vec<(&str, &[u8])> = prism_recipe::HARNESS_FILES
        .iter()
        .map(|(path, contents)| (*path, contents.as_bytes()))
        .collect();
    let tree = tree_blob
        .map(prism_automodel::expand_tree_blob_for_pod)
        .transpose()
        .map_err(|error| LiumError::Exec(format!("tree blob: {error}")))?;
    prism_tree::pack_harness_upload(
        &harness,
        architecture_py.as_bytes(),
        training_py.as_bytes(),
        tree.as_ref(),
    )
    .map_err(|error| LiumError::Exec(error.to_string()))
}

/// Build the shell-safe, allowlisted environment for the remote harness.
#[must_use]
pub fn harness_env_pairs(
    train_hours_cap: f64,
    gpu_type: &str,
    assets_pending: bool,
) -> Vec<(&'static str, String)> {
    let mut values = vec![
        ("PRISM_DATASET_URL", prism_recipe::DATASET_URL.to_owned()),
        ("PRISM_DATASET_SHA256", prism_recipe::dataset_sha256()),
        (
            "PRISM_MAX_TRAIN_STEPS",
            prism_recipe::MAX_TRAIN_STEPS.to_string(),
        ),
        ("PRISM_TRAIN_HOURS_CAP", train_hours_cap.to_string()),
        ("PRISM_GPU_TYPE", gpu_type.replace('\'', "")),
        (
            "PRISM_HARNESS_FILES_SHA256",
            prism_recipe::harness_files_sha256(),
        ),
    ];
    for key in [
        "PRISM_TEST_TRAIN_MINUTES",
        "PRISM_TEST_MAX_PARAMS",
        "PRISM_TEST_EVAL_CAPS",
        "PRISM_EVAL_N_ITEMS",
        "PRISM_EVAL_G5_N_ITEMS",
        "PRISM_EVAL_G1_CAP",
        "PRISM_EVAL_G2_CAP",
        "PRISM_EVAL_NATURAL_ITEMS",
        "PRISM_EVAL_MIRROR_G2_CAP",
        "PRISM_EVAL_G1_BUDGET_S",
        "PRISM_EVAL_G2_BUDGET_S",
        "PRISM_EVAL_G3_BUDGET_S",
        "PRISM_EVAL_G4_BUDGET_S",
        "PRISM_EVAL_G5_BUDGET_S",
        "PRISM_EVAL_G7_BUDGET_S",
        "PRISM_EVAL_G8_SWEEP_S",
        "PRISM_EVAL_G8_SWEEP",
        "PRISM_EVAL_MIRROR_BUDGET_S",
        "PRISM_EVAL_G5_NATURAL_BUDGET_S",
        "PRISM_EVAL_G5_RULER_PROBE_BUDGET_S",
        "PRISM_EVAL_G5_BABILONG_PROBE_BUDGET_S",
    ] {
        if let Ok(value) = std::env::var(key) {
            if value.trim().parse::<f64>().is_ok() {
                values.push((key, value.trim().to_owned()));
            }
        }
    }
    if let Ok(flow) = std::env::var("PRISM_FLOW") {
        let flow = flow.trim().to_ascii_lowercase();
        if flow == "v1" || flow == "v3" {
            values.push(("PRISM_FLOW", flow));
        }
    }
    if assets_pending {
        values.push(("PRISM_EVAL_ASSETS_DIR", EVAL_ASSETS_POD_DIR.to_owned()));
    }
    values
}

/// Resolve an existing master-side eval assets directory.
#[must_use]
pub fn eval_assets_dir() -> Option<PathBuf> {
    let value = std::env::var("PRISM_EVAL_ASSETS_DIR").ok()?;
    let path = PathBuf::from(value.trim());
    path.is_dir().then_some(path)
}

/// Read a fresh 128-bit seed from the operating system entropy source.
///
/// # Errors
/// `/dev/urandom` could not be opened or read.
pub fn random_seed_hex() -> Result<String, LiumError> {
    use std::io::Read as _;
    let mut bytes = [0u8; 16];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut file| file.read_exact(&mut bytes))
        .map_err(|error| LiumError::Exec(format!("entropy: {error}")))?;
    Ok(hex::encode(bytes))
}
