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

// Complete CUDA 13 + Transformer Engine execution image built from
// deploy/prism-pod. The recipe crate owns the advertised immutable pin so the
// API and Lium template cannot drift.
const RECIPES_TEMPLATE_IMAGE: &str = prism_recipe::POD_IMAGE_REF;
const RECIPES_TEMPLATE_TAG: &str = "v10-cuda13-te";
/// Default Lium template name for the pinned recipe-v10 image.
pub const RECIPES_TEMPLATE_NAME: &str = "prism-recipe-v10-digest-5d2508aea5f3-tagged";
/// Template name used automatically when the image is env-overridden
/// (template identity is name-based / reuse-if-exists on Lium: a new image
/// must ship under a new name or pods would keep the old template).
pub const RECIPES_TEMPLATE_NAME_V10: &str = "prism-recipe-v10";
/// Startup commands (empty: sshd comes from image init).
pub const RECIPES_TEMPLATE_STARTUP: &str = "";

/// Resolved pod `(image, tag, default_template_name)`.
///
/// `PRISM_POD_IMAGE_REF` lets ops stage a replacement without a code bump. The
/// value must be a complete `repository@sha256:<64 hex>` reference; floating
/// and mutable tags fail closed. Unset uses [`prism_recipe::POD_IMAGE_REF`].
///
/// # Errors
///
/// Returns [`prism_lium_types::LiumError::Integrity`] for a non-digest override.
pub fn resolved_pod_image() -> Result<(String, Option<String>, String), prism_lium_types::LiumError>
{
    let env = |k: &str| std::env::var(k).ok().filter(|s| !s.trim().is_empty());
    let credential = env("PRISM_POD_DOCKER_CREDENTIAL_ID");
    match env("PRISM_POD_IMAGE_REF") {
        Some(image) if is_digest_image_ref(&image) => {
            let template_name = credential_scoped_template_name(
                &template_name_for_image(&image),
                credential.as_deref(),
            );
            let tag = env("PRISM_POD_IMAGE_TAG").unwrap_or_else(|| RECIPES_TEMPLATE_TAG.into());
            Ok((image, Some(tag), template_name))
        }
        Some(_) => Err(prism_lium_types::LiumError::Integrity(
            "PRISM_POD_IMAGE_REF must be repository@sha256:<64 lowercase hex>".into(),
        )),
        None => Ok((
            RECIPES_TEMPLATE_IMAGE.to_owned(),
            Some(RECIPES_TEMPLATE_TAG.to_owned()),
            credential_scoped_template_name(RECIPES_TEMPLATE_NAME, credential.as_deref()),
        )),
    }
}

fn credential_scoped_template_name(base: &str, credential: Option<&str>) -> String {
    credential.map_or_else(
        || base.to_owned(),
        |id| format!("{base}-cred-{}", id.get(..8).unwrap_or(id)),
    )
}

/// Build the Lium template payload without conflating a digest with an image
/// name. Private provider images fail closed unless Lium owns a credential.
pub fn lium_template_create_body(
    name: &str,
    docker_image: &str,
    docker_image_tag: Option<&str>,
    startup_commands: Option<&str>,
    docker_credential_id: Option<&str>,
) -> Result<serde_json::Value, LiumError> {
    let (repository, digest) = docker_image
        .rsplit_once('@')
        .map_or((docker_image, None), |(repository, digest)| {
            (repository, Some(digest))
        });
    if repository.starts_with("registry.digitalocean.com/basecrawl/")
        && docker_credential_id.is_none()
    {
        return Err(LiumError::Integrity(
            "PRISM_POD_DOCKER_CREDENTIAL_ID is required to create the private Prism template"
                .into(),
        ));
    }
    let mut body = serde_json::json!({
        "name": name,
        "docker_image": repository,
        "internal_ports": [22],
        "is_private": true,
        "container_start_immediately": true,
    });
    for (key, value) in [
        ("docker_image_digest", digest),
        ("docker_image_tag", docker_image_tag),
        ("startup_commands", startup_commands),
        ("docker_credential_id", docker_credential_id),
    ] {
        if let Some(value) = value {
            body[key] = serde_json::Value::String(value.to_owned());
        }
    }
    Ok(body)
}

fn template_name_for_image(image: &str) -> String {
    let digest = image
        .rsplit_once("@sha256:")
        .map_or("", |(_, digest)| digest);
    let short = digest.get(..12).unwrap_or(digest);
    format!("{RECIPES_TEMPLATE_NAME_V10}-digest-{short}-tagged")
}

fn is_digest_image_ref(image: &str) -> bool {
    let Some((repository, digest)) = image.rsplit_once("@sha256:") else {
        return false;
    };
    !repository.is_empty()
        && digest.len() == 64
        && digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
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
        // The budget CURRENCY. Sent explicitly, like the wall cap above, so
        // the pod enforces the master's constant rather than the default
        // compiled into whatever harness copy it happens to run.
        (
            "PRISM_TRAIN_FLOPS_CAP",
            prism_recipe::TRAIN_FLOPS_CAP.to_string(),
        ),
        (
            "PRISM_MIN_SPEND_FRACTION",
            prism_recipe::MIN_SPEND_FRACTION.to_string(),
        ),
        ("PRISM_GPU_TYPE", gpu_type.replace('\'', "")),
        (
            "PRISM_HARNESS_FILES_SHA256",
            prism_recipe::harness_files_sha256(),
        ),
    ];
    for key in [
        "PRISM_TEST_TRAIN_MINUTES",
        // Miner-visible train geometry. Default harness stream is 8×512,
        // which DataParallel then shards (2 samples/GPU on 4×5090). Passing
        // these through lets an isolated 4-GPU proof raise per-rank batch
        // and seq without a harness rebuild on the pod.
        "PRISM_SEQ_LEN",
        "PRISM_TRAIN_BATCH_SIZE",
        // Operator seed-variance sweep. This list is an ALLOWLIST, so a knob
        // absent from it never reaches the pod: without this entry the
        // Phase-0 sweep would set the seed on the control-plane container,
        // see it ignored over SSH, and silently train every "different seed"
        // run on the same lattice seed — producing sigma_seed ≈ 0 and a
        // confident wrong answer. Never set for a scored round: a
        // miner-chosen seed would make two submissions incomparable.
        "PRISM_SEED_OVERRIDE",
        // Reduced-budget FLOPs cap for operator measurement waves.
        "PRISM_TEST_TRAIN_FLOPS",
        // FLOPs-probe knobs, so a measurement wave can widen the probe or
        // the analytic tolerance without rebuilding the harness.
        "PRISM_FLOPS_PROBE_SAMPLES",
        "PRISM_FLOPS_PROBE_CV_MAX",
        "PRISM_FLOPS_ANALYTIC_GAP_MAX",
        "PRISM_PROBE_EVERY",
        "PRISM_PROBE_TIME_BUDGET_S",
        "PRISM_G6_BPB_THRESHOLD",
        "PRISM_BUILD_TIMEOUT_S",
        "PRISM_SCORE_TIMEOUT_S",
        "PRISM_EVAL_TIMEOUT_S",
        "PRISM_INSTALL_TIMEOUT_SECS",
        "PRISM_TEST_MAX_PARAMS",
        "PRISM_TEST_EVAL_CAPS",
        "PRISM_EVAL_BATTERY_BUDGET_S",
        "PRISM_EVAL_N_ITEMS",
        "PRISM_EVAL_G5_N_ITEMS",
        "PRISM_EVAL_G1_CAP",
        "PRISM_EVAL_G2_CAP",
        "PRISM_EVAL_G2_CAP_USABLE",
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
    if let Ok(raw) = std::env::var("PRISM_EVAL_G2_TASKS") {
        let allowed = [
            "lambada",
            "hellaswag",
            "piqa",
            "arc_easy",
            "arc_challenge",
            "winogrande",
            "boolq",
            "openbookqa",
        ];
        let picked: Vec<&str> = raw
            .split(',')
            .map(str::trim)
            .filter(|task| allowed.contains(task))
            .collect();
        if !picked.is_empty() {
            values.push(("PRISM_EVAL_G2_TASKS", picked.join(",")));
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

#[cfg(test)]
mod tests {
    use super::{credential_scoped_template_name, is_digest_image_ref, template_name_for_image};

    #[test]
    fn pod_image_override_requires_digest() {
        assert!(is_digest_image_ref(
            "ghcr.io/baseintelligence/prism-pod@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        ));
        assert!(!is_digest_image_ref(
            "ghcr.io/baseintelligence/prism-pod:v10-cuda13-te"
        ));
        assert!(!is_digest_image_ref(
            "ghcr.io/baseintelligence/prism-pod@sha256:ABCDEF"
        ));
    }

    #[test]
    fn pod_image_template_name_is_digest_scoped() {
        assert_eq!(
            template_name_for_image(
                "registry.digitalocean.com/basecrawl/prism-pod@sha256:\
                 5d2508aea5f3eca9e57f1d27d11354249c7bde315feb35c1308f4e2175dfa3aa"
            ),
            "prism-recipe-v10-digest-5d2508aea5f3-tagged"
        );
    }

    #[test]
    fn pod_image_template_name_is_credential_scoped() {
        assert_eq!(
            credential_scoped_template_name(
                "prism-recipe-v10-digest-abc",
                Some("4ca23da3-8f5c-4b41-b742-636d2d8c6be7")
            ),
            "prism-recipe-v10-digest-abc-cred-4ca23da3"
        );
    }
}
