//! Integration: real Harbor pack `realpr-click-3442` round-trip + strip.
//!
//! Skips when the fixture pack is not present locally (CI default).
//! Set `GBASE_REAL_PACK_DIR` or place the pack on one of the candidate paths.

use std::path::{Path, PathBuf};

use agent_pack::{load_pack, PackError, STRIPPED_FIELD_NAMES};

fn real_pack_candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(p) = std::env::var("GBASE_REAL_PACK_DIR") {
        out.push(PathBuf::from(p));
    }
    out.extend(
        [
            "/tmp/deepagent/datasets/prod_hard_deepswe_med/tasks/realpr-click-3442",
            "/tmp/baseintel/deepagent/datasets/prod_hard_deepswe_med/tasks/realpr-click-3442",
            "/tmp/da_m29d_hf_pull_verify/tasks/realpr-click-3442",
        ]
        .into_iter()
        .map(PathBuf::from),
    );
    out
}

fn resolve_real_pack() -> Option<PathBuf> {
    real_pack_candidates()
        .into_iter()
        .find(|p| p.join("task.toml").is_file())
}

#[test]
fn realpr_click_3442_round_trip_strip_and_stable_digest() {
    let Some(root) = resolve_real_pack() else {
        eprintln!(
            "skip: realpr-click-3442 not found; tried {:?}; set GBASE_REAL_PACK_DIR to run",
            real_pack_candidates()
        );
        return;
    };
    let pack = load_pack(&root).unwrap_or_else(|e| panic!("load {}: {e}", root.display()));

    assert_eq!(pack.task_id, "realpr-click-3442");
    assert_eq!(
        pack.base_commit_hash,
        "d5fbd32842da361cc9be8658d94a64e9cc417fb5"
    );
    assert_eq!(pack.repository_url, "https://github.com/pallets/click.git");
    assert!(pack.instruction.contains("Merge Main into Stable"));
    assert_eq!(pack.agent_timeout_sec, 5400);
    assert!(
        pack.held_out.solution_patch.is_some(),
        "real pack must load solution for operator side"
    );
    assert!(
        pack.held_out.test_patch.is_some(),
        "real pack must load held-out test.patch"
    );

    let d1 = pack.pack_digest_hex();
    let pack2 = load_pack(&root).expect("second load");
    let d2 = pack2.pack_digest_hex();
    assert_eq!(d1, d2, "pack_digest must be stable across loads");
    assert_eq!(d1.len(), 64);

    let stripped = pack.strip();
    stripped.assert_total_keys().expect("total");
    assert_eq!(stripped.task_id, "realpr-click-3442");
    assert_eq!(stripped.deadline_sec, 5400);
    assert!(stripped.environment_image_digest.starts_with("sha256:"));
    assert_eq!(
        stripped.environment_image_digest.len(),
        "sha256:".len() + 64
    );

    let json = serde_json::to_value(&stripped).expect("json");
    let obj = json.as_object().expect("object");
    assert_eq!(obj.len(), STRIPPED_FIELD_NAMES.len());
    for k in STRIPPED_FIELD_NAMES {
        assert!(obj.contains_key(*k), "missing {k}");
    }
    for forbidden in [
        "solution",
        "solution_patch",
        "test_patch",
        "tests",
        "grader",
    ] {
        assert!(!obj.contains_key(forbidden), "leaked {forbidden}");
    }

    // Solution patch body must not appear in stripped serialization.
    if let Some(sol) = &pack.held_out.solution_patch {
        let sol_str = String::from_utf8_lossy(sol);
        let needle = sol_str
            .lines()
            .find(|l| l.len() > 40)
            .unwrap_or("solution.patch");
        let ser = serde_json::to_string(&stripped).expect("ser");
        assert!(
            !ser.contains(needle) || needle.len() < 8,
            "stripped must not embed solution patch body"
        );
    }
}

#[test]
fn real_pack_path_is_directory() {
    let Some(root) = resolve_real_pack() else {
        eprintln!(
            "skip: realpr-click-3442 not found; tried {:?}; set GBASE_REAL_PACK_DIR to run",
            real_pack_candidates()
        );
        return;
    };
    assert!(Path::new(&root).is_dir());
}

#[test]
fn missing_field_error_display_names_field() {
    let err = PackError::MissingField {
        field: "base_commit_hash",
    };
    assert_eq!(err.to_string(), "missing required field `base_commit_hash`");
}
