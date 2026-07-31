//! Fail if `docs/HYPERTRAINING.md` is missing required plan item 3 / task 17 pins.

use std::fs;
use std::path::Path;

/// Heading markers that must appear in `HYPERTRAINING.md`.
/// Pin letters match [`docs/HYPERTRAINING_CHECKLIST.md`](../../docs/HYPERTRAINING_CHECKLIST.md).
const SECTION_MARKERS: &[(&str, &str)] = &[
    ("T", "## 1. What runs where (topology)"),
    ("I", "## 2. Identifiers and versions"),
    ("A", "## 3. Attestation precondition"),
    ("P", "## 4. Miner submit protocol"),
    ("S", "## 5. Sealed surface summary"),
    ("G", "## 6. Three guards summary"),
    ("R", "## 7. Score meaning and scoring rule"),
    ("K", "## 8. Key custody (challenge signing key)"),
    ("D", "## 9. Declared participant set and"),
    ("L", "## 10. Leaf emission and gateway POST"),
    ("C", "## 11. Compose services, ports, image contract"),
    ("B", "## 13. ClusterBackend contract"),
];

/// Content pins (not only headings). `scoring_version` 1 / task 17 freeze.
const CONTENT_PINS: &[(&str, &str)] = &[
    ("challenge_id", "hypertraining"),
    ("challenge_id_field", "challenge_id"),
    ("scoring_version", "challenge_scoring_version"),
    ("scoring_version_1", "u16 = 1"),
    ("bundle_protocol_version", "protocol_version = 1"),
    ("emission_zero", "emission_share_bps = 0"),
    ("agent_v1_bps", "10000"),
    ("SCORE_MAX", "1_000_000"),
    ("rawweight_domain", "base-rawweight-v1"),
    ("BUNDLE_SPEC_link", "BUNDLE_SPEC.md"),
    ("design_source", "challenge-training-fork.md"),
    ("sim_backend", "SimBackend"),
    ("real_backend_deferred", "RealBackend"),
    ("not_live_b300", "Not live"),
    ("kernel_kappa", "κ = 2"),
    ("guard_1", "Guard 1"),
    ("guard_2", "Guard 2"),
    ("guard_3", "Guard 3"),
    ("screen_k", "K=3"),
    ("promotion_k", "K=5"),
    ("binary_reject", "> 0.85"),
    ("D24_silence", "Silence is a bug"),
    ("no_llm_gate", "LLM is never a gate"),
    (
        "no_cuda_sandbox_security",
        "CUDA / container sandbox as security boundary",
    ),
    ("no_harbor", "Harbor"),
    ("no_aws_required", "AWS"),
    ("compose_port", "8091"),
    ("te_pin", "2.18.0+e7c550c5"),
    ("mlm_commit", "cd4afffa648426a959dc7cb1e24b5ce7d0c3ff54"),
    ("allowlist", "megatron/core/fusions/**"),
    ("denylist", "megatron/core/datasets/**"),
    ("marginal_delta", "Δ(candidate)"),
    ("branding_base", "product name is **base**"),
    ("task_id_domain", "base-hypertraining-task-id-v1"),
    ("task_blob_domain", "base-hypertraining-task-blob-v1"),
    ("answer_domain", "base-hypertraining-answer-v1"),
    ("receipt_domain", "base-hypertraining-receipt-v1"),
    ("NoScore_attestation", "AttestationNotVerified"),
    ("ChallengeInternal", "ChallengeInternal"),
    ("agent_challenge_port", "8090"),
];

/// Run the hypertraining freeze-doc completeness gate.
///
/// # Errors
///
/// Returns a multi-line error when the spec file is missing or any pin fails.
pub fn run(workspace_root: &Path) -> Result<(), String> {
    let spec_path = workspace_root.join("docs/HYPERTRAINING.md");
    let checklist_path = workspace_root.join("docs/HYPERTRAINING_CHECKLIST.md");

    if !checklist_path.is_file() {
        return Err(format!(
            "missing checklist file: {}",
            checklist_path.display()
        ));
    }

    let body = fs::read_to_string(&spec_path).map_err(|e| {
        format!(
            "read {}: {e} (HYPERTRAINING.md is required; plan task 17 freeze gate)",
            spec_path.display()
        )
    })?;

    let mut failures = Vec::new();

    for (pin, marker) in SECTION_MARKERS {
        if !body.contains(marker) {
            failures.push(format!(
                "section ({pin}): missing heading marker:\n  {marker}"
            ));
        }
    }

    for (name, needle) in CONTENT_PINS {
        if !body.contains(needle) {
            failures.push(format!("content pin {name}: missing substring {needle:?}"));
        }
    }

    // Explicit attestation precondition wording (YES path).
    let lower = body.to_ascii_lowercase();
    if !lower.contains("precondition for emitting") && !lower.contains("is a precondition for") {
        failures
            .push("content pin attestation_explicit: need explicit precondition wording".into());
    }

    // Must forbid :latest in the image contract discussion.
    if !body.contains(":latest") {
        failures.push("content pin no_latest_ban: spec must mention :latest as forbidden".into());
    }

    // Emission must stay zero until ceremony — require explicit zero posture language.
    if !body.contains("emission_share_bps = 0")
        && !body.contains("emission_share_bps` for hypertraining is **0**")
    {
        failures.push(
            "content pin emission_posture: need explicit hypertraining emission 0 bps".into(),
        );
    }

    // Live path is SimBackend; Real B300 deferred.
    if !body.contains("SimBackend") {
        failures.push("content pin sim_path: need SimBackend as current path".into());
    }

    let checklist = fs::read_to_string(&checklist_path)
        .map_err(|e| format!("read {}: {e}", checklist_path.display()))?;
    for (pin, _) in SECTION_MARKERS {
        let token = format!("({pin})");
        if !checklist.contains(&token) {
            failures.push(format!(
                "checklist missing pin token {token} in {}",
                checklist_path.display()
            ));
        }
    }
    // Emission pin (E) is content-only in the checklist table.
    if !checklist.contains("(E)") {
        failures.push(format!(
            "checklist missing pin token (E) in {}",
            checklist_path.display()
        ));
    }

    if failures.is_empty() {
        println!(
            "hypertraining-check: OK ({} section pins, {} content pins) — {}",
            SECTION_MARKERS.len(),
            CONTENT_PINS.len(),
            spec_path.display()
        );
        Ok(())
    } else {
        Err(format!(
            "hypertraining-check failed ({}):\n{}",
            failures.len(),
            failures.join("\n")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{CONTENT_PINS, SECTION_MARKERS};

    #[test]
    fn twelve_section_markers() {
        assert_eq!(SECTION_MARKERS.len(), 12);
        let pins: String = SECTION_MARKERS.iter().map(|(p, _)| *p).collect();
        assert_eq!(pins, "TIAPSGRKDLCB");
    }

    #[test]
    fn content_pins_nonempty() {
        assert!(CONTENT_PINS.len() >= 20);
    }

    #[test]
    fn v1_pins_present_agent_domains_absent() {
        let names: Vec<&str> = CONTENT_PINS.iter().map(|(n, _)| *n).collect();
        assert!(names.contains(&"scoring_version_1"));
        assert!(names.contains(&"challenge_id"));
        assert!(names.contains(&"task_id_domain"));
        assert!(names.contains(&"compose_port"));
        assert!(!names.contains(&"scoring_version_2"));
        assert!(!names.contains(&"SOFT_MS"));
        assert!(!names.contains(&"HARD_MS"));

        let needles: Vec<&str> = CONTENT_PINS.iter().map(|(_, n)| *n).collect();
        assert!(needles.contains(&"hypertraining"));
        assert!(needles.contains(&"base-hypertraining-task-id-v1"));
        assert!(!needles.iter().any(|n| n.contains("base-agent-task-id")));
    }

    #[test]
    fn challenge_id_pin_is_hypertraining_not_agent() {
        let id = CONTENT_PINS
            .iter()
            .find(|(n, _)| *n == "challenge_id")
            .map(|(_, v)| *v);
        assert_eq!(id, Some("hypertraining"));
    }
}
