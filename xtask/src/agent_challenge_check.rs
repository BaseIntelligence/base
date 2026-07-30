//! Fail if `docs/AGENT_CHALLENGE.md` is missing required plan item 9 / task 16 pins.

use std::fs;
use std::path::Path;

/// Heading markers that must appear in `AGENT_CHALLENGE.md`.
const SECTION_MARKERS: &[(&str, &str)] = &[
    ("T", "## 1. What runs where (topology)"),
    ("A", "## 3. Attestation precondition (explicit)"),
    ("P", "## 4. Challenge ↔ miner CVM protocol"),
    ("S", "## 5. Score meaning and scoring rule"),
    ("K", "## 6. Key custody (challenge signing key)"),
    ("D", "## 7. Declared participant set and"),
    ("C", "## 9. Compose services, ports, image contract"),
];

/// Content pins (not only headings). scoring_version 2 / task 16 delta.
const CONTENT_PINS: &[(&str, &str)] = &[
    ("bundle_protocol_version", "protocol_version = 1"),
    ("challenge_id", "agent-v1"),
    ("scoring_version", "challenge_scoring_version"),
    ("scoring_version_2", "u16 = 2"),
    ("SCORE_MAX", "1_000_000"),
    ("task_id_v2_domain", "gbase-agent-task-id-v2"),
    ("task_blob_v2_domain", "gbase-agent-task-blob-v2"),
    ("answer_v2_domain", "gbase-agent-answer-v2"),
    ("pack_select_domain", "gbase-agent-pack-select-v1"),
    ("work_receipt_domain", "gbase-agent-work-receipt-v1"),
    ("model_patch", "model.patch"),
    ("ChallengeInternal", "ChallengeInternal"),
    ("HarborVerifier", "HarborVerifier"),
    (
        "fixture_task_id_v2",
        "b1c18e56abe993e20e8dadcb72c7a7cadee8975e5741d15d1acb37f5ea367644",
    ),
    (
        "fixture_answer_v2",
        "703b806158d655e5d37a5b45e3cbdf1e04735517805377199d108ae2a45ead5d",
    ),
    ("attestation_precondition", "precondition for emitting"),
    ("NoScore_attestation", "AttestationNotVerified"),
    ("D24_silence", "Silence is a bug"),
    ("compose_agent_port", "8080"),
    ("compose_challenge_port", "8090"),
    ("image_agent", "ghcr.io/baseintelligence/gbase-agent"),
    ("BUNDLE_SPEC_link", "BUNDLE_SPEC.md"),
    ("D10_report_data", "gbase-attest-v1"),
    ("rawweight_domain", "gbase-rawweight-v1"),
    ("socket_proxy", "socket-proxy"),
    ("PARTICIPATION_FLOOR", "PARTICIPATION_FLOOR"),
    ("no_score_in_cvm", "NO challenge signing key"),
    ("park_no_credit", "Park grants"),
];

/// Run the agent-challenge spec completeness gate.
///
/// # Errors
///
/// Returns a multi-line error when the spec file is missing or any pin fails.
pub fn run(workspace_root: &Path) -> Result<(), String> {
    let spec_path = workspace_root.join("docs/AGENT_CHALLENGE.md");
    let checklist_path = workspace_root.join("docs/AGENT_CHALLENGE_CHECKLIST.md");

    if !checklist_path.is_file() {
        return Err(format!(
            "missing checklist file: {}",
            checklist_path.display()
        ));
    }

    let body = fs::read_to_string(&spec_path).map_err(|e| {
        format!(
            "read {}: {e} (AGENT_CHALLENGE.md is required; plan task 9 wave gate)",
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

    // Explicit attestation precondition (either polarity must be stated; we require YES path).
    let lower = body.to_ascii_lowercase();
    if !lower.contains("precondition for emitting") && !lower.contains("is a precondition for") {
        failures
            .push("content pin attestation_explicit: need explicit precondition wording".into());
    }

    // Must forbid :latest in the image contract discussion (the words appear as a ban).
    if !body.contains(":latest") {
        failures.push("content pin no_latest_ban: spec must mention :latest as forbidden".into());
    }

    // Live scoring must not reintroduce SOFT_MS/HARD_MS as active constants (historical OK).
    // Require v2 correctness language instead.
    if !body.contains("correctness-only") && !body.contains("pure correctness") {
        failures.push(
            "content pin v2_correctness: spec must state pure/correctness-only scoring".into(),
        );
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

    if failures.is_empty() {
        println!(
            "agent-challenge-check: OK ({} section pins, {} content pins) — {}",
            SECTION_MARKERS.len(),
            CONTENT_PINS.len(),
            spec_path.display()
        );
        Ok(())
    } else {
        Err(format!(
            "agent-challenge-check failed ({}):\n{}",
            failures.len(),
            failures.join("\n")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{CONTENT_PINS, SECTION_MARKERS};

    #[test]
    fn seven_section_markers() {
        assert_eq!(SECTION_MARKERS.len(), 7);
        let pins: String = SECTION_MARKERS.iter().map(|(p, _)| *p).collect();
        assert_eq!(pins, "TAPSKDC");
    }

    #[test]
    fn content_pins_nonempty() {
        assert!(CONTENT_PINS.len() >= 10);
    }

    #[test]
    fn v2_pins_present_soft_hard_absent() {
        let names: Vec<&str> = CONTENT_PINS.iter().map(|(n, _)| *n).collect();
        assert!(names.contains(&"scoring_version_2"));
        assert!(names.contains(&"task_id_v2_domain"));
        assert!(names.contains(&"ChallengeInternal"));
        assert!(names.contains(&"model_patch"));
        assert!(!names.contains(&"SOFT_MS"));
        assert!(!names.contains(&"HARD_MS"));
    }
}
