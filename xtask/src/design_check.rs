//! Fail if `docs/DESIGN_CHALLENGE.md` is missing required freeze pins.

use std::fs;
use std::path::Path;

/// Heading markers that must appear in `DESIGN_CHALLENGE.md`.
/// Pin letters match [`docs/DESIGN_CHALLENGE_CHECKLIST.md`].
const SECTION_MARKERS: &[(&str, &str)] = &[
    ("T", "## 1. What runs where (topology)"),
    ("I", "## 2. Identifiers and versions"),
    ("H", "## 3. Miner harness contract"),
    ("S", "## 4. Sandbox hardening"),
    ("Z", "## 5. Sanitize rules"),
    ("V", "## 6. Viewer headers and CSP"),
    ("R", "## 7. Rounds and quotas"),
    ("L", "## 8. Admin winners + agentic anti-cheat"),
    ("X", "## 9. Elimination"),
    ("D", "## 10. Declared participant set and"),
    ("K", "## 11. Key custody (challenge signing key)"),
    ("C", "## 12. Compose services, ports, image contract"),
    ("A", "## 13. HTTP API surface"),
];

/// Content pins (not only headings).
const CONTENT_PINS: &[(&str, &str)] = &[
    ("challenge_id", "design"),
    ("challenge_id_field", "challenge_id"),
    ("scoring_version", "challenge_scoring_version"),
    ("scoring_version_3", "u16 = 3"),
    ("bundle_protocol_version", "protocol_version = 1"),
    ("emission_zero", "emission_share_bps = 0"),
    ("prism_bps_sole", "10000"),
    ("SCORE_MAX", "1_000_000"),
    ("compose_port", "8093"),
    ("round_secs", "8_640"),
    ("round_id_floor", "floor(unix_secs / 8640)"),
    ("rounds_per_day", "10 rounds"),
    ("agent_run_timeout", "AGENT_RUN_TIMEOUT_SECS = 1_800"),
    ("scoring_window", "SCORING_WINDOW_ROUNDS = 10"),
    ("daily_quota", "DAILY_RUN_QUOTA = 10"),
    ("prompts_per_round", "3 prompts"),
    ("bank_v1", "bank_v1.json"),
    ("agent_py", "agent.py"),
    ("pyproject", "pyproject.toml"),
    ("zip_submit", "application/zip"),
    ("env_vars", "env_vars"),
    ("required_pages", "index.html"),
    ("pricing_page", "pricing.html"),
    ("components_page", "components.html"),
    ("readonly_rootfs", "ReadonlyRootfs"),
    ("cap_drop", "CapDrop"),
    ("no_new_privileges", "no-new-privileges:true"),
    ("network_mode", "design-sandbox-egress"),
    ("name_prefix", "base-design-"),
    ("csp_sandbox", "sandbox; default-src 'none'"),
    ("html_never_served", "Produced HTML is never served"),
    ("agentic_review_stage", "AgenticReview"),
    ("admin_winners_route", "/v1/admin/rounds/{id}/winners"),
    ("admin_not_gateway", "not exposed via gateway"),
    ("window_share", "SCORE_MAX × p / total_window_points"),
    ("allowed_inspiration", "Mobbin"),
    ("master_only", "master-only"),
    ("no_prompt_validation", "no human prompt validation"),
    ("elimination_bps", "ELIMINATION_BOTTOM_BPS = 2000"),
    ("cooldown_rounds", "ELIMINATION_COOLDOWN_ROUNDS = 10"),
    ("bottom_20", "bottom 20%"),
    ("D24_silence", "Silence is a bug"),
    ("exact_E", "emit_signed_leaf_set"),
    ("no_phala", "no Phala/CVM"),
    ("BUNDLE_SPEC_link", "BUNDLE_SPEC.md"),
    ("rawweight_domain", "base-rawweight-v1"),
    ("round_domain", "base-design-round-id-v1"),
    ("submission_domain", "base-design-submission-v1"),
    ("pair_domain", "base-design-pair-id-v1"),
    ("sim_sandbox", "SimSandbox"),
    ("ChallengeInternal", "ChallengeInternal"),
    ("NotAttempted", "NotAttempted"),
    ("run_logs_route", "/v1/runs/{id}/logs"),
    ("stats_route", "/v1/stats"),
    ("dashboard_route", "/v1/dashboard"),
    ("miners_route", "/v1/miners/{hotkey}"),
    ("poll_hint", "poll_hint_ms"),
];

/// Run the design-challenge freeze-doc completeness gate.
///
/// # Errors
///
/// Returns a multi-line error when the spec file is missing or any pin fails.
pub fn run(workspace_root: &Path) -> Result<(), String> {
    let spec_path = workspace_root.join("docs/DESIGN_CHALLENGE.md");
    let checklist_path = workspace_root.join("docs/DESIGN_CHALLENGE_CHECKLIST.md");

    if !checklist_path.is_file() {
        return Err(format!(
            "missing checklist file: {}",
            checklist_path.display()
        ));
    }

    let body = fs::read_to_string(&spec_path).map_err(|e| {
        format!(
            "read {}: {e} (DESIGN_CHALLENGE.md is required; design-check freeze gate)",
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

    // Must forbid :latest in the image contract discussion.
    if !body.contains(":latest") {
        failures.push("content pin no_latest_ban: spec must mention :latest as forbidden".into());
    }

    // Emission must stay zero until ceremony.
    if !body.contains("emission_share_bps = 0") {
        failures.push("content pin emission_posture: need explicit design emission 0 bps".into());
    }

    // CSP sandbox without allow-scripts is the viewer guarantee.
    if !body.contains("without `allow-scripts`") && !body.contains("without allow-scripts") {
        failures
            .push("content pin csp_no_scripts: need explicit no allow-scripts CSP wording".into());
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
            "design-check: OK ({} section pins, {} content pins) — {}",
            SECTION_MARKERS.len(),
            CONTENT_PINS.len(),
            spec_path.display()
        );
        Ok(())
    } else {
        Err(format!(
            "design-check failed ({}):\n{}",
            failures.len(),
            failures.join("\n")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{CONTENT_PINS, SECTION_MARKERS};

    #[test]
    fn section_markers_nonempty() {
        assert!(SECTION_MARKERS.len() >= 12);
    }

    #[test]
    fn content_pins_include_identity() {
        assert!(CONTENT_PINS
            .iter()
            .any(|(n, v)| *n == "challenge_id" && *v == "design"));
        assert!(CONTENT_PINS
            .iter()
            .any(|(n, v)| *n == "emission_zero" && *v == "emission_share_bps = 0"));
        assert!(CONTENT_PINS
            .iter()
            .any(|(n, v)| *n == "bank_v1" && *v == "bank_v1.json"));
        assert!(CONTENT_PINS
            .iter()
            .any(|(n, v)| *n == "agentic_review_stage" && *v == "AgenticReview"));
    }
}
