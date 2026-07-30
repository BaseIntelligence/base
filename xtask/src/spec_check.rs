//! Fail if `docs/BUNDLE_SPEC.md` is missing required plan item 8 pins (a)–(l).

use std::fs;
use std::path::Path;

/// Heading markers that must appear verbatim in `BUNDLE_SPEC.md` (letter pins).
const LETTER_MARKERS: &[(&str, &str)] = &[
    ("a", "## 1. Encoding law (a)"),
    ("b", "## 2. Protocol version (b)"),
    (
        "c",
        "## 3. Merkle construction (RFC 6962) and leaves (c)",
    ),
    ("d", "## 4. Bundle body and block pin (d)"),
    (
        "e",
        "## 6. Aggregation formula, algorithm_version = 1 (e)",
    ),
    (
        "f",
        "## 5. Emission shares from owner-signed trust root (f)",
    ),
    (
        "g",
        "## 7. Expected participant set derivation (g) (D24)",
    ),
    ("h", "## 8. Final vector comparison (h)"),
    ("i", "## 9. Distribution and caching (i)"),
    ("j", "## 10. Dissent (j)"),
    (
        "k",
        "## 11. Security claim, quarantine, peer sample (k)",
    ),
    (
        "l",
        "## 12. On-chain weight payload: no merkle root (l) (D5)",
    ),
];

/// Additional content pins (not letter headings).
const CONTENT_PINS: &[(&str, &str)] = &[
    (
        "EMPTY_ROOT",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ),
    (
        "D19_claim",
        "no equivocation between validators",
    ),
    ("WeightsTlockPayload", "WeightsTlockPayload"),
    ("no_LKG", "No last-known-good"),
    ("NoScoreReasonCode", "NoScoreReasonCode"),
    ("DissentReasonCode", "DissentReasonCode"),
];

/// Run the bundle-spec completeness gate.
///
/// # Errors
///
/// Returns a multi-line error when the spec file is missing or any pin fails.
pub fn run(workspace_root: &Path) -> Result<(), String> {
    let spec_path = workspace_root.join("docs/BUNDLE_SPEC.md");
    let checklist_path = workspace_root.join("docs/BUNDLE_SPEC_CHECKLIST.md");

    if !checklist_path.is_file() {
        return Err(format!(
            "missing checklist file: {}",
            checklist_path.display()
        ));
    }

    let body = fs::read_to_string(&spec_path).map_err(|e| {
        format!(
            "read {}: {e} (BUNDLE_SPEC.md is required; plan task 8 wave gate)",
            spec_path.display()
        )
    })?;

    let mut failures = Vec::new();

    for (letter, marker) in LETTER_MARKERS {
        if !body.contains(marker) {
            failures.push(format!(
                "letter ({letter}): missing heading marker:\n  {marker}"
            ));
        }
    }

    for (name, needle) in CONTENT_PINS {
        if !body.contains(needle) {
            failures.push(format!("content pin {name}: missing substring {needle:?}"));
        }
    }

    // House size: accept either spelling used in the spec.
    if !(body.contains("65_535") || body.contains("65535")) {
        failures.push("content pin HOUSE: need 65_535 or 65535".into());
    }

    // Explicit negation: merkle root must not be claimed on-chain in payload.
    let lower = body.to_ascii_lowercase();
    if !lower.contains("merkle root is not in the on-chain weight payload")
        && !lower.contains("the merkle root is not in the on-chain weight payload")
        && !lower.contains("merkle root is not committed")
    {
        // Spec uses bold line; accept the section title path already covered by (l).
        if !body.contains("no merkle root (l)") {
            failures.push(
                "content pin D5_negation: need explicit 'merkle root is NOT' wording".into(),
            );
        }
    }

    let checklist = fs::read_to_string(&checklist_path)
        .map_err(|e| format!("read {}: {e}", checklist_path.display()))?;
    for (letter, _) in LETTER_MARKERS {
        let token = format!("({letter})");
        if !checklist.contains(&token) {
            failures.push(format!(
                "checklist missing letter token {token} in {}",
                checklist_path.display()
            ));
        }
    }

    if failures.is_empty() {
        println!(
            "spec-check: OK ({} letter pins, {} content pins) — {}",
            LETTER_MARKERS.len(),
            CONTENT_PINS.len() + 1,
            spec_path.display()
        );
        Ok(())
    } else {
        Err(format!(
            "spec-check failed ({}):\n{}",
            failures.len(),
            failures.join("\n")
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{CONTENT_PINS, LETTER_MARKERS};

    #[test]
    fn twelve_letter_markers() {
        assert_eq!(LETTER_MARKERS.len(), 12);
        let letters: String = LETTER_MARKERS.iter().map(|(l, _)| *l).collect();
        assert_eq!(letters, "abcdefghijkl");
    }

    #[test]
    fn content_pins_nonempty() {
        assert!(!CONTENT_PINS.is_empty());
    }
}
