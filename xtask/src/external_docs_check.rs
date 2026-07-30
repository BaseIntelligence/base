//! Fail if external miner docs drift from `bundle` `PROTOCOL_VERSION`,
//! or if `docs/THREAT_MODEL.md` D19 claim is not word-for-word vs plan pin.

use std::fs;
use std::path::Path;

/// Plan D19 claim body (after "verbatim in docs:"). Must match `THREAT_MODEL` section 1.
const D19_VERBATIM: &str = "gbase guarantees *no equivocation between validators* and *no undetected deviation by the gateway from the owner-signed challenge and measurement artifacts*. It does **not** guarantee (i) that a challenge's scores are honest, (ii) that the owner is honest — the owner signs the trust roots and runs the gateway, so a malicious owner can authorize a dishonest challenge or a backdoored measurement, (iii) completeness beyond what D24 provides, nor (iv) **chain-anchored, third-party-auditable non-equivocation** — per D5 the property is peer-consensus plus local evidence, verifiable by the participating validators and not by an outside observer after the fact.";

/// Marker comment required in external miner docs.
const BADGE_COMMENT_PREFIX: &str = "<!-- protocol_version:";

/// Run external-docs + D19 gates.
///
/// # Errors
///
/// Returns a multi-line error when any pin fails.
pub fn run(workspace_root: &Path) -> Result<(), String> {
    let mut failures = Vec::new();

    let protocol_version = read_bundle_protocol_version(workspace_root)?;
    check_external_miner_docs(workspace_root, protocol_version, &mut failures)?;
    check_threat_model_d19(workspace_root, &mut failures)?;
    check_threat_model_supporting_pins(workspace_root, &mut failures)?;

    if failures.is_empty() {
        println!(
            "external-docs-check OK (protocol_version={protocol_version}, D19 verbatim match)"
        );
        Ok(())
    } else {
        Err(format!(
            "external-docs-check failed ({}):\n{}",
            failures.len(),
            failures
                .iter()
                .map(|f| format!("  - {f}"))
                .collect::<Vec<_>>()
                .join("\n")
        ))
    }
}

fn read_bundle_protocol_version(workspace_root: &Path) -> Result<u16, String> {
    let types_rs = workspace_root.join("crates/bundle/src/types.rs");
    let body =
        fs::read_to_string(&types_rs).map_err(|e| format!("read {}: {e}", types_rs.display()))?;
    // pub const PROTOCOL_VERSION: u16 = 1;
    for line in body.lines() {
        let t = line.trim();
        if t.starts_with("pub const PROTOCOL_VERSION") {
            let Some(eq) = t.split('=').nth(1) else {
                continue;
            };
            let num = eq
                .trim()
                .trim_end_matches(';')
                .trim()
                .parse::<u16>()
                .map_err(|e| format!("parse PROTOCOL_VERSION from {t:?}: {e}"))?;
            return Ok(num);
        }
    }
    Err(format!(
        "PROTOCOL_VERSION const not found in {}",
        types_rs.display()
    ))
}

fn check_external_miner_docs(
    workspace_root: &Path,
    expected: u16,
    failures: &mut Vec<String>,
) -> Result<(), String> {
    let dir = workspace_root.join("docs/external-miner");
    if !dir.is_dir() {
        failures.push(format!("missing directory {}", dir.display()));
        return Ok(());
    }

    let readme = dir.join("README.md");
    let readme_body =
        fs::read_to_string(&readme).map_err(|e| format!("read {}: {e}", readme.display()))?;
    match extract_badge_version(&readme_body) {
        Ok(v) if v == expected => {}
        Ok(v) => failures.push(format!(
            "docs/external-miner/README.md protocol_version badge={v} != bundle PROTOCOL_VERSION={expected}"
        )),
        Err(e) => failures.push(format!("docs/external-miner/README.md: {e}")),
    }

    // Every markdown file under external-miner must declare the same badge comment.
    let entries = fs::read_dir(&dir).map_err(|e| format!("read_dir {}: {e}", dir.display()))?;
    for ent in entries {
        let ent = ent.map_err(|e| format!("dirent: {e}"))?;
        let path = ent.path();
        if path.extension().and_then(|s| s.to_str()) != Some("md") {
            continue;
        }
        let body =
            fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        match extract_badge_version(&body) {
            Ok(v) if v == expected => {}
            Ok(v) => failures.push(format!(
                "{} badge protocol_version={v} != {expected}",
                path.strip_prefix(workspace_root).unwrap_or(&path).display()
            )),
            Err(e) => failures.push(format!(
                "{}: {e}",
                path.strip_prefix(workspace_root).unwrap_or(&path).display()
            )),
        }
    }

    Ok(())
}

fn extract_badge_version(body: &str) -> Result<u16, String> {
    for line in body.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix(BADGE_COMMENT_PREFIX) {
            let num = rest
                .trim()
                .trim_end_matches("-->")
                .trim()
                .parse::<u16>()
                .map_err(|e| format!("bad protocol_version comment {t:?}: {e}"))?;
            return Ok(num);
        }
    }
    // Also accept bold badge line: **Bundle `protocol_version`:** `1`
    for line in body.lines() {
        if line.contains("protocol_version") && line.contains('`') {
            // find last `N` numeric
            let mut last = None;
            let mut cur = String::new();
            let mut in_tick = false;
            for ch in line.chars() {
                if ch == '`' {
                    if in_tick {
                        if let Ok(v) = cur.parse::<u16>() {
                            last = Some(v);
                        }
                        cur.clear();
                        in_tick = false;
                    } else {
                        in_tick = true;
                        cur.clear();
                    }
                } else if in_tick {
                    cur.push(ch);
                }
            }
            if let Some(v) = last {
                return Ok(v);
            }
        }
    }
    Err(format!(
        "missing `{BADGE_COMMENT_PREFIX} N -->` badge (and no parseable protocol_version backticks)"
    ))
}

fn check_threat_model_d19(workspace_root: &Path, failures: &mut Vec<String>) -> Result<(), String> {
    let path = workspace_root.join("docs/THREAT_MODEL.md");
    let body = fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;

    // Prefer fenced section after the D19 heading.
    let Some(idx) = body.find("## 1. D19") else {
        failures.push("docs/THREAT_MODEL.md missing heading `## 1. D19`".into());
        return Ok(());
    };
    let rest = &body[idx..];
    let Some(after_blank) = rest.split("\n\n").nth(2) else {
        failures.push("docs/THREAT_MODEL.md: could not locate D19 claim paragraph".into());
        return Ok(());
    };
    // Paragraph until next blank line or heading
    let para = after_blank
        .lines()
        .take_while(|l| !l.starts_with('#') && !l.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    let para = para.trim();

    // Also try: find the claim by unique prefix if structure drifts.
    let matched = para == D19_VERBATIM || body.contains(D19_VERBATIM);

    if !matched {
        failures.push(format!(
            "docs/THREAT_MODEL.md D19 claim is not word-for-word vs plan D19.\n  expected (first 120 chars): {:?}\n  found paragraph (first 120): {:?}",
            D19_VERBATIM.chars().take(120).collect::<String>(),
            para.chars().take(120).collect::<String>()
        ));
    }
    Ok(())
}

fn check_threat_model_supporting_pins(
    workspace_root: &Path,
    failures: &mut Vec<String>,
) -> Result<(), String> {
    let path = workspace_root.join("docs/THREAT_MODEL.md");
    let body = fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let pins = [
        ("D11", "do **not** claim env **values** are verified"),
        ("D5_payload", "WeightsTlockPayload"),
        (
            "D5_negation",
            "merkle root is NOT committed in the on-chain weight payload",
        ),
        (
            "R12",
            "The owner is the trust root AND the gateway operator",
        ),
        ("peer_evidence", "peer-consensus plus local evidence"),
    ];
    for (name, needle) in pins {
        if !body.contains(needle) {
            failures.push(format!(
                "docs/THREAT_MODEL.md missing pin {name}: {needle:?}"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_badge_from_comment() {
        let s = "<!-- protocol_version: 1 -->\nhello\n";
        assert_eq!(extract_badge_version(s).unwrap(), 1);
    }

    #[test]
    fn d19_constant_nonempty() {
        assert!(D19_VERBATIM.contains("no equivocation between validators"));
        assert!(D19_VERBATIM.contains("chain-anchored, third-party-auditable non-equivocation"));
    }
}
