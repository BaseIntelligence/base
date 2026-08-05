//! Pre-LLM copy gate: byte / AST copy detection against a harness corpus,
//! ordered by `created_at`. A candidate that copies an **earlier** harness is
//! rejected without spending an LLM review; ordering ties or unknown
//! timestamps fall through to the LLM.

use sha2::{Digest, Sha256};

use crate::{fingerprint_source, similarity_bps};

/// AST similarity ≥ this → copy class (`cheat` band).
pub const AST_CHEAT_BPS: u16 = 9_500;
/// AST similarity ≥ this (and < cheat) → `suspicious` band.
pub const AST_SUSPICIOUS_BPS: u16 = 8_500;

/// Corpus ids with this prefix are public reference material (the published
/// miner baseline). Miners are told to start from the baseline, so matching it
/// is never a cheat signal — only another *miner's* harness can be copied.
/// This is the baseline-zeroing fix: the reference agent must come out clean.
pub const BASELINE_CORPUS_PREFIX: &str = "baseline";

/// One corpus harness for the copy gate.
#[derive(Debug, Clone)]
pub struct GateCorpusEntry {
    /// Stable id (`harness:<hex>` …).
    pub id: String,
    /// Full primary Python source.
    pub source: String,
    /// Store-assigned creation unix ms (0 = unknown).
    pub created_at_ms: u64,
}

/// Copy-gate hit: the candidate clones `nearest_id` which was created earlier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CopyGateHit {
    /// Nearest corpus id (byte or AST match).
    pub nearest_id: String,
    /// Similarity basis points (10000 for byte-identical).
    pub similarity_bps: u16,
    /// Byte-identical source hash.
    pub byte_identical: bool,
    /// Creation time of the copied harness.
    pub similar_created_at_ms: u64,
}

fn source_hash(source: &str) -> [u8; 32] {
    Sha256::digest(source.as_bytes()).into()
}

/// Pre-LLM gate. Returns `Some(hit)` when the candidate is a byte copy or an
/// AST copy (`similarity >= AST_CHEAT_BPS`) of a corpus harness created
/// **strictly earlier** than the candidate. `None` lets the LLM review run.
///
/// Corpus entries with unknown timestamps (`created_at_ms == 0`) cannot be
/// ordered and never hard-reject (baseline / legacy rows stay LLM-judged).
#[must_use]
pub fn copy_gate(
    candidate_source: &str,
    candidate_created_at_ms: u64,
    corpus: &[GateCorpusEntry],
) -> Option<CopyGateHit> {
    let cand_hash = source_hash(candidate_source);
    let cand_fp = fingerprint_source(candidate_source).ok();

    let mut best: Option<CopyGateHit> = None;
    for entry in corpus {
        // Public baseline entries are reference material, never copy victims.
        if entry.id.starts_with(BASELINE_CORPUS_PREFIX)
            || entry.created_at_ms == 0
            || candidate_created_at_ms == 0
            || entry.created_at_ms >= candidate_created_at_ms
        {
            continue;
        }
        let byte_identical = source_hash(&entry.source) == cand_hash;
        let sim = if byte_identical {
            10_000
        } else {
            match (&cand_fp, fingerprint_source(&entry.source).ok()) {
                (Some(q), Some(fp)) => similarity_bps(q, &fp),
                _ => 0,
            }
        };
        if !byte_identical && sim < AST_CHEAT_BPS {
            continue;
        }
        let hit = CopyGateHit {
            nearest_id: entry.id.clone(),
            similarity_bps: sim,
            byte_identical,
            similar_created_at_ms: entry.created_at_ms,
        };
        let better = best
            .as_ref()
            .is_none_or(|b| hit.similarity_bps > b.similarity_bps);
        if better {
            best = Some(hit);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    const A: &str =
        "def run(task, llm, out):\n    out.write_page('index.html', '<html>a</html>')\n";
    const B: &str = "import math\ndef train(xs):\n    return sum(math.sin(x) for x in xs)\n";

    fn entry(id: &str, source: &str, created: u64) -> GateCorpusEntry {
        GateCorpusEntry {
            id: id.into(),
            source: source.into(),
            created_at_ms: created,
        }
    }

    #[test]
    fn byte_copy_of_earlier_rejected() {
        let hit = copy_gate(A, 2_000, &[entry("harness:v", A, 1_000)]).unwrap();
        assert_eq!(hit.nearest_id, "harness:v");
        assert!(hit.byte_identical);
        assert_eq!(hit.similarity_bps, 10_000);
    }

    #[test]
    fn copy_of_newer_or_same_round_allowed_through() {
        // Victim created *after* the candidate → cannot be the copied one.
        assert!(copy_gate(A, 1_000, &[entry("harness:v", A, 2_000)]).is_none());
        assert!(copy_gate(A, 1_000, &[entry("harness:v", A, 1_000)]).is_none());
    }

    #[test]
    fn unknown_timestamps_fall_through() {
        assert!(copy_gate(A, 1_000, &[entry("baseline", A, 0)]).is_none());
        assert!(copy_gate(A, 0, &[entry("harness:v", A, 1_000)]).is_none());
    }

    #[test]
    fn baseline_entries_never_reject() {
        assert!(copy_gate(A, 2_000, &[entry("baseline", A, 1_000)]).is_none());
    }

    #[test]
    fn unrelated_source_clean() {
        assert!(copy_gate(B, 2_000, &[entry("harness:v", A, 1_000)]).is_none());
    }

    #[test]
    fn ast_copy_of_earlier_rejected() {
        // Cosmetic rename: same structure, different identifiers/comments.
        let renamed = A.replace("task", "brief").replace("out", "sink");
        let hit = copy_gate(&renamed, 2_000, &[entry("harness:v", A, 1_000)]);
        // Identifier renames normalize away in the fingerprint.
        assert!(hit.is_some(), "AST near-copy must trip the gate");
    }
}
