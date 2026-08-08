//! Post-parse guard: LLM similarity must not Score(0) on generic LM tropes.

use crate::types::{SimilarityKind, SimilarityVerdict};

/// Substrings that are standard modern-LM vocabulary (case-insensitive).
/// When *every* evidence line matches one of these (and nothing else
/// substantive), a `suspicious` / soft-`copied` verdict is coerced to
/// `original`.
const GENERIC_TROPE_NEEDLES: &[&str] = &[
    "rmsnorm",
    "layer norm",
    "layernorm",
    "batchnorm",
    "groupnorm",
    "rotary",
    "rope",
    "alibi",
    "positional embed",
    "swiglu",
    "geglu",
    "gated residual",
    "parallel residual",
    "feed-forward",
    "feed forward",
    "multi-head attention",
    "multihead attention",
    "grouped-query",
    "gqa",
    "mqa",
    "pre-norm",
    "post-norm",
    "weight ty",
    "flashattention",
    "flash attention",
];

/// True when `evidence` is non-empty and every line is only a generic trope.
#[must_use]
pub fn evidence_is_only_generic_tropes(evidence: &[String]) -> bool {
    if evidence.is_empty() {
        return false;
    }
    evidence.iter().all(|e| {
        let lower = e.to_ascii_lowercase();
        let trimmed = lower.trim();
        if trimmed.is_empty() {
            return true;
        }
        GENERIC_TROPE_NEEDLES.iter().any(|n| trimmed.contains(n))
    })
}

/// Coerce false-positive LLM similarity verdicts that cite only standard
/// components. Hard `copied` with score ≥ 0.95 is left alone (near-verbatim
/// copies can still mention shared blocks in evidence).
#[must_use]
pub fn coerce_generic_similarity(mut v: SimilarityVerdict) -> SimilarityVerdict {
    let only_generic = evidence_is_only_generic_tropes(&v.evidence);
    match v.kind {
        SimilarityKind::Suspicious if only_generic || v.score < 0.9 => {
            v.kind = SimilarityKind::Original;
            if only_generic {
                v.evidence.insert(
                    0,
                    "coerced: generic LM components are not plagiarism".into(),
                );
                v.evidence.truncate(3);
            }
        }
        SimilarityKind::Copied if only_generic && v.score < 0.95 => {
            v.kind = SimilarityKind::Original;
            v.evidence.insert(
                0,
                "coerced: generic LM components are not plagiarism".into(),
            );
            v.evidence.truncate(3);
        }
        _ => {}
    }
    v
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use crate::prompts::SIMILARITY_PROMPT_VERSION;

    fn verdict(kind: SimilarityKind, score: f64, evidence: &[&str]) -> SimilarityVerdict {
        SimilarityVerdict {
            kind,
            score,
            closest: Some("subm:deadbeef".into()),
            evidence: evidence.iter().map(|s| (*s).to_owned()).collect(),
            prompt_version: SIMILARITY_PROMPT_VERSION,
        }
    }

    #[test]
    fn tropes_coerce_suspicious() {
        let v = coerce_generic_similarity(verdict(
            SimilarityKind::Suspicious,
            0.7,
            &[
                "RMSNorm usage",
                "Rotary embeddings",
                "Gated residual connections",
            ],
        ));
        assert!(matches!(v.kind, SimilarityKind::Original));
    }

    #[test]
    fn tropes_coerce_swiglu_case() {
        let v = coerce_generic_similarity(verdict(
            SimilarityKind::Suspicious,
            0.7,
            &[
                "Parallel residual blocks",
                "SwiGLU feed-forward",
                "Layer normalization",
            ],
        ));
        assert!(matches!(v.kind, SimilarityKind::Original));
    }

    #[test]
    fn unique_structure_keeps_copied() {
        let v = coerce_generic_similarity(verdict(
            SimilarityKind::Copied,
            0.97,
            &["same custom DualPathBlock wiring as subm:aabbccdd"],
        ));
        assert!(matches!(v.kind, SimilarityKind::Copied));
    }
}
