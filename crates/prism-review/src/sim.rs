//! Deterministic offline reviewer (CI / local sim, never billed).
//!
//! Maps submission bytes through hashes onto the same lattice as the real
//! reviewer. Identical sources → identical verdicts; a byte-identical copy of
//! a corpus snippet is classified `copied` with score 1.0 (the only way sim
//! deterministically reproduces the real judge's most important edge).

use async_trait::async_trait;
use sha2::{Digest, Sha256};

use crate::prompts::{REVIEW_PROMPT_VERSION, SIMILARITY_PROMPT_VERSION};
use crate::types::{ReviewError, ReviewVerdict, SimilarityKind, SimilarityVerdict, SourceSnippet};
use crate::ReviewBackend;

/// Offline deterministic reviewer.
#[derive(Debug, Default)]
pub struct SimReviewer {
    _private: (),
}

impl SimReviewer {
    /// New sim backend.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

fn latch(bytes: &[u8], mod_u: u64) -> u64 {
    let mut h = Sha256::new();
    h.update(bytes);
    let d = h.finalize();
    u64::from_be_bytes(d[0..8].try_into().unwrap_or([0; 8])) % mod_u
}

#[async_trait]
impl ReviewBackend for SimReviewer {
    async fn review(
        &self,
        architecture_py: &str,
        training_py: &str,
    ) -> Result<ReviewVerdict, ReviewError> {
        let q = 200
            + latch(
                format!("q\0{architecture_py}\0{training_py}").as_bytes(),
                700,
            );
        Ok(ReviewVerdict {
            quality_score: u16::try_from(q.min(u64::from(u16::MAX))).unwrap_or(0),
            issues: vec!["sim-review".into()],
            prompt_version: REVIEW_PROMPT_VERSION,
        })
    }

    /// Architecture-only similarity (similarity v2): the candidate and every
    /// corpus entry are hashed/compared on `architecture.py` bytes alone;
    /// `training.py` never influences the verdict.
    async fn similarity(
        &self,
        architecture_py: &str,
        corpus: &[SourceSnippet],
    ) -> Result<SimilarityVerdict, ReviewError> {
        let cand = hex::encode(Sha256::digest(architecture_py.as_bytes()));

        for s in corpus {
            let d = hex::encode(Sha256::digest(s.architecture_py.as_bytes()));
            if cand == d {
                return Ok(SimilarityVerdict {
                    kind: SimilarityKind::Copied,
                    score: 1.0,
                    closest: Some(s.label.clone()),
                    evidence: vec!["byte-identical architecture".into()],
                    prompt_version: SIMILARITY_PROMPT_VERSION,
                });
            }
        }
        let score =
            f64::from(u32::try_from(latch(format!("s\0{}", &cand).as_bytes(), 1000)).unwrap_or(0))
                / 2000.0;
        Ok(SimilarityVerdict {
            kind: SimilarityKind::Original,
            score,
            closest: corpus.first().map(|c| c.label.clone()),
            evidence: vec!["sim-review".into()],
            prompt_version: SIMILARITY_PROMPT_VERSION,
        })
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[tokio::test]
    async fn sim_deterministic() {
        let r = SimReviewer::new();
        let a = r
            .review("def build_model(ctx): pass", "def train(m,c):\n  return {}")
            .await
            .unwrap();
        let b = r
            .review("def build_model(ctx): pass", "def train(m,c):\n  return {}")
            .await
            .unwrap();
        assert_eq!(a, b);
        assert!((200..=900).contains(&a.quality_score));
    }

    #[tokio::test]
    async fn sim_flags_byte_copy() {
        let r = SimReviewer::new();
        let arch = "def build_model(ctx):\n    pass\n";
        let train = "def train(m,c):\n    return {}\n";
        let corpus = vec![SourceSnippet {
            label: "baseline".into(),
            architecture_py: arch.into(),
            training_py: train.into(),
        }];
        let v = r.similarity(arch, &corpus).await.unwrap();
        assert!(matches!(v.kind, SimilarityKind::Copied));
        assert_eq!(v.closest.as_deref(), Some("baseline"));
    }

    #[tokio::test]
    async fn sim_training_py_is_exempt() {
        // Same training script, different architecture → original.
        let r = SimReviewer::new();
        let train = "def train(m,c):\n    return {}\n";
        let corpus = vec![SourceSnippet {
            label: "subm:aa".into(),
            architecture_py: "def build_model(ctx):\n    return 1\n".into(),
            training_py: train.into(),
        }];
        let v = r
            .similarity("def build_model(ctx):\n    return 2\n", &corpus)
            .await
            .unwrap();
        assert!(matches!(v.kind, SimilarityKind::Original));
    }

    #[tokio::test]
    async fn sim_same_arch_new_training_is_copied() {
        // Same architecture, different training script → still an arch copy.
        let r = SimReviewer::new();
        let arch = "def build_model(ctx):\n    return 1\n";
        let corpus = vec![SourceSnippet {
            label: "subm:aa".into(),
            architecture_py: arch.into(),
            training_py: "def train(m,c):\n    return {'a': 1}\n".into(),
        }];
        let v = r.similarity(arch, &corpus).await.unwrap();
        assert!(matches!(v.kind, SimilarityKind::Copied));
        assert_eq!(v.closest.as_deref(), Some("subm:aa"));
    }
}
