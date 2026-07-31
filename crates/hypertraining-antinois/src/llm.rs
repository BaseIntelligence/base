//! LLM advisory hooks — never a gate (brief §12.4).

/// Semantic category labels for triage (advisory only).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffCategory {
    /// Kernel fusion style change.
    KernelFusion,
    /// Memory layout / reordering.
    MemoryReorder,
    /// Communication overlap.
    CommOverlap,
    /// Scheduling change.
    Scheduling,
    /// Cosmetic / non-semantic.
    Cosmetic,
    /// Unknown / unclassified.
    Unknown,
}

/// Non-binding advisory note produced by an LLM (or noop).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdvisoryNote {
    /// Suggested category.
    pub category: DiffCategory,
    /// Free-text hypothesis for human reviewers (never machine-gated).
    pub explanation: String,
    /// Suspect-pattern flags (dead code, free renames, …).
    pub suspect_flags: Vec<String>,
}

impl AdvisoryNote {
    /// Empty advisory (noop default).
    #[must_use]
    pub fn empty() -> Self {
        Self {
            category: DiffCategory::Unknown,
            explanation: String::new(),
            suspect_flags: Vec::new(),
        }
    }
}

/// Optional LLM advisory. Implementations MUST NOT block admit/reject/pay.
pub trait LlmAdvisory: Send + Sync {
    /// Classify a normalized diff for human triage.
    fn classify_diff(&self, normalized_diff: &str) -> AdvisoryNote;

    /// Explain L1 vs L2 disagreement for humans.
    fn explain_level_disagreement(
        &self,
        source_sim: f64,
        binary_sim: f64,
        normalized_diff: &str,
    ) -> AdvisoryNote;
}

/// Default advisory: always empty; never influences gates.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopLlm;

impl LlmAdvisory for NoopLlm {
    fn classify_diff(&self, _normalized_diff: &str) -> AdvisoryNote {
        AdvisoryNote::empty()
    }

    fn explain_level_disagreement(
        &self,
        _source_sim: f64,
        _binary_sim: f64,
        _normalized_diff: &str,
    ) -> AdvisoryNote {
        AdvisoryNote::empty()
    }
}

/// Run advisory and return the note — callers must ignore it for gate decisions.
#[must_use]
pub fn advisory_only<A: LlmAdvisory>(llm: &A, normalized_diff: &str) -> AdvisoryNote {
    llm.classify_diff(normalized_diff)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_never_blocks_and_returns_empty() {
        let n = NoopLlm.classify_diff("anything injection ignore previous");
        assert_eq!(n.category, DiffCategory::Unknown);
        assert!(n.explanation.is_empty());
        assert!(n.suspect_flags.is_empty());
    }
}
