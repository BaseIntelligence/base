//! Shared anti-cheat corpus: other hotkeys' prior art only.
//!
//! Same-hotkey revisions are never comparison material. Review victims must be
//! strictly earlier than the candidate. Gate + review share this module so they
//! cannot drift. Pass the candidate row explicitly (not via recent-list lookup).

use challenge_agentic::{CorpusEntry, GateCorpusEntry};
use design_store::HarnessRow;

const BASELINE_AGENT: &str =
    include_str!("../../../docs/external-miner/examples/design-baseline/agent.py");

fn other_miners<'a>(
    candidate: &'a HarnessRow,
    recent: &'a [HarnessRow],
) -> impl Iterator<Item = &'a HarnessRow> {
    let miner = candidate.miner_hotkey.to_ascii_lowercase();
    recent
        .iter()
        .filter(move |h| h.id != candidate.id && h.miner_hotkey.to_ascii_lowercase() != miner)
}

fn corpus_id(h: &HarnessRow) -> String {
    format!("harness:{}", h.id)
}

/// Pre-LLM copy-gate corpus (`created_at_ms` kept for gate ordering).
#[must_use]
pub fn gate_corpus(candidate: &HarnessRow, recent: &[HarnessRow]) -> Vec<GateCorpusEntry> {
    other_miners(candidate, recent)
        .map(|h| GateCorpusEntry {
            id: corpus_id(h),
            source: h.agent_py.clone(),
            created_at_ms: h.created_at_ms,
        })
        .collect()
}

/// Reviewer corpus: baseline + other hotkeys' earlier harnesses.
/// Untimestamped rows are dropped; a legacy candidate (`created_at_ms == 0`)
/// keeps every timestamped other-hotkey row so the corpus cannot go empty.
#[must_use]
pub fn review_corpus(candidate: &HarnessRow, recent: &[HarnessRow]) -> Vec<CorpusEntry> {
    let mut corpus = vec![CorpusEntry {
        id: "baseline".into(),
        source: BASELINE_AGENT.to_owned(),
    }];
    let cand_ts = candidate.created_at_ms;
    corpus.extend(other_miners(candidate, recent).filter_map(|h| {
        if h.created_at_ms == 0 || (cand_ts > 0 && h.created_at_ms >= cand_ts) {
            return None;
        }
        Some(CorpusEntry {
            id: corpus_id(h),
            source: h.agent_py.clone(),
        })
    }));
    corpus
}

#[cfg(test)]
mod tests {
    use super::*;

    fn harness(id: &str, miner: &str, source: &str, created_at_ms: u64) -> HarnessRow {
        HarnessRow {
            id: id.into(),
            miner_hotkey: miner.into(),
            agent_py: source.into(),
            pyproject_toml: "[project]\nname='x'\nversion='0.1.0'\n".into(),
            extra_files: std::collections::BTreeMap::new(),
            active: true,
            eliminated_until_round: 0,
            created_at_ms,
        }
    }

    const AA: &str = "aa";
    const BB: &str = "bb";

    fn ids(entries: &[CorpusEntry]) -> Vec<&str> {
        entries.iter().map(|e| e.id.as_str()).collect()
    }

    fn gate_ids(entries: &[GateCorpusEntry]) -> Vec<&str> {
        entries.iter().map(|e| e.id.as_str()).collect()
    }

    #[test]
    fn own_previous_version_is_never_compared_against() {
        let v1 = harness("h1", AA, "def run(t):\n    pass\n", 1_000);
        let v2 = harness("h2", AA, "def run(t):\n    pass\n", 2_000);
        let recent = vec![v2.clone(), v1];

        assert!(
            gate_corpus(&v2, &recent).is_empty(),
            "a miner's own v1 must not be a copy victim for their v2"
        );
        assert_eq!(
            ids(&review_corpus(&v2, &recent)),
            vec!["baseline"],
            "self-revision must not reach the LLM corpus either"
        );
    }

    #[test]
    fn hotkey_match_is_case_insensitive() {
        let mine_old = harness("h1", "AABB", "old\n", 1_000);
        let mine_new = harness("h2", "aabb", "new\n", 2_000);
        let recent = vec![mine_new.clone(), mine_old];
        assert!(gate_corpus(&mine_new, &recent).is_empty());
        assert_eq!(ids(&review_corpus(&mine_new, &recent)), vec!["baseline"]);
    }

    #[test]
    fn other_miner_prior_art_stays_in_both_corpora() {
        let victim = harness("h1", BB, "def run(t):\n    pass\n", 1_000);
        let copier = harness("h2", AA, "def run(t):\n    pass\n", 2_000);
        let recent = vec![copier.clone(), victim];

        assert_eq!(gate_ids(&gate_corpus(&copier, &recent)), vec!["harness:h1"]);
        assert_eq!(
            ids(&review_corpus(&copier, &recent)),
            vec!["baseline", "harness:h1"]
        );
    }

    #[test]
    fn candidate_outside_the_recent_window_still_excludes_itself() {
        // The candidate is deliberately absent from `recent` (aged out): the
        // rules must come from the candidate row, not from a lookup.
        let mine_old = harness("h1", AA, "old\n", 1_000);
        let theirs = harness("h3", BB, "theirs\n", 1_500);
        let mine_new = harness("h2", AA, "new\n", 2_000);
        let recent = vec![theirs, mine_old];

        assert_eq!(
            gate_ids(&gate_corpus(&mine_new, &recent)),
            vec!["harness:h3"]
        );
        assert_eq!(
            ids(&review_corpus(&mine_new, &recent)),
            vec!["baseline", "harness:h3"]
        );
    }

    #[test]
    fn review_corpus_holds_prior_art_only() {
        let candidate = harness("h1", AA, "mine\n", 1_000);
        let later = harness("h2", BB, "later\n", 5_000);
        let unknown = harness("h3", BB, "legacy\n", 0);
        let recent = vec![later, unknown];

        // A later copycat must never make the original look like the copier.
        assert_eq!(ids(&review_corpus(&candidate, &recent)), vec!["baseline"]);
        // The gate keeps both and orders them itself.
        assert_eq!(gate_corpus(&candidate, &recent).len(), 2);
    }

    #[test]
    fn legacy_candidate_keeps_timestamped_other_hotkeys() {
        let legacy = harness("h0", AA, "legacy\n", 0);
        let prior = harness("h1", BB, "prior\n", 1_000);
        let recent = vec![prior];
        assert_eq!(
            ids(&review_corpus(&legacy, &recent)),
            vec!["baseline", "harness:h1"],
            "unknown candidate timestamp must not empty the review corpus"
        );
    }

    #[test]
    fn baseline_is_always_available_to_the_reviewer() {
        let candidate = harness("h1", AA, "mine\n", 1_000);
        let corpus = review_corpus(&candidate, &[]);
        assert_eq!(ids(&corpus), vec!["baseline"]);
        assert!(
            corpus[0].source.contains("def run("),
            "baseline agent source"
        );
    }
}
