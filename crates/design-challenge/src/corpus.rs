//! Anti-cheat corpus: other miners' prior art only (hotkey + coldkey).
//! Gate + review share this module; pass the candidate row explicitly.

use challenge_agentic::{same_miner_identity, CorpusEntry, GateCorpusEntry};
use design_store::HarnessRow;

const BASELINE_AGENT: &str =
    include_str!("../../../docs/external-miner/examples/design-baseline/agent.py");

fn other_miners<'a>(
    cand: &'a HarnessRow,
    recent: &'a [HarnessRow],
) -> impl Iterator<Item = &'a HarnessRow> {
    recent.iter().filter(move |h| {
        h.id != cand.id
            && !same_miner_identity(
                &cand.miner_hotkey,
                cand.miner_coldkey.as_deref(),
                &h.miner_hotkey,
                h.miner_coldkey.as_deref(),
            )
    })
}

/// Pre-LLM copy-gate corpus (`created_at_ms` kept for gate ordering).
#[must_use]
pub fn gate_corpus(candidate: &HarnessRow, recent: &[HarnessRow]) -> Vec<GateCorpusEntry> {
    other_miners(candidate, recent)
        .map(|h| GateCorpusEntry {
            id: format!("harness:{}", h.id),
            source: h.agent_py.clone(),
            created_at_ms: h.created_at_ms,
        })
        .collect()
}

/// Reviewer corpus: baseline + other miners' earlier harnesses.
#[must_use]
pub fn review_corpus(candidate: &HarnessRow, recent: &[HarnessRow]) -> Vec<CorpusEntry> {
    let mut corpus = vec![CorpusEntry {
        id: "baseline".into(),
        source: BASELINE_AGENT.to_owned(),
    }];
    let ts = candidate.created_at_ms;
    corpus.extend(other_miners(candidate, recent).filter_map(|h| {
        if h.created_at_ms == 0 || (ts > 0 && h.created_at_ms >= ts) {
            return None;
        }
        Some(CorpusEntry {
            id: format!("harness:{}", h.id),
            source: h.agent_py.clone(),
        })
    }));
    corpus
}

#[cfg(test)]
mod tests {
    use super::*;

    fn harness(
        id: &str,
        miner: &str,
        coldkey: Option<&str>,
        source: &str,
        created_at_ms: u64,
    ) -> HarnessRow {
        HarnessRow {
            id: id.into(),
            miner_hotkey: miner.into(),
            miner_coldkey: coldkey.map(str::to_owned),
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
    const CC: &str = "cc";
    const COLD_X: &str = "11";
    const COLD_Y: &str = "22";

    fn ids(entries: &[CorpusEntry]) -> Vec<&str> {
        entries.iter().map(|e| e.id.as_str()).collect()
    }

    fn gate_ids(entries: &[GateCorpusEntry]) -> Vec<&str> {
        entries.iter().map(|e| e.id.as_str()).collect()
    }

    #[test]
    fn own_previous_version_is_never_compared_against() {
        let v1 = harness("h1", AA, Some(COLD_X), "def run(t):\n    pass\n", 1_000);
        let v2 = harness("h2", AA, Some(COLD_X), "def run(t):\n    pass\n", 2_000);
        let recent = vec![v2.clone(), v1];
        assert!(gate_corpus(&v2, &recent).is_empty());
        assert_eq!(ids(&review_corpus(&v2, &recent)), vec!["baseline"]);
    }

    #[test]
    fn hotkey_match_is_case_insensitive() {
        let mine_old = harness("h1", "AABB", Some(COLD_X), "old\n", 1_000);
        let mine_new = harness("h2", "aabb", Some(COLD_X), "new\n", 2_000);
        let recent = vec![mine_new.clone(), mine_old];
        assert!(gate_corpus(&mine_new, &recent).is_empty());
        assert_eq!(ids(&review_corpus(&mine_new, &recent)), vec!["baseline"]);
    }

    #[test]
    fn same_coldkey_different_hotkey_is_excluded() {
        let prior = harness("h1", AA, Some(COLD_X), "prior\n", 1_000);
        let next = harness("h2", BB, Some(COLD_X), "next\n", 2_000);
        let recent = vec![next.clone(), prior];
        assert!(gate_corpus(&next, &recent).is_empty());
        assert_eq!(ids(&review_corpus(&next, &recent)), vec!["baseline"]);
    }

    #[test]
    fn different_coldkey_prior_art_stays_in_both_corpora() {
        let victim = harness("h1", BB, Some(COLD_Y), "def run(t):\n    pass\n", 1_000);
        let copier = harness("h2", AA, Some(COLD_X), "def run(t):\n    pass\n", 2_000);
        let recent = vec![copier.clone(), victim];
        assert_eq!(gate_ids(&gate_corpus(&copier, &recent)), vec!["harness:h1"]);
        assert_eq!(
            ids(&review_corpus(&copier, &recent)),
            vec!["baseline", "harness:h1"]
        );
    }

    #[test]
    fn other_miner_prior_art_stays_in_both_corpora() {
        let victim = harness("h1", BB, None, "def run(t):\n    pass\n", 1_000);
        let copier = harness("h2", AA, None, "def run(t):\n    pass\n", 2_000);
        let recent = vec![copier.clone(), victim];
        assert_eq!(gate_ids(&gate_corpus(&copier, &recent)), vec!["harness:h1"]);
        assert_eq!(
            ids(&review_corpus(&copier, &recent)),
            vec!["baseline", "harness:h1"]
        );
    }

    #[test]
    fn missing_coldkey_falls_back_to_hotkey_only() {
        let prior = harness("h1", AA, None, "prior\n", 1_000);
        let next = harness("h2", BB, Some(COLD_X), "next\n", 2_000);
        let foreign = harness("h3", CC, Some(COLD_Y), "foreign\n", 1_500);
        let recent = vec![next.clone(), foreign, prior];
        assert_eq!(
            gate_ids(&gate_corpus(&next, &recent)),
            vec!["harness:h3", "harness:h1"]
        );
    }

    #[test]
    fn candidate_outside_the_recent_window_still_excludes_itself() {
        let mine_old = harness("h1", AA, Some(COLD_X), "old\n", 1_000);
        let theirs = harness("h3", BB, Some(COLD_Y), "theirs\n", 1_500);
        let mine_new = harness("h2", AA, Some(COLD_X), "new\n", 2_000);
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
        let candidate = harness("h1", AA, Some(COLD_X), "mine\n", 1_000);
        let later = harness("h2", BB, Some(COLD_Y), "later\n", 5_000);
        let unknown = harness("h3", BB, Some(COLD_Y), "legacy\n", 0);
        let recent = vec![later, unknown];
        assert_eq!(ids(&review_corpus(&candidate, &recent)), vec!["baseline"]);
        assert_eq!(gate_corpus(&candidate, &recent).len(), 2);
    }

    #[test]
    fn legacy_candidate_keeps_timestamped_other_hotkeys() {
        let legacy = harness("h0", AA, Some(COLD_X), "legacy\n", 0);
        let prior = harness("h1", BB, Some(COLD_Y), "prior\n", 1_000);
        assert_eq!(
            ids(&review_corpus(&legacy, &[prior])),
            vec!["baseline", "harness:h1"]
        );
    }

    #[test]
    fn baseline_is_always_available_to_the_reviewer() {
        let corpus = review_corpus(&harness("h1", AA, None, "mine\n", 1_000), &[]);
        assert_eq!(ids(&corpus), vec!["baseline"]);
        assert!(corpus[0].source.contains("def run("));
    }
}
