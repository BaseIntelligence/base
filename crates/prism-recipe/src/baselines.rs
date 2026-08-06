//! Embedded v3 reference baselines (E6): the two anchor submissions the
//! composite anchor protocol is measured against (`docs/PRISM.md`).
//!
//! Both are full source trees (`architecture.py` + `training.py` satisfy the
//! recipe contract; `count_params.py` statically verifies the parameter cap;
//! `NOTES.md` documents the config and harness-contract findings). Corpus
//! ids start with `baseline` (`BASELINE_CORPUS_PREFIX` in `challenge-agentic`)
//! so the copy gate exempts them as published prior art.

/// Submission id for the Transformer++ anchor (modern GPT at the 350M cap).
pub const BASELINE_V3_TRANSFORMER_PP_ID: &str = "baseline-transformer-pp";

/// Submission id for the 3:1 gated delta-net/attention hybrid anchor.
pub const BASELINE_V3_HYBRID_DELTA_ID: &str = "baseline-hybrid-delta";

/// Transformer++ tree: baseline-relative path → contents (sorted by path).
pub const BASELINE_V3_TRANSFORMER_PP_FILES: &[(&str, &str)] = &[
    (
        "NOTES.md",
        include_str!("../baselines/transformer_pp/NOTES.md"),
    ),
    (
        "architecture.py",
        include_str!("../baselines/transformer_pp/architecture.py"),
    ),
    (
        "count_params.py",
        include_str!("../baselines/transformer_pp/count_params.py"),
    ),
    (
        "training.py",
        include_str!("../baselines/transformer_pp/training.py"),
    ),
];

/// Hybrid delta-net tree: baseline-relative path → contents (sorted).
pub const BASELINE_V3_HYBRID_DELTA_FILES: &[(&str, &str)] = &[
    (
        "NOTES.md",
        include_str!("../baselines/hybrid_delta/NOTES.md"),
    ),
    (
        "architecture.py",
        include_str!("../baselines/hybrid_delta/architecture.py"),
    ),
    (
        "count_params.py",
        include_str!("../baselines/hybrid_delta/count_params.py"),
    ),
    (
        "training.py",
        include_str!("../baselines/hybrid_delta/training.py"),
    ),
];

/// Both v3 baselines as `(submission id, tree)` pairs.
#[must_use]
pub fn baselines_v3() -> [(&'static str, &'static [(&'static str, &'static str)]); 2] {
    [
        (
            BASELINE_V3_TRANSFORMER_PP_ID,
            BASELINE_V3_TRANSFORMER_PP_FILES,
        ),
        (BASELINE_V3_HYBRID_DELTA_ID, BASELINE_V3_HYBRID_DELTA_FILES),
    ]
}

/// One file from a baseline tree by path.
#[must_use]
pub fn file<'a>(tree: &'a [(&str, &str)], path: &str) -> Option<&'a str> {
    tree.iter().find(|(p, _)| *p == path).map(|(_, c)| *c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::check_contract;

    #[test]
    fn trees_are_sorted_and_complete() {
        for (id, tree) in baselines_v3() {
            let paths: Vec<&str> = tree.iter().map(|(p, _)| *p).collect();
            let mut sorted = paths.clone();
            sorted.sort_unstable();
            assert_eq!(paths, sorted, "{id} tree must stay sorted by path");
            for required in [
                "architecture.py",
                "training.py",
                "count_params.py",
                "NOTES.md",
            ] {
                assert!(file(tree, required).is_some(), "{id} missing {required}");
            }
        }
    }

    #[test]
    fn v3_baselines_satisfy_contract() {
        for (id, tree) in baselines_v3() {
            let arch = file(tree, "architecture.py").expect("architecture");
            let train = file(tree, "training.py").expect("training");
            check_contract(arch, train).unwrap_or_else(|e| panic!("{id}: {e}"));
        }
    }

    #[test]
    fn count_params_documents_cap_check() {
        // `count_params.py` convention (see each baseline's NOTES.md):
        // prints a per-component breakdown plus `TOTAL (static math)` and,
        // when torch is available, `TOTAL (torch build)` which must equal
        // the static figure; then a `cap check: <total> <= 350,000,000:
        // True` line and a non-zero exit on overflow. The anchor runs MUST
        // pass under the production 350M cap.
        for (id, tree) in baselines_v3() {
            let cp = file(tree, "count_params.py").expect("count_params");
            assert!(cp.contains("350_000_000"), "{id} cap constant");
            assert!(cp.contains("cap check"), "{id} cap check line");
            assert!(cp.contains("TOTAL (static math)"), "{id} static total");
        }
    }
}
