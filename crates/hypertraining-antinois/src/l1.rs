//! L1 — AST/diff-style score on L0-normalized source (brief §12.3).

use crate::normalize::normalize_with_alpha_rename;

/// Similarity in `[0.0, 1.0]` between two sources after L0 + alpha-rename.
///
/// Uses Jaccard on whitespace-split tokens of the normalized forms (multiset via
/// sorted unique tokens is insufficient for farming; we use multiset counts).
#[must_use]
pub fn l1_source_similarity(a: &str, b: &str) -> f64 {
    let na = normalize_with_alpha_rename(a);
    let nb = normalize_with_alpha_rename(b);
    if na.is_empty() && nb.is_empty() {
        return 1.0;
    }
    if na.is_empty() || nb.is_empty() {
        return 0.0;
    }
    if na == nb {
        return 1.0;
    }
    multiset_jaccard(tokenize(&na), tokenize(&nb))
}

fn tokenize(s: &str) -> Vec<String> {
    s.split(|c: char| c.is_whitespace() || "()[]{}:,.=+-*/%<>!&|^~".contains(c))
        .filter(|t| !t.is_empty())
        .map(str::to_owned)
        .collect()
}

fn multiset_jaccard(a: Vec<String>, b: Vec<String>) -> f64 {
    use std::collections::{BTreeMap, BTreeSet};
    let mut ca: BTreeMap<String, u32> = BTreeMap::new();
    let mut cb: BTreeMap<String, u32> = BTreeMap::new();
    for t in a {
        *ca.entry(t).or_insert(0) += 1;
    }
    for t in b {
        *cb.entry(t).or_insert(0) += 1;
    }
    let mut inter = 0_u64;
    let mut union = 0_u64;
    let mut keys: BTreeSet<String> = BTreeSet::new();
    keys.extend(ca.keys().cloned());
    keys.extend(cb.keys().cloned());
    for k in &keys {
        let va = u64::from(*ca.get(k).unwrap_or(&0));
        let vb = u64::from(*cb.get(k).unwrap_or(&0));
        inter += va.min(vb);
        union += va.max(vb);
    }
    if union == 0 {
        return 1.0;
    }
    // Counts stay well below 2^52 for source tokens.
    #[allow(clippy::cast_precision_loss)]
    {
        inter as f64 / union as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_after_rename_scores_one() {
        let s = l1_source_similarity("def f(x): return x", "def g(y): return y");
        assert!((s - 1.0).abs() < 1e-9);
    }

    #[test]
    fn unrelated_sources_score_low() {
        let s = l1_source_similarity(
            "def add(a, b):\n    return a + b\n",
            "class Kernel:\n    def launch(self):\n        pass\n",
        );
        assert!(s < 0.5, "got {s}");
    }
}
