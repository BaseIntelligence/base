//! Integer similarity in basis points (0..=10000) + top-k corpus search.

use std::collections::{BTreeMap, BTreeSet};

use crate::fingerprint::Fingerprint;

/// One ranked corpus neighbor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Neighbor {
    /// Corpus entry id.
    pub id: String,
    /// Similarity in basis points (0..=10000).
    pub similarity_bps: u16,
    /// Short structural overlap summary for LLM tools.
    pub summary: String,
}

/// Similarity in basis points: `0` unrelated … `10000` structurally identical.
///
/// Weighted blend of structure digest equality, node-count cosine (scaled),
/// and Jaccard on imports / defs / calls. All integer arithmetic.
#[must_use]
pub fn similarity_bps(a: &Fingerprint, b: &Fingerprint) -> u16 {
    if a.structure_digest == b.structure_digest {
        return 10_000;
    }
    // Weights sum to 10_000.
    let nodes = (u32::from(cosine_bps(&a.node_counts, &b.node_counts)) * 5_000) / 10_000;
    let imports = (u32::from(jaccard_bps(&a.imports, &b.imports)) * 2_000) / 10_000;
    let defs = (u32::from(jaccard_bps(&a.defs, &b.defs)) * 1_500) / 10_000;
    let calls = (u32::from(jaccard_bps(&a.calls, &b.calls)) * 1_500) / 10_000;
    u16::try_from((nodes + imports + defs + calls).min(10_000)).unwrap_or(10_000)
}

/// Rank corpus by descending similarity; ties broken by id ascending.
#[must_use]
pub fn top_k_nearest(
    query: &Fingerprint,
    corpus: &[(String, Fingerprint)],
    k: usize,
) -> Vec<Neighbor> {
    if k == 0 {
        return Vec::new();
    }
    let mut ranked: Vec<Neighbor> = corpus
        .iter()
        .map(|(id, fp)| {
            let similarity_bps = similarity_bps(query, fp);
            Neighbor {
                id: id.clone(),
                similarity_bps,
                summary: structural_diff_summary(query, fp),
            }
        })
        .collect();
    ranked.sort_by(|a, b| {
        b.similarity_bps
            .cmp(&a.similarity_bps)
            .then_with(|| a.id.cmp(&b.id))
    });
    ranked.truncate(k);
    ranked
}

/// Compact fingerprint summary for `ast_summary` tool.
#[must_use]
pub fn summarize_fingerprint(fp: &Fingerprint) -> String {
    let top_nodes: Vec<String> = {
        let mut items: Vec<(&String, &u32)> = fp.node_counts.iter().collect();
        items.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
        items
            .into_iter()
            .take(12)
            .map(|(k, v)| format!("{k}:{v}"))
            .collect()
    };
    format!(
        "digest={} nodes=[{}] imports={} defs={} calls={}",
        fp.structure_hex(),
        top_nodes.join(","),
        join_set(&fp.imports, 16),
        join_set(&fp.defs, 16),
        join_set(&fp.calls, 16),
    )
}

/// Structural overlap / delta summary for `ast_diff_nearest`.
#[must_use]
pub fn structural_diff_summary(a: &Fingerprint, b: &Fingerprint) -> String {
    let bps = similarity_bps(a, b);
    let imports_left = set_diff(&a.imports, &b.imports);
    let imports_right = set_diff(&b.imports, &a.imports);
    let defs_left = set_diff(&a.defs, &b.defs);
    let defs_right = set_diff(&b.defs, &a.defs);
    format!(
        "similarity_bps={bps} digest_eq={} only_a_imports=[{}] only_b_imports=[{}] only_a_defs=[{}] only_b_defs=[{}]",
        a.structure_digest == b.structure_digest,
        join_set(&imports_left, 8),
        join_set(&imports_right, 8),
        join_set(&defs_left, 8),
        join_set(&defs_right, 8),
    )
}

fn set_diff(a: &BTreeSet<String>, b: &BTreeSet<String>) -> BTreeSet<String> {
    a.difference(b).cloned().collect()
}

fn join_set(set: &BTreeSet<String>, max: usize) -> String {
    set.iter().take(max).cloned().collect::<Vec<_>>().join(",")
}

/// Jaccard similarity as bps (empty∩empty → 10000).
fn jaccard_bps(a: &BTreeSet<String>, b: &BTreeSet<String>) -> u16 {
    if a.is_empty() && b.is_empty() {
        return 10_000;
    }
    let inter = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 {
        return 10_000;
    }
    u16::try_from((inter * 10_000) / union).unwrap_or(0)
}

/// Cosine similarity on count bags as bps (both empty → 10000).
fn cosine_bps(a: &BTreeMap<String, u32>, b: &BTreeMap<String, u32>) -> u16 {
    if a.is_empty() && b.is_empty() {
        return 10_000;
    }
    let mut dot: u64 = 0;
    let mut na: u64 = 0;
    let mut nb: u64 = 0;
    for (k, &va) in a {
        let va = u64::from(va);
        na += va * va;
        if let Some(&vb) = b.get(k) {
            dot += va * u64::from(vb);
        }
    }
    for &vb in b.values() {
        let vb = u64::from(vb);
        nb += vb * vb;
    }
    if na == 0 || nb == 0 {
        return 0;
    }
    // cos = dot / sqrt(na*nb); compute with integer scale via isqrt.
    let denom = isqrt(na * nb);
    if denom == 0 {
        return 0;
    }
    u16::try_from(((dot * 10_000) / denom).min(10_000)).unwrap_or(0)
}

fn isqrt(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    let mut x = n;
    let mut y = x.div_ceil(2);
    while y < x {
        x = y;
        y = x.saturating_add(n / x) / 2;
    }
    x
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use crate::fingerprint_source;

    const AGENT_A: &str = r"
import os
import json

def build_pages(ctx):
    data = {'ok': True}
    return json.dumps(data)

class Runner:
    def run(self, path):
        with open(path) as f:
            return f.read()
";

    const AGENT_A_RENAMED: &str = r"
import os
import json

def build_pages(ctx):
    data = {'ok': True}
    return json.dumps(data)

class Runner:
    def run(self, path):
        with open(path) as f:
            return f.read()
";

    const UNRELATED: &str = r"
import math

def train_epoch(model, batch):
    loss = 0.0
    for x, y in batch:
        pred = model(x)
        loss += (pred - y) ** 2
    return loss / max(len(batch), 1)

async def evaluate(loader):
    total = 0
    async for item in loader:
        total += math.ceil(item)
    return total
";

    #[test]
    fn identical_sources_max_bps() {
        let a = fingerprint_source(AGENT_A).unwrap();
        let b = fingerprint_source(AGENT_A).unwrap();
        assert_eq!(similarity_bps(&a, &b), 10_000);
    }

    #[test]
    fn structurally_same_high_bps() {
        let a = fingerprint_source(AGENT_A).unwrap();
        let b = fingerprint_source(AGENT_A_RENAMED).unwrap();
        assert_eq!(similarity_bps(&a, &b), 10_000);
    }

    #[test]
    fn unrelated_low_bps() {
        let a = fingerprint_source(AGENT_A).unwrap();
        let b = fingerprint_source(UNRELATED).unwrap();
        let bps = similarity_bps(&a, &b);
        assert!(bps < 5_000, "expected low overlap, got {bps}");
    }

    #[test]
    fn top_k_orders_nearest_first() {
        let q = fingerprint_source(AGENT_A).unwrap();
        let near = fingerprint_source(AGENT_A).unwrap();
        let far = fingerprint_source(UNRELATED).unwrap();
        let corpus = vec![("far".into(), far), ("near".into(), near)];
        let ranked = top_k_nearest(&q, &corpus, 2);
        assert_eq!(ranked[0].id, "near");
        assert_eq!(ranked[0].similarity_bps, 10_000);
        assert!(ranked[1].similarity_bps < ranked[0].similarity_bps);
    }
}
