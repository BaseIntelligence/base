//! L3 — telemetry fingerprint compare (brief §12.3 level 3).

/// Validator-side dynamic counters reused from guard-3 style telemetry.
#[derive(Debug, Clone, PartialEq)]
pub struct TelemetryFingerprint {
    /// Instruction mix histogram (fixed bins; fixture-sized).
    pub insn_mix: Vec<f64>,
    /// DRAM bytes transferred.
    pub dram_bytes: u64,
    /// Tensor-core op count.
    pub tensor_ops: u64,
    /// Occupancy in `[0.0, 1.0]`.
    pub occupancy: f64,
    /// Ordered kernel launch name hashes (or fixture tags).
    pub launch_seq: Vec<u64>,
}

impl TelemetryFingerprint {
    /// Construct a fingerprint; empty `insn_mix` is allowed (scores via other fields).
    #[must_use]
    pub fn new(
        insn_mix: Vec<f64>,
        dram_bytes: u64,
        tensor_ops: u64,
        occupancy: f64,
        launch_seq: Vec<u64>,
    ) -> Self {
        Self {
            insn_mix,
            dram_bytes,
            tensor_ops,
            occupancy,
            launch_seq,
        }
    }
}

/// Similarity in `[0.0, 1.0]` between two telemetry fingerprints.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn l3_telemetry_similarity(a: &TelemetryFingerprint, b: &TelemetryFingerprint) -> f64 {
    let mix = cosine_sim(&a.insn_mix, &b.insn_mix);
    let dram = ratio_sim(a.dram_bytes as f64, b.dram_bytes as f64);
    let tensor = ratio_sim(a.tensor_ops as f64, b.tensor_ops as f64);
    let occ = 1.0 - (a.occupancy - b.occupancy).abs().min(1.0);
    let launch = seq_sim(&a.launch_seq, &b.launch_seq);
    // Equal weights; all components in [0,1].
    (mix + dram + tensor + occ + launch) / 5.0
}

fn ratio_sim(a: f64, b: f64) -> f64 {
    let max = a.max(b);
    if max == 0.0 {
        return 1.0;
    }
    a.min(b) / max
}

fn cosine_sim(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let n = a.len().max(b.len());
    if n == 0 {
        return 1.0;
    }
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..n {
        let va = a.get(i).copied().unwrap_or(0.0);
        let vb = b.get(i).copied().unwrap_or(0.0);
        dot += va * vb;
        na += va * va;
        nb += vb * vb;
    }
    if na == 0.0 && nb == 0.0 {
        return 1.0;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())).clamp(0.0, 1.0)
}

#[allow(clippy::cast_precision_loss)]
fn seq_sim(a: &[u64], b: &[u64]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    if a == b {
        return 1.0;
    }
    let n = a.len().max(b.len());
    let mut match_n = 0_usize;
    for i in 0..n {
        if a.get(i) == b.get(i) {
            match_n += 1;
        }
    }
    match_n as f64 / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_telemetry_scores_one() {
        let t = TelemetryFingerprint::new(vec![1.0, 2.0, 3.0], 1000, 50, 0.8, vec![1, 2, 3]);
        assert!((l3_telemetry_similarity(&t, &t) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn divergent_telemetry_scores_lower() {
        let a = TelemetryFingerprint::new(vec![1.0, 0.0, 0.0], 1000, 50, 0.9, vec![1, 2]);
        let b = TelemetryFingerprint::new(vec![0.0, 0.0, 1.0], 10, 1, 0.1, vec![9, 8]);
        let s = l3_telemetry_similarity(&a, &b);
        assert!(s < 0.5, "got {s}");
    }
}
