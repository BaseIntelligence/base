//! Promotion K indexed by binary similarity (brief §12.6).

/// Binary similarity above this threshold → automatic reject, no measure.
pub const BINARY_SIM_REJECT: f64 = 0.85;

/// Base promotion K when binary similarity is low.
pub const K_BASE: u32 = 5;
/// Escalated K for mid similarity band.
pub const K_MID: u32 = 7;
/// High-similarity band K (still measured).
pub const K_HIGH: u32 = 11;

/// Outcome of the K-by-similarity table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KBySim {
    /// Measure with this promotion K.
    Measure {
        /// Promotion segment count K.
        k: u32,
    },
    /// Similarity `> 0.85` — do not measure.
    AutoReject {
        /// Observed binary similarity.
        binary_similarity: f64,
    },
}

/// Map binary similarity in `[0, 1]` to promotion K or auto-reject.
///
/// | Binary similarity | K |
/// |-------------------|---|
/// | `< 0.30` | 5 |
/// | `0.30 ..= 0.60` | 7 |
/// | `> 0.60 ..= 0.85` | 11 |
/// | `> 0.85` | auto-reject |
#[must_use]
pub fn k_for_binary_similarity(sim: f64) -> KBySim {
    let s = sim.clamp(0.0, 1.0);
    if s > BINARY_SIM_REJECT {
        return KBySim::AutoReject {
            binary_similarity: s,
        };
    }
    let k = if s < 0.30 {
        K_BASE
    } else if s <= 0.60 {
        K_MID
    } else {
        // 0.60 < s <= 0.85
        K_HIGH
    };
    KBySim::Measure { k }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_bands() {
        assert_eq!(k_for_binary_similarity(0.0), KBySim::Measure { k: 5 });
        assert_eq!(k_for_binary_similarity(0.29), KBySim::Measure { k: 5 });
        assert_eq!(k_for_binary_similarity(0.30), KBySim::Measure { k: 7 });
        assert_eq!(k_for_binary_similarity(0.60), KBySim::Measure { k: 7 });
        assert_eq!(k_for_binary_similarity(0.61), KBySim::Measure { k: 11 });
        assert_eq!(k_for_binary_similarity(0.85), KBySim::Measure { k: 11 });
        assert!(matches!(
            k_for_binary_similarity(0.850_000_1),
            KBySim::AutoReject { .. }
        ));
        assert!(matches!(
            k_for_binary_similarity(1.0),
            KBySim::AutoReject { .. }
        ));
    }
}
