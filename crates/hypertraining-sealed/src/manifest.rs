//! `sealed_surface.v1` manifest types.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Manifest kind string.
pub const MANIFEST_KIND: &str = "sealed_surface.v1";

/// Frozen Megatron-LM commit pin (brief).
pub const DEFAULT_MLM_COMMIT: &str = "cd4afffa648426a959dc7cb1e24b5ce7d0c3ff54";

/// Frozen `TransformerEngine` version pin (brief).
pub const DEFAULT_TE_VERSION: &str = "2.18.0+e7c550c5";

/// Dataset identity pin inside the sealed surface.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DatasetPin {
    /// Corpus name (e.g. `fineweb-edu`).
    pub corpus: String,
    /// Corpus revision / snapshot sha.
    pub revision: String,
    /// Fixed shuffle / order seed.
    pub order_seed: u64,
}

/// Segment measurement pin.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentPin {
    /// Tokens per segment (`T_seg`).
    pub tokens: u64,
    /// Global batch size.
    pub gbs: u64,
    /// Sequence length.
    pub seq_len: u64,
}

/// Normative sealed-surface manifest (`sealed_surface.v1`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SealedSurfaceV1 {
    /// Must be [`MANIFEST_KIND`].
    pub kind: String,
    /// Base Megatron-Bridge (or umbrella) commit the fork is measured against.
    pub base_commit: String,
    /// Megatron-LM commit pin.
    pub mlm_commit: String,
    /// `TransformerEngine` version pin.
    pub te_version: String,
    /// Path → SHA-256 hex for every denylisted file that must stay byte-identical.
    pub denylist_hashes: BTreeMap<String, String>,
    /// `path:symbol` → simplified AST hash hex.
    pub sealed_symbols: BTreeMap<String, String>,
    /// Dataset pin.
    pub dataset_pin: DatasetPin,
    /// Segment pin.
    pub segment: SegmentPin,
}

impl SealedSurfaceV1 {
    /// Build a manifest with frozen TE/MLM pins and empty hash maps (fill before admit).
    #[must_use]
    pub fn with_pins(
        base_commit: impl Into<String>,
        dataset: DatasetPin,
        segment: SegmentPin,
    ) -> Self {
        Self {
            kind: MANIFEST_KIND.to_owned(),
            base_commit: base_commit.into(),
            mlm_commit: DEFAULT_MLM_COMMIT.to_owned(),
            te_version: DEFAULT_TE_VERSION.to_owned(),
            denylist_hashes: BTreeMap::new(),
            sealed_symbols: BTreeMap::new(),
            dataset_pin: dataset,
            segment,
        }
    }
}
