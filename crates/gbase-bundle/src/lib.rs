//! Epoch bundle SCALE types, signing, and verification (`BUNDLE_SPEC` `protocol_version` = 1).
//!
//! Field order of [`EpochBundleBodyV1`] is normative (§4.1). Re-encoding a decoded
//! bundle MUST yield identical bytes (§15).
//!
//! Consensus lint (D7/D8): no `HashMap`, `f32`/`f64`, or `wrapping_*`.

#![forbid(unsafe_code)]

mod error;
mod merkle_util;
mod sign;
mod types;
mod verify;

pub use error::BundleError;
pub use merkle_util::{
    compute_merkle_root, compute_metagraph_root, expected_participants, leaf_preimage, leaf_sort_key,
    metagraph_rows_from_chain, sort_leaves, uid_map_from_rows,
};
pub use sign::{
    finalize_body_merkle, make_signed_leaf, raw_weight_payload, sign_bundle, sign_leaf_challenge,
};
pub use types::{
    field_order_canary_body, field_order_canary_prefix, EpochBundleBodyV1, EpochBundleV1, LeafV1,
    LocalTrustRoot, MetagraphRow, NoScoreReasonCode, RawWeightBodyV1, ScoreOrAbsence,
    VerifiedBundle, ALGORITHM_VERSION, BODY_FIELD_ORDER, MAX_CHALLENGE_ID_LEN, PROTOCOL_VERSION,
};
pub use verify::{final_vectors_equal, verify_bundle};

use parity_scale_codec::{Decode, Encode};

use crate::error::BundleError as BE;
use crate::types::{EpochBundleBodyV1 as Body, EpochBundleV1 as Bundle};
use crate::verify::verify_bundle as verify_fn;
use gbase_chain::ChainClient;

/// SCALE-encode any bundle type.
#[must_use]
pub fn encode<T: Encode>(value: &T) -> Vec<u8> {
    value.encode()
}

/// SCALE-decode `EpochBundleV1`.
///
/// # Errors
///
/// [`BundleError::Codec`] on decode failure.
pub fn decode_bundle(bytes: &[u8]) -> Result<Bundle, BE> {
    Bundle::decode(&mut &bytes[..]).map_err(|e| BE::Codec(e.to_string()))
}

/// SCALE-decode `EpochBundleBodyV1`.
///
/// # Errors
///
/// [`BundleError::Codec`] on decode failure.
pub fn decode_body(bytes: &[u8]) -> Result<Body, BE> {
    Body::decode(&mut &bytes[..]).map_err(|e| BE::Codec(e.to_string()))
}

impl Bundle {
    /// Encode this bundle to SCALE bytes.
    #[must_use]
    pub fn encode_bytes(&self) -> Vec<u8> {
        self.encode()
    }

    /// Decode from SCALE bytes.
    ///
    /// # Errors
    ///
    /// Codec errors.
    pub fn decode_bytes(bytes: &[u8]) -> Result<Self, BE> {
        decode_bundle(bytes)
    }

    /// Verify this bundle against chain + local trust root (`BUNDLE_SPEC` §14).
    ///
    /// # Errors
    ///
    /// One of the rejection branches in [`BundleError`].
    pub fn verify<C: ChainClient>(
        &self,
        chain: &C,
        trust: &LocalTrustRoot,
    ) -> Result<VerifiedBundle, BE> {
        verify_fn(self, chain, trust)
    }
}

/// Full emission mass in basis points.
pub const BPS_TOTAL: u16 = gbase_trustroot::BPS_DENOM;

/// Doc-test: field order list must match §4.1.
///
/// ```
/// use gbase_bundle::{BODY_FIELD_ORDER, field_order_canary_body, field_order_canary_prefix};
/// assert_eq!(BODY_FIELD_ORDER[0], "protocol_version");
/// assert_eq!(BODY_FIELD_ORDER[3], "block_B");
/// assert_eq!(BODY_FIELD_ORDER[7], "emission_shares");
/// assert_eq!(BODY_FIELD_ORDER[13], "gateway_hotkey");
/// let enc = parity_scale_codec::Encode::encode(&field_order_canary_body());
/// assert_eq!(&enc[..10], &field_order_canary_prefix());
/// assert_eq!(&enc[..2], &[1u8, 0]);
/// ```
pub mod field_order_doc {}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::pedantic)]
mod tests;
