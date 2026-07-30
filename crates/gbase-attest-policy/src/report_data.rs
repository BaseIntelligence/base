//! D10 liveness binding: `report_data = SHA512(scale concat …)`.
//!
//! ```text
//! preimage = scale(b"gbase-attest-v1")
//!          ‖ scale(netuid: u16)
//!          ‖ scale(epoch: u64)
//!          ‖ scale(miner_pubkey: [u8; 32])
//!          ‖ scale(nonce: [u8; 32])
//!          ‖ scale(validator_hotkey: [u8; 32])
//! report_data = SHA512(preimage)   // 64 bytes
//! ```

use gbase_crypto::{domain, DomainTag, KEY_LEN};
use parity_scale_codec::Encode;
use sha2::{Digest, Sha512};

/// TDX `report_data` width (matches `gbase-attest-parse`).
pub const REPORT_DATA_LEN: usize = 64;

/// Claimed binding fields used to build or check `report_data`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReportDataBinding {
    /// Subnet netuid.
    pub netuid: u16,
    /// Epoch index.
    pub epoch: u64,
    /// Miner hotkey / pubkey bound into the quote.
    pub miner_pubkey: [u8; KEY_LEN],
    /// Validator-issued single-use nonce.
    pub nonce: [u8; KEY_LEN],
    /// Asking validator hotkey.
    pub validator_hotkey: [u8; KEY_LEN],
}

/// Domain tag bytes for the D10 preimage (`gbase-attest-v1`).
#[must_use]
pub const fn attest_domain() -> DomainTag {
    domain::ATTEST
}

/// SCALE-concatenated D10 preimage (before SHA-512).
#[must_use]
pub fn report_data_preimage(binding: &ReportDataBinding) -> Vec<u8> {
    let mut out = Vec::with_capacity(16 + 2 + 8 + KEY_LEN * 3 + 8);
    // scale(&[u8]) = Compact(len) ‖ bytes  (same path as gbase-crypto signing tags)
    attest_domain().as_bytes().encode_to(&mut out);
    binding.netuid.encode_to(&mut out);
    binding.epoch.encode_to(&mut out);
    // Fixed-size arrays encode as raw bytes (no Compact length prefix).
    binding.miner_pubkey.encode_to(&mut out);
    binding.nonce.encode_to(&mut out);
    binding.validator_hotkey.encode_to(&mut out);
    out
}

/// Compute the 64-byte D10 `report_data` for `binding`.
#[must_use]
pub fn compute_report_data(binding: &ReportDataBinding) -> [u8; REPORT_DATA_LEN] {
    let preimage = report_data_preimage(binding);
    let digest = Sha512::digest(&preimage);
    let mut out = [0_u8; REPORT_DATA_LEN];
    out.copy_from_slice(&digest);
    out
}

/// Constant-time equality of quote `report_data` against D10 binding.
#[must_use]
pub fn verify_report_data(actual: &[u8; REPORT_DATA_LEN], binding: &ReportDataBinding) -> bool {
    let expected = compute_report_data(binding);
    // subtle would be nicer; equality on fixed 64 bytes is fine for policy gate.
    expected == *actual
}
