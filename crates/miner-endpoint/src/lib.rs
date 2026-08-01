//! Miner-signed announcement of the public CVM base URL
//! (`AGENT_CHALLENGE.md` §9.3 step 5).
//!
//! `agent-challenge` dispatches Harbor packs over HTTPS to the miner's Phala
//! CVM (§1). It therefore needs a hotkey → URL map, and there was none:
//! `gateway_registry::Backend` is keyed by `challenge_id` and has no hotkey
//! column, so it can route to an operator backend but cannot find a miner.
//!
//! This crate is the producer side of that map. A miner POSTs its base URL to
//! the gateway signed by its hotkey; the gateway checks the epoch against the
//! chain, checks the hotkey is actually registered, verifies the signature, and
//! stores the row. `db::miner_endpoints` is the consumer read.
//!
//! # Trust
//!
//! The signature proves *who* announced, never *what* is safe to dial. The URL
//! itself is attacker-chosen input to an outbound request, so it is filtered by
//! [`validate_base_url`], whose module documents the SSRF model and the DNS
//! rebinding gap that only the dispatcher can close.

//! # Features
//!
//! The `server` feature (default) adds the axum router and its Postgres write
//! path. The miner CLI turns it off: it only needs to build and sign a body,
//! and has no business linking sqlx.

#![forbid(unsafe_code)]

#[cfg(feature = "server")]
mod server;
mod url_guard;

use crypto::{domain, sign_raw, verify_raw, CryptoError, KEY_LEN, SIGNATURE_LEN};
use parity_scale_codec::{Decode, Encode};

#[cfg(feature = "server")]
pub use server::{
    miner_endpoint_router, AnnounceRequest, AnnounceResponse, MinerEndpointState, SharedChain,
};
pub use url_guard::{
    is_forbidden_ip, validate_base_url, UrlRejection, MAX_BASE_URL_LEN, MIN_UNPRIVILEGED_PORT,
};

/// Route the miner CLI POSTs to.
pub const ENDPOINT_ROUTE: &str = "/v1/miners/endpoint";

/// Body a miner signs to announce where its CVM can be reached.
///
/// Field order is the SCALE encoding order and is consensus-visible: the
/// gateway rebuilds this struct from the request and verifies against it, so
/// reordering a field silently invalidates every miner's signature.
///
/// `epoch` is in the body (not just the envelope) so an announcement cannot be
/// replayed into a later epoch to keep a dead URL alive.
#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode)]
pub struct MinerEndpointBodyV1 {
    /// Subnet netuid.
    pub netuid: u16,
    /// Announcing miner hotkey (sr25519 public key).
    pub miner_hotkey: [u8; KEY_LEN],
    /// Public base URL bytes, exactly as announced.
    pub base_url: Vec<u8>,
    /// Chain epoch the announcement is valid for.
    pub epoch: u64,
}

impl MinerEndpointBodyV1 {
    /// SCALE preimage the signature covers.
    #[must_use]
    pub fn payload(&self) -> Vec<u8> {
        self.encode()
    }
}

/// Sign an announcement with the miner's hotkey mini-secret.
///
/// # Errors
///
/// [`CryptoError::InvalidSecretKey`] when the mini-secret is malformed.
pub fn sign_endpoint(
    secret: &[u8; KEY_LEN],
    body: &MinerEndpointBodyV1,
) -> Result<[u8; SIGNATURE_LEN], CryptoError> {
    sign_raw(secret, domain::MINER_ENDPOINT, &body.payload())
}

/// Verify an announcement against the hotkey named inside it.
///
/// The key comes from the body rather than the caller, so a signature can never
/// be checked against a hotkey other than the one it claims to speak for.
///
/// # Errors
///
/// [`CryptoError`] on a malformed key/signature or a failed verification.
pub fn verify_endpoint(body: &MinerEndpointBodyV1, signature: &[u8]) -> Result<(), CryptoError> {
    verify_raw(
        &body.miner_hotkey,
        domain::MINER_ENDPOINT,
        &body.payload(),
        signature,
    )
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use super::*;
    use crypto::signing_preimage;

    fn body(epoch: u64, base_url: &str) -> MinerEndpointBodyV1 {
        MinerEndpointBodyV1 {
            netuid: 7,
            miner_hotkey: [0x5a; KEY_LEN],
            base_url: base_url.as_bytes().to_vec(),
            epoch,
        }
    }

    /// Field order is the wire contract; pin it against the manual encoding.
    #[test]
    fn scale_field_order_is_netuid_hotkey_url_epoch() {
        let b = body(9, "https://x.example.com");
        let mut expected = Vec::new();
        expected.extend_from_slice(&b.netuid.to_le_bytes());
        expected.extend_from_slice(&b.miner_hotkey);
        b.base_url.encode_to(&mut expected);
        expected.extend_from_slice(&b.epoch.to_le_bytes());
        assert_eq!(b.payload(), expected);
        assert_eq!(
            MinerEndpointBodyV1::decode(&mut b.payload().as_slice()).expect("round trip"),
            b
        );
    }

    /// The signature must cover the domain-tagged preimage, not the bare body.
    #[test]
    fn signature_covers_the_domain_separated_preimage() {
        let sk = [0x11u8; KEY_LEN];
        let pk = crypto::public_key_from_mini_secret(&sk).expect("public");
        let mut b = body(9, "https://x.example.com");
        b.miner_hotkey = pk;
        let sig = sign_endpoint(&sk, &b).expect("sign");
        verify_endpoint(&b, &sig).expect("verify");

        let preimage = signing_preimage(domain::MINER_ENDPOINT, &b.payload());
        assert!(preimage.starts_with(&{
            let mut v = Vec::new();
            domain::MINER_ENDPOINT.as_bytes().encode_to(&mut v);
            v
        }));

        // A different domain over the same bytes must not verify.
        let other = crypto::sign_raw(&sk, domain::RAW_WEIGHT, &b.payload()).expect("sign");
        verify_endpoint(&b, &other).expect_err("cross-domain replay must fail");
    }

    /// Changing any field invalidates the signature.
    #[test]
    fn every_field_is_bound_by_the_signature() {
        let sk = [0x22u8; KEY_LEN];
        let pk = crypto::public_key_from_mini_secret(&sk).expect("public");
        let mut b = body(9, "https://x.example.com");
        b.miner_hotkey = pk;
        let sig = sign_endpoint(&sk, &b).expect("sign");

        let mut tampered = b.clone();
        tampered.base_url = b"https://evil.example.com".to_vec();
        verify_endpoint(&tampered, &sig).expect_err("url is bound");

        let mut tampered = b.clone();
        tampered.epoch = 10;
        verify_endpoint(&tampered, &sig).expect_err("epoch is bound");

        let mut tampered = b;
        tampered.netuid = 8;
        verify_endpoint(&tampered, &sig).expect_err("netuid is bound");
    }
}
