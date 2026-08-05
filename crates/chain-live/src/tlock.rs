//! Drand Quicknet timelock encryption for CRV4 commit blobs.
//!
//! Port of `bittensor_drand::encrypt_and_compress` / `get_encrypted_commit_v2`
//! ciphertext construction. Uses the same git-pinned `tle` crate as subtensor
//! (`ideal-lab5/timelock` @ `5416406…`) so Finney can decrypt on reveal.

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use chain::DRAND_PUBLIC_KEY;
use rand_core::{OsRng, RngCore};
use sha2::{Digest, Sha256};
use tle::{
    curves::drand::TinyBLS381, ibe::fullident::Identity,
    stream_ciphers::AESGCMStreamCipherProvider, tlock::tle,
};
use w3f_bls::EngineBLS;

/// Encrypt SCALE-encoded [`chain::WeightsTlockPayload`] bytes for `reveal_round`.
///
/// Returns the compressed TLE ciphertext that belongs in
/// `commit_timelocked_mechanism_weights.commit`.
///
/// # Errors
///
/// Public-key decode/deserialize failure or TLE encrypt / serialize failure.
pub fn encrypt_commit(serialized_payload: &[u8], reveal_round: u64) -> Result<Vec<u8>, String> {
    let pub_key_bytes =
        hex::decode(DRAND_PUBLIC_KEY).map_err(|e| format!("drand public key hex: {e}"))?;
    let pub_key =
        <TinyBLS381 as EngineBLS>::PublicKeyGroup::deserialize_compressed(pub_key_bytes.as_slice())
            .map_err(|e| format!("drand public key deserialize: {e}"))?;

    let round_hash = Sha256::digest(reveal_round.to_be_bytes()).to_vec();
    let identity = Identity::new(b"", vec![round_hash]);

    let mut esk = [0u8; 32];
    OsRng.fill_bytes(&mut esk);
    let ct = tle::<TinyBLS381, AESGCMStreamCipherProvider, OsRng>(
        pub_key,
        esk,
        serialized_payload,
        identity,
        OsRng,
    )
    .map_err(|e| format!("tle encrypt: {e:?}"))?;

    let mut ct_bytes = Vec::new();
    ct.serialize_compressed(&mut ct_bytes)
        .map_err(|e| format!("tle ciphertext serialize: {e}"))?;
    if ct_bytes.is_empty() {
        return Err("tle ciphertext serialized empty".into());
    }
    Ok(ct_bytes)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use ark_serialize::CanonicalDeserialize;
    use chain::WeightsTlockPayload;
    use parity_scale_codec::Encode;
    use tle::tlock::{tld, TLECiphertext};

    fn sample_payload() -> WeightsTlockPayload {
        WeightsTlockPayload {
            hotkey: vec![7u8; 32],
            uids: vec![0],
            values: vec![u16::MAX],
            version_key: 0,
        }
    }

    #[test]
    fn encrypt_commit_produces_nonempty_ciphertext() {
        let plain = sample_payload().encode();
        let ct = encrypt_commit(&plain, 17_200_000).expect("encrypt");
        assert!(ct.len() > 64, "ciphertext too short: {}", ct.len());
        // Random esk → two encryptions of the same plaintext differ.
        let ct2 = encrypt_commit(&plain, 17_200_000).expect("encrypt again");
        assert_ne!(ct, ct2);
    }

    /// Live Quicknet signature fetch — run with `--ignored` when online.
    #[test]
    #[ignore = "requires Drand HTTP"]
    fn encrypt_commit_round_trip_with_drand_signature() {
        let reveal_round = 17_200_000_u64;
        let plain = sample_payload().encode();
        let ct_bytes = encrypt_commit(&plain, reveal_round).expect("encrypt");

        let sig_hex = fetch_drand_signature(reveal_round).expect("drand signature");
        let sig_bytes = hex::decode(&sig_hex).expect("sig hex");
        let ciphertext = TLECiphertext::<TinyBLS381>::deserialize_compressed(ct_bytes.as_slice())
            .expect("deserialize ct");
        let sign =
            <TinyBLS381 as EngineBLS>::SignatureGroup::deserialize_compressed(sig_bytes.as_slice())
                .expect("deserialize sig");
        let decrypted =
            tld::<TinyBLS381, AESGCMStreamCipherProvider>(ciphertext, sign).expect("tld decrypt");
        assert_eq!(decrypted, plain);
    }

    /// Fetch hex BLS signature for a Quicknet round (same endpoints as SDK).
    fn fetch_drand_signature(round: u64) -> Result<String, String> {
        let chain_hash = "52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971";
        let endpoints = [
            "https://api.drand.sh",
            "https://api2.drand.sh",
            "https://drand.cloudflare.com",
        ];
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(15))
            .build()
            .map_err(|e| e.to_string())?;
        let mut last = String::new();
        for ep in endpoints {
            let url = format!("{ep}/{chain_hash}/public/{round}");
            match client.get(&url).send() {
                Ok(resp) => match resp.json::<serde_json::Value>() {
                    Ok(v) => {
                        if let Some(s) = v.get("signature").and_then(|x| x.as_str()) {
                            return Ok(s.to_owned());
                        }
                        last = format!("missing signature in {url}");
                    }
                    Err(e) => last = format!("json {url}: {e}"),
                },
                Err(e) => last = format!("http {url}: {e}"),
            }
        }
        Err(last)
    }
}
