//! Owner-signed local trust roots for challenges and measurements.
//!
//! # Design
//!
//! - Load **only** from local filesystem paths (D18). This crate has **no** HTTP client.
//! - Bodies are signed as sr25519 under [`gbase_crypto::domain::TRUST_ROOT`] over
//!   `scale(version, introduced_epoch, scale(body))`.
//! - Detached signatures live beside TOML as `*.toml.sig` (hex or raw 64 bytes).
//! - `measurements` empty ⇒ every quote rejected (fail closed; populated in task 35).
//! - D21 dual-accept: when `v(n+1)` is introduced at epoch `T`, both `v(n)` and
//!   `v(n+1)` are active for `rotation_epochs` epochs, then only `v(n+1)`.
//!
//! # Ceremony
//!
//! See `config/CEREMONY.md` and the `gbase-trustroot` CLI (`keygen` / `sign` / `verify`).

#![forbid(unsafe_code)]

mod error;
mod hexutil;
mod load;
mod sign;
mod types;

pub use error::TrustRootError;
pub use hexutil::{decode_hex_array, encode_hex};
pub use load::{
    filter_active, kind_name, load_challenges_dir, load_challenges_file,
    load_challenges_file_with_sig, load_config_dir, load_measurements_dir, load_measurements_file,
    load_measurements_file_with_sig, load_owner_public_key, sig_path_for, ActiveRoots,
    VerifiedRoot,
};
pub use sign::{
    encode_challenges_body, encode_measurements_body, measurements_digest, parse_signature_bytes,
    sign_trust_root, sign_trust_root_raw, signing_payload_bytes, verify_trust_root,
    verify_trust_root_raw,
};
pub use types::{
    challenge_to_toml, ChallengeEntry, ChallengeToml, ChallengesBody, ChallengesToml,
    MeasurementEntry, MeasurementToml, MeasurementsBody, MeasurementsToml, ParticipantPolicy,
    PolicyTable, PolicyToml, TrustRootKind, BPS_DENOM, COMPOSE_HASH_LEN, REGISTER_LEN,
};
