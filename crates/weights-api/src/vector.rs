//! Projection of a sealed [`EpochBundleV1`] into [`WeightsLatestResponse`].

use bundle::{leaf_preimage, EpochBundleV1, LeafV1};
use keystore::{ss58_encode, BITTENSOR_SS58_PREFIX};
use sha2::{Digest, Sha256};
use trustroot::BPS_DENOM;
use uuid::Builder;

use crate::{
    ChallengeWeightsResult, FloatMap, MetagraphIdentity, SealRecord, SourceOutcome, SourceRef,
    WeightsLatestResponse, BURN_POLICY_VERSION, BURN_UID, EMISSION_POLICY_VERSION,
    FRESHNESS_SECONDS, MAPPING_POLICY_VERSION, OUTCOME_ACCEPTED, OUTCOME_MISSING, PROTOCOL_VERSION,
};

/// Project a sealed bundle plus its seal record into the validator response.
#[must_use]
pub fn build_latest(bundle: &EpochBundleV1, seal: SealRecord) -> WeightsLatestResponse {
    let body = &bundle.body;
    let vector_digest = hex::encode(Sha256::digest(bundle::encode(body)));
    // The reference service serves the aggregator's raw doubles. Rounding them into the
    // chain's u16 space and back is a different number (`19660/65535 != 0.3`), so we
    // recompute the floats from the signed body instead of projecting `final_vector`.
    // They are a pure function of the body, so nothing needs persisting.
    let floats = bundle::python_weights(body).ok().map(|v| v.floats);
    let uids: Vec<u16> = floats.as_ref().map_or_else(
        || body.final_vector.iter().map(|(uid, _)| *uid).collect(),
        |f| f.uids.clone(),
    );
    let weights = floats
        .as_ref()
        .map_or_else(|| normalise(&body.final_vector), |f| f.weights.clone());
    let hotkey_weights = floats.as_ref().map_or_else(
        || hotkey_weights_from_uid_map(body, &uids, &weights),
        |f| hotkey_weights_from_floats(&f.hotkey_weights),
    );
    let computed_at = iso8601_micros(seal.sealed_at_micros);
    let expires_at = iso8601_micros(
        seal.sealed_at_micros
            .saturating_add(FRESHNESS_SECONDS * 1_000_000),
    );
    let sources = sources(body);

    WeightsLatestResponse {
        protocol_version: PROTOCOL_VERSION.to_owned(),
        vector_id: Some(vector_id(&vector_digest)),
        vector_digest: Some(vector_digest),
        epoch: Some(body.epoch),
        revision: seal.revision,
        netuid: body.netuid,
        chain_endpoint: std::env::var(config::keys::CHAIN_ENDPOINT).unwrap_or_default(),
        chain_domain_bytes: Some(chain_domain_bytes(body.netuid, &uids, &weights)),
        hotkey_weights,
        uids,
        weights,
        computed_at: computed_at.clone(),
        expires_at,
        source_challenges: sources.iter().map(Source::to_challenge).collect(),
        source_snapshots: sources.iter().filter_map(Source::to_ref).collect(),
        source_outcomes: sources.iter().map(Source::to_outcome).collect(),
        emission_policy_version: Some(EMISSION_POLICY_VERSION.to_owned()),
        emission_shares: FloatMap::new(
            sources
                .iter()
                .map(|s| (s.slug.clone(), s.share))
                .collect::<Vec<_>>(),
        ),
        burn_policy_version: Some(BURN_POLICY_VERSION.to_owned()),
        mapping_policy_version: Some(MAPPING_POLICY_VERSION.to_owned()),
        metagraph_identity: MetagraphIdentity {
            hash: Some(hex::encode(body.metagraph_root)),
            block: Some(body.block_b),
            uid_count: body.uid_map.len(),
            burn_uid: BURN_UID,
        },
        metagraph_hash: Some(hex::encode(body.metagraph_root)),
        metagraph_block: Some(body.block_b),
        burn_outcome: Some(body.final_vector.iter().any(|(uid, _)| *uid == BURN_UID)),
        // The metagraph is read at seal time from `block_b`, so the seal instant
        // is also the instant the snapshot was taken.
        metagraph_updated_at: computed_at,
        merkle_root: hex::encode(body.merkle_root),
        final_vector: body.final_vector.clone(),
    }
}

/// Per-challenge sealed-source view assembled once and projected three ways.
struct Source {
    slug: String,
    share: f64,
    snapshot: Option<(String, String)>,
}

impl Source {
    fn outcome(&self) -> &'static str {
        if self.snapshot.is_some() {
            OUTCOME_ACCEPTED
        } else {
            OUTCOME_MISSING
        }
    }

    fn to_challenge(&self) -> ChallengeWeightsResult {
        let ok = self.snapshot.is_some();
        ChallengeWeightsResult {
            slug: self.slug.clone(),
            emission_percent: self.share * 100.0,
            // Matches the reference service, which never republishes miner
            // weights on this endpoint.
            weights: FloatMap::default(),
            ok,
            error: if ok {
                None
            } else {
                Some(OUTCOME_MISSING.to_owned())
            },
        }
    }

    fn to_ref(&self) -> Option<SourceRef> {
        let (snapshot_id, payload_digest) = self.snapshot.clone()?;
        Some(SourceRef {
            challenge_slug: self.slug.clone(),
            snapshot_id,
            payload_digest,
            outcome: self.outcome().to_owned(),
        })
    }

    fn to_outcome(&self) -> SourceOutcome {
        let (snapshot_id, payload_digest) = match self.snapshot.clone() {
            Some((id, digest)) => (Some(id), Some(digest)),
            None => (None, None),
        };
        SourceOutcome {
            challenge_slug: self.slug.clone(),
            outcome: self.outcome().to_owned(),
            reason_code: self.outcome().to_owned(),
            snapshot_id,
            payload_digest,
            // No analogue: sealed leaves are immutable, so there is no
            // revisioned snapshot row to report.
            revision: None,
        }
    }
}

/// One [`Source`] per expected challenge, in the bundle's normative share order.
fn sources(body: &bundle::EpochBundleBodyV1) -> Vec<Source> {
    body.emission_shares
        .iter()
        .map(|(challenge_id, bps)| {
            let leaves: Vec<&LeafV1> = body
                .leaves
                .iter()
                .filter(|leaf| leaf.challenge_id == *challenge_id)
                .collect();
            Source {
                slug: String::from_utf8_lossy(challenge_id).into_owned(),
                share: f64::from(*bps) / f64::from(BPS_DENOM),
                snapshot: snapshot_digest(&leaves).map(|digest| (snapshot_id(&digest), digest)),
            }
        })
        .collect()
}

/// SHA-256 over the sealed leaf preimages of one challenge.
///
/// Our durable unit is the per-miner leaf, not a per-challenge snapshot row, so
/// the challenge snapshot digest commits to exactly the leaves that were sealed.
fn snapshot_digest(leaves: &[&LeafV1]) -> Option<String> {
    if leaves.is_empty() {
        return None;
    }
    let mut hasher = Sha256::new();
    for leaf in leaves {
        hasher.update(leaf_preimage(leaf));
    }
    Some(hex::encode(hasher.finalize()))
}

/// UUID derived from a hex digest so snapshot identities are reproducible.
fn snapshot_id(digest_hex: &str) -> String {
    uuid_from_digest(digest_hex)
}

/// UUID derived from `vector_digest`; stable for the life of the sealed vector.
fn vector_id(digest_hex: &str) -> String {
    uuid_from_digest(digest_hex)
}

fn uuid_from_digest(digest_hex: &str) -> String {
    let mut bytes = [0u8; 16];
    let digest = Sha256::digest(digest_hex.as_bytes());
    bytes.copy_from_slice(&digest[..16]);
    Builder::from_sha1_bytes(bytes).into_uuid().to_string()
}

/// Python's `kept_hotkeys` re-keyed from hex to SS58, order preserved.
///
/// The aggregator already excluded the burn uid and any unmapped hotkey, so every
/// surviving entry is a real miner.
fn hotkey_weights_from_floats(kept: &[(String, f64)]) -> FloatMap {
    let entries = kept
        .iter()
        .filter_map(|(hex_key, weight)| {
            let bytes: [u8; 32] = hex::decode(hex_key).ok()?.try_into().ok()?;
            Some((ss58_encode(&bytes, BITTENSOR_SS58_PREFIX), *weight))
        })
        .collect::<Vec<_>>();
    FloatMap::new(entries)
}

/// Chain-scale weights as fractions summing to 1.0 (0.0 when the vector is empty).
///
/// Fallback only: reachable if the body's leaves cannot be re-aggregated, which a
/// sealed body cannot be, since the seal itself required the aggregation to succeed.
fn normalise(final_vector: &[(u16, u16)]) -> Vec<f64> {
    let total: f64 = final_vector.iter().map(|(_, w)| f64::from(*w)).sum();
    if total <= 0.0 {
        return vec![0.0; final_vector.len()];
    }
    final_vector
        .iter()
        .map(|(_, w)| f64::from(*w) / total)
        .collect()
}

/// Fallback pairing of the stored vector against `uid_map` (see [`normalise`]).
fn hotkey_weights_from_uid_map(
    body: &bundle::EpochBundleBodyV1,
    uids: &[u16],
    weights: &[f64],
) -> FloatMap {
    let entries = uids
        .iter()
        .zip(weights)
        // The burn uid is a policy sink, not a registered miner, so it carries
        // no hotkey entry (`burn-uid0.v1`).
        .filter(|(uid, _)| **uid != BURN_UID)
        .filter_map(|(uid, weight)| {
            body.uid_map
                .iter()
                .find(|(_, mapped)| mapped == uid)
                .map(|(hotkey, _)| (ss58_encode(hotkey, BITTENSOR_SS58_PREFIX), *weight))
        })
        .collect::<Vec<_>>();
    FloatMap::new(entries)
}

/// Canonical `{"netuid","uids","weights"}` JSON (sorted keys, no whitespace).
fn chain_domain_bytes(netuid: u16, uids: &[u16], weights: &[f64]) -> String {
    serde_json::json!({
        "netuid": netuid,
        "uids": uids,
        "weights": weights,
    })
    .to_string()
}

/// RFC 3339 with exactly six fractional digits and a `Z` suffix, matching the
/// reference service's Pydantic datetime rendering.
///
/// Hand-rolled rather than pulling a date library: `deny.toml` bans a direct
/// `chrono` dependency, and the input is always a post-epoch UTC instant.
fn iso8601_micros(micros: u64) -> String {
    let secs = micros / 1_000_000;
    let frac = micros % 1_000_000;
    let days = secs / 86_400;
    let tod = secs % 86_400;
    let (year, month, day) = civil_from_days(days);
    let (hh, mm, ss) = (tod / 3600, (tod % 3600) / 60, tod % 60);
    format!("{year:04}-{month:02}-{day:02}T{hh:02}:{mm:02}:{ss:02}.{frac:06}Z")
}

/// Days since the Unix epoch to a proleptic Gregorian date (Howard Hinnant's
/// `civil_from_days`, shifted to an era starting 0000-03-01).
fn civil_from_days(days: u64) -> (u64, u64, u64) {
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z % 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + u64::from(month <= 2);
    (year, month, day)
}
