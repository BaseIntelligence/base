//! Periodic gateway latest → bundle fetch → compare → Match log (F3 continuous path).

use std::sync::Arc;
use std::time::Duration;

use bundle::LocalTrustRoot;
use chain::ChainClient;
use tracing::{info, warn};

use crate::coordination::{CoordinationClient, CoordinationError};
use crate::recompute::{fetch_and_compare, ComparisonOutcome};

/// Format Match line identical to `full_local_e2e` `VALIDATOR_LOG` shape.
#[must_use]
pub fn format_match_line(
    epoch: u64,
    merkle_root: &[u8; 32],
    vector_hash: &[u8; 32],
    local_len: usize,
    gateway_len: usize,
) -> String {
    format!(
        "Match epoch={epoch} merkle_root={} vector_hash={} local_vector_len={local_len} gateway_vector_len={gateway_len}",
        hex::encode(merkle_root),
        hex::encode(vector_hash),
    )
}

/// One coordination compare cycle: latest → bundle → `compare_bundle`.
///
/// Soft-ok when gateway missing or latest 404 (no sealed bundle yet).
///
/// # Errors
///
/// Transport / unexpected HTTP (not 404) from coordination client.
pub async fn coordination_compare_once<C: ChainClient>(
    client: &CoordinationClient,
    chain: &C,
    trust: &LocalTrustRoot,
) -> Result<Option<ComparisonOutcome>, CoordinationError> {
    if !client.has_gateway() {
        return Ok(None);
    }
    let latest = match client.fetch_weights_latest().await {
        Ok(v) => v,
        Err(CoordinationError::HttpStatus { status: 404, .. }) => {
            return Ok(None);
        }
        Err(e) => return Err(e),
    };
    let outcome = fetch_and_compare(client, latest.epoch, chain, trust).await;
    match &outcome {
        ComparisonOutcome::Match {
            epoch,
            merkle_root,
            vector_hash,
            local_vector,
            gateway_vector,
            ..
        } => {
            let line = format_match_line(
                *epoch,
                merkle_root,
                vector_hash,
                local_vector.len(),
                gateway_vector.len(),
            );
            info!(
                event = "validator_match",
                epoch = *epoch,
                merkle_root = %hex::encode(merkle_root),
                vector_hash = %hex::encode(vector_hash),
                local_vector_len = local_vector.len(),
                gateway_vector_len = gateway_vector.len(),
                "{line}"
            );
        }
        ComparisonOutcome::VectorMismatch { epoch, .. } => {
            warn!(epoch, "coordination compare VectorMismatch");
        }
        ComparisonOutcome::InputInvalid { error } => {
            warn!(error = %error, "coordination compare InputInvalid");
        }
        ComparisonOutcome::NoSubmission { reason } => {
            warn!(?reason, "coordination compare NoSubmission");
        }
    }
    Ok(Some(outcome))
}

/// Spawn a background loop that periodically runs [`coordination_compare_once`].
pub fn spawn_coordination_loop<C>(
    client: Arc<CoordinationClient>,
    chain: Arc<C>,
    trust: LocalTrustRoot,
    interval: Duration,
) -> tokio::task::JoinHandle<()>
where
    C: ChainClient + Send + Sync + 'static,
{
    tokio::spawn(async move {
        // Immediate first attempt (covers seal-before-start and seal-soon-after).
        if let Err(e) = coordination_compare_once(client.as_ref(), chain.as_ref(), &trust).await {
            warn!(error = %e, "coordination compare tick failed (non-fatal)");
        }
        let mut ticker = tokio::time::interval(interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            ticker.tick().await;
            if let Err(e) = coordination_compare_once(client.as_ref(), chain.as_ref(), &trust).await
            {
                warn!(error = %e, "coordination compare tick failed (non-fatal)");
            }
        }
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::coordination::CoordinationClient;
    use bundle::{make_signed_leaf, LocalTrustRoot, ScoreOrAbsence};
    use chain::{FakeChain, FakeChainConfig};
    use crypto::{secret_from_bytes, KEY_LEN};
    use gateway::{
        seal_epoch, BundleStore, ChallengeEntry, ChallengesBody, MemoryBundleStore,
        MemoryRawWeightStore, ParticipantPolicy, RawWeightRow, RawWeightStore, SealParams,
        BPS_DENOM,
    };
    use sha2::{Digest, Sha256};
    use trustroot::{measurements_digest, MeasurementsBody};
    use uuid::Uuid;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn sk(tag: u8) -> [u8; KEY_LEN] {
        let dig = Sha256::digest([0x5A, tag, 0xA5, tag]);
        let mut seed = [0u8; KEY_LEN];
        seed.copy_from_slice(&dig);
        seed
    }

    fn pk_of(secret: &[u8; KEY_LEN]) -> [u8; KEY_LEN] {
        secret_from_bytes(secret)
            .expect("sk")
            .to_public()
            .to_bytes()
    }

    #[test]
    fn format_match_line_shape() {
        let line = format_match_line(33, &[0xab; 32], &[0xcd; 32], 3, 3);
        assert!(line.starts_with("Match epoch=33 merkle_root="));
        assert!(line.contains("vector_hash="));
        assert!(line.contains("local_vector_len=3"));
        assert!(line.contains("gateway_vector_len=3"));
    }

    #[tokio::test]
    async fn tick_404_is_soft_none() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/weights/latest"))
            .respond_with(ResponseTemplate::new(404).set_body_json(serde_json::json!({
                "error": "no sealed bundle"
            })))
            .mount(&server)
            .await;
        let client = CoordinationClient::new(Some(server.uri())).unwrap();
        let chain = FakeChain::with_defaults();
        let trust = LocalTrustRoot {
            challenges: ChallengesBody::default(),
            measurements_digest: measurements_digest(&MeasurementsBody::default()),
        };
        let out = coordination_compare_once(&client, &chain, &trust)
            .await
            .expect("soft");
        assert!(out.is_none());
    }

    #[tokio::test]
    #[allow(clippy::too_many_lines)]
    async fn tick_sealed_latest_yields_match() {
        let csk = sk(1);
        let gsk = sk(2);
        let cid = b"prism";
        let miner = [0xA1u8; 32];
        let epoch = 77u64;
        let block_b = 500u64;
        let ch_body = ChallengesBody {
            challenges: vec![ChallengeEntry {
                id: cid.to_vec(),
                public_key: pk_of(&csk),
                emission_share_bps: BPS_DENOM,
                policy: ParticipantPolicy::AllMetagraphHotkeys,
            }],
        };
        let mdigest = measurements_digest(&MeasurementsBody::default());
        let weights = MemoryRawWeightStore::new();
        let leaf = make_signed_leaf(
            &csk,
            cid,
            miner,
            epoch,
            ScoreOrAbsence::Score { value: 100 },
        )
        .expect("leaf");
        let payload = bundle::raw_weight_payload(
            &leaf.challenge_id,
            &leaf.miner_hotkey,
            leaf.epoch,
            &leaf.score_or_absence,
        );
        let digest = Sha256::digest(&payload);
        let mut payload_digest = [0u8; 32];
        payload_digest.copy_from_slice(&digest);
        weights
            .insert(RawWeightRow {
                id: Uuid::new_v4(),
                challenge_id: "prism".into(),
                epoch,
                miner_hotkey: hex::encode(miner),
                kind: "score".into(),
                score: Some(100),
                absence_reason: None,
                payload,
                payload_digest,
                challenge_sig: leaf.challenge_sig.to_vec(),
            })
            .expect("append");
        let bundles = MemoryBundleStore::new();
        let chain = FakeChain::new(FakeChainConfig {
            current_block: block_b.max(10),
            hotkeys: vec![miner.to_vec()],
            owner_hotkey: miner.to_vec(),
            ..FakeChainConfig::default()
        });
        let bundle = seal_epoch(
            &chain,
            &ch_body,
            &weights,
            &bundles,
            &SealParams {
                epoch,
                netuid: 1,
                block_b,
                gateway_secret: gsk,
                measurements_digest: mdigest,
            },
        )
        .expect("seal");
        let bytes = bundles.get_by_epoch(epoch).expect("stored");
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/weights/latest"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "epoch": epoch,
                "merkle_root": hex::encode(bundle.body.merkle_root),
                "final_vector": bundle.body.final_vector,
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path(format!("/v1/bundle/{epoch}")))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/octet-stream")
                    .set_body_bytes(bytes),
            )
            .mount(&server)
            .await;
        let client = CoordinationClient::new(Some(server.uri())).unwrap();
        let trust = LocalTrustRoot {
            challenges: ch_body,
            measurements_digest: mdigest,
        };
        let out = coordination_compare_once(&client, &chain, &trust)
            .await
            .expect("ok")
            .expect("some");
        match out {
            ComparisonOutcome::Match {
                epoch: e,
                merkle_root,
                ..
            } => {
                assert_eq!(e, epoch);
                assert_eq!(merkle_root, bundle.body.merkle_root);
                let line = format_match_line(e, &merkle_root, &[0u8; 32], 1, 1);
                assert!(line.contains("Match epoch=77"));
            }
            other => panic!("expected Match, got {other:?}"),
        }
    }
}
