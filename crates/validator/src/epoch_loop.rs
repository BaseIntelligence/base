//! Periodic gateway latest → bundle fetch → compare → Match log → CRV4 submit (F3).

use std::sync::{Arc, Mutex};
use std::time::Duration;

use bundle::LocalTrustRoot;
use chain::ChainClient;
use dissent::{SubmissionIntent, SubmissionSource};
use submit::{submit_intent, DrandClient, ReadyDrand, SubmitConfig, SubmitOutcome, SystemClock};
use tracing::{info, warn};

use crate::coordination::{CoordinationClient, CoordinationError};
use crate::epoch::epoch_from_block;
use crate::lkg::SealedBundleLkg;
use crate::recompute::{compare_bundle_bytes, no_submission_from_fetch, ComparisonOutcome};

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

/// On-chain submit parameters for the coordination loop (task 33 wiring).
///
/// When present, a [`ComparisonOutcome::Match`] builds a [`SubmissionIntent`] and
/// calls [`submit_intent`] (CRV4 timelocked when commit-reveal is enabled).
#[derive(Debug, Clone)]
pub struct CoordinationSubmitConfig {
    /// Subnet netuid.
    pub netuid: u16,
    /// Validator hotkey public key bytes (payload field).
    pub hotkey: Vec<u8>,
    /// Weights `version_key`.
    pub version_key: u64,
    /// Epoch length in blocks (drand / submit deadline = current epoch end).
    pub epoch_length: u64,
}

/// In-memory per-epoch submit dedupe for the coordination loop.
///
/// Records epochs that successfully submitted (or intentionally skipped drand).
/// Failures leave the epoch unmarked so the next tick can retry.
#[derive(Debug, Default)]
pub struct EpochSubmitDedupe {
    last_ok: Mutex<Option<u64>>,
}

impl EpochSubmitDedupe {
    /// Empty dedupe state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether this epoch already completed a submit attempt successfully.
    #[must_use]
    pub fn already_submitted(&self, epoch: u64) -> bool {
        self.last_ok.lock().ok().is_some_and(|g| *g == Some(epoch))
    }

    /// Mark epoch as submitted (or intentionally skipped).
    pub fn mark_submitted(&self, epoch: u64) {
        if let Ok(mut g) = self.last_ok.lock() {
            *g = Some(epoch);
        }
    }
}

/// Build a recompute [`SubmissionIntent`] from a Match outcome.
#[must_use]
pub fn intent_from_match(outcome: &ComparisonOutcome) -> Option<SubmissionIntent> {
    match outcome {
        ComparisonOutcome::Match {
            epoch,
            local_vector,
            ..
        } => Some(SubmissionIntent {
            epoch: *epoch,
            vector: local_vector.clone(),
            source: SubmissionSource::Recompute,
        }),
        _ => None,
    }
}

/// Submit Match vector via [`submit_intent`] with per-epoch dedupe.
///
/// No-op when `submit` is `None`, outcome is not Match, or epoch already submitted.
/// Submit errors are logged and do **not** mark the epoch (retry next tick).
pub fn maybe_submit_match<C, D>(
    outcome: &ComparisonOutcome,
    chain: &C,
    drand: &D,
    submit: Option<&CoordinationSubmitConfig>,
    dedupe: &EpochSubmitDedupe,
) where
    C: ChainClient + ?Sized,
    D: DrandClient + ?Sized,
{
    let Some(cfg) = submit else {
        return;
    };
    let Some(intent) = intent_from_match(outcome) else {
        return;
    };
    if dedupe.already_submitted(intent.epoch) {
        info!(
            event = "validator_submit_deduped",
            epoch = intent.epoch,
            "submit skipped: epoch already submitted"
        );
        return;
    }

    let tip = match chain.current_block() {
        Ok(t) => t,
        Err(e) => {
            warn!(error = %e, epoch = intent.epoch, "submit skipped: tip read failed");
            return;
        }
    };
    let deadline = match epoch_from_block(tip, cfg.epoch_length) {
        Ok(snap) => snap.epoch_end_block,
        Err(e) => {
            warn!(error = %e, epoch = intent.epoch, "submit skipped: epoch clock failed");
            return;
        }
    };

    let submit_cfg = SubmitConfig {
        netuid: cfg.netuid,
        mecid: submit::MECID_MAIN,
        version_key: cfg.version_key,
        hotkey: cfg.hotkey.clone(),
        epoch_deadline_block: deadline,
        // Live submits are dispatch-confirmed via `LastUpdate`; a rate-limited
        // commit fails ~100 blocks early, so an immediate in-call burst only
        // spams doomed extrinsics. The coordination tick is the retry loop.
        max_rate_limit_retries: 0,
        max_drand_retries: 32,
    };

    info!(
        event = "validator_submit_attempt",
        epoch = intent.epoch,
        netuid = cfg.netuid,
        vector_len = intent.vector.len(),
        "Match → submit_intent"
    );

    match submit_intent(&intent, chain, drand, &SystemClock, &submit_cfg) {
        Ok(SubmitOutcome::SubmittedTimelocked {
            reveal_round,
            mecid,
        }) => {
            info!(
                event = "validator_submit_timelocked",
                epoch = intent.epoch,
                reveal_round,
                mecid,
                "submit_timelocked ok"
            );
            dedupe.mark_submitted(intent.epoch);
        }
        Ok(SubmitOutcome::SubmittedSetWeights) => {
            info!(
                event = "validator_submit_set_weights",
                epoch = intent.epoch,
                "set_weights ok (CR disabled)"
            );
            dedupe.mark_submitted(intent.epoch);
        }
        Ok(SubmitOutcome::SkippedDrand { epoch }) => {
            warn!(
                event = "validator_submit_skipped_drand",
                epoch,
                "drand unavailable through deadline; epoch skipped (no set_weights downgrade)"
            );
            // Intentional skip — do not retry forever within the same epoch.
            dedupe.mark_submitted(epoch);
        }
        Err(e) => {
            let msg = e.to_string();
            // Permanent gaps (wrong CR version) — do not hammer the RPC every
            // coordination tick. Transient tlock/RPC failures retry next tick.
            let permanent =
                msg.contains("commit_reveal_version") || msg.contains("WrongCommitRevealVersion");
            if permanent {
                warn!(
                    event = "validator_submit_error",
                    epoch = intent.epoch,
                    error = %e,
                    "submit_intent failed permanently for epoch (deduped)"
                );
                dedupe.mark_submitted(intent.epoch);
            } else {
                warn!(
                    event = "validator_submit_error",
                    epoch = intent.epoch,
                    error = %e,
                    "submit_intent failed (will retry next tick)"
                );
            }
        }
    }
}

/// One coordination compare cycle: latest → bundle → `compare_bundle` → optional submit.
///
/// Soft-ok when gateway missing, latest 404 (legacy), or unsealed burn fallback.
///
/// # Errors
///
/// Transport / unexpected HTTP (not 404) from coordination client.
pub async fn coordination_compare_once<C: ChainClient>(
    client: &CoordinationClient,
    chain: &C,
    trust: &LocalTrustRoot,
    submit: Option<&CoordinationSubmitConfig>,
    dedupe: &EpochSubmitDedupe,
) -> Result<Option<ComparisonOutcome>, CoordinationError> {
    coordination_compare_once_with_drand(
        client,
        chain,
        trust,
        submit,
        dedupe,
        &ReadyDrand,
        &SealedBundleLkg::disabled(),
    )
    .await
}

/// Same as [`coordination_compare_once`] with an injectable [`DrandClient`].
pub async fn coordination_compare_once_with_drand<C, D>(
    client: &CoordinationClient,
    chain: &C,
    trust: &LocalTrustRoot,
    submit: Option<&CoordinationSubmitConfig>,
    dedupe: &EpochSubmitDedupe,
    drand: &D,
    lkg: &SealedBundleLkg,
) -> Result<Option<ComparisonOutcome>, CoordinationError>
where
    C: ChainClient,
    D: DrandClient + ?Sized,
{
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
    // Wrong-network guard: a gateway serving another netuid's seal (staging vs
    // prod VPC mix-up) must fail loudly here, not as a bundle verify error.
    if let (Some(gateway_netuid), Some(cfg)) = (latest.netuid, submit) {
        if gateway_netuid != cfg.netuid {
            warn!(
                gateway_netuid,
                validator_netuid = cfg.netuid,
                "coordination compare skipped: gateway serves a different netuid"
            );
            return Ok(None);
        }
    }
    let Some(epoch) = latest.epoch.filter(|_| latest.is_sealed_bundle()) else {
        return Ok(apply_unsealed_latest(
            chain, trust, submit, dedupe, drand, lkg,
        ));
    };
    // Pressure / verify: sealed vector must stay near the live chain epoch.
    // A stuck real-seal (D24 incomplete) leaves `/v1/weights/latest` on an old
    // chain-scale bundle; Match can still succeed while emission is stale.
    if let Some(cfg) = submit {
        if let Ok(chain_epoch) = chain.subnet_epoch_index(cfg.netuid) {
            let lag = chain_epoch.saturating_sub(epoch);
            if lag > 1 {
                warn!(
                    event = "validator_seal_lag",
                    sealed_epoch = epoch,
                    chain_epoch,
                    lag_epochs = lag,
                    metagraph_block = ?latest.metagraph_block,
                    "pressure verify: sealed weights lag chain epoch; check design/prism emit + base-real-seal"
                );
            }
        }
        if let (Ok(tip), Some(mg_block)) = (chain.current_block(), latest.metagraph_block) {
            let block_lag = tip.saturating_sub(mg_block);
            // Public Finney RPC prunes ~256 blocks; beyond that Match cannot
            // re-fetch the seal's metagraph on a cold RPC (ops red line).
            if block_lag > 256 {
                warn!(
                    event = "validator_seal_metagraph_stale",
                    sealed_epoch = epoch,
                    metagraph_block = mg_block,
                    tip_block = tip,
                    block_lag,
                    "pressure verify: seal metagraph_block outside ~256-block prune window"
                );
            }
        }
    }
    let outcome = match client.fetch_bundle(epoch).await {
        Ok(bytes) => {
            let outcome = compare_bundle_bytes(&bytes, chain, trust);
            if matches!(outcome, ComparisonOutcome::Match { .. }) {
                lkg.save(&bytes);
            }
            outcome
        }
        Err(e) => no_submission_from_fetch(e),
    };
    record_compare_outcome(&outcome, chain, drand, submit, dedupe);
    Ok(Some(outcome))
}

/// Gateway serving the fail-closed burn (`sealed: false`): log, then Match the
/// last independently verified seal so set-weights survives a master bounce.
fn apply_unsealed_latest<C, D>(
    chain: &C,
    trust: &LocalTrustRoot,
    submit: Option<&CoordinationSubmitConfig>,
    dedupe: &EpochSubmitDedupe,
    drand: &D,
    lkg: &SealedBundleLkg,
) -> Option<ComparisonOutcome>
where
    C: ChainClient,
    D: DrandClient + ?Sized,
{
    warn!(
        event = "validator_latest_unsealed",
        "gateway /v1/weights/latest is unsealed burn; trying persisted sealed bundle"
    );
    let Some(bytes) = lkg.load() else {
        warn!(
            event = "validator_lkg_missing",
            "no persisted sealed bundle; cannot submit weights until gateway reseals"
        );
        return None;
    };
    let outcome = compare_bundle_bytes(&bytes, chain, trust);
    record_compare_outcome(&outcome, chain, drand, submit, dedupe);
    Some(outcome)
}

fn record_compare_outcome<C, D>(
    outcome: &ComparisonOutcome,
    chain: &C,
    drand: &D,
    submit: Option<&CoordinationSubmitConfig>,
    dedupe: &EpochSubmitDedupe,
) where
    C: ChainClient,
    D: DrandClient + ?Sized,
{
    match outcome {
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
            maybe_submit_match(outcome, chain, drand, submit, dedupe);
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
}

/// Spawn a background loop that periodically runs [`coordination_compare_once`].
///
/// When `submit` is `Some`, Match outcomes call [`submit_intent`] once per epoch.
pub fn spawn_coordination_loop<C>(
    client: Arc<CoordinationClient>,
    chain: Arc<C>,
    trust: LocalTrustRoot,
    interval: Duration,
    submit: Option<CoordinationSubmitConfig>,
) -> tokio::task::JoinHandle<()>
where
    C: ChainClient + Send + Sync + 'static,
{
    let dedupe = Arc::new(EpochSubmitDedupe::new());
    let lkg = SealedBundleLkg::from_env();
    tokio::spawn(async move {
        // Immediate first attempt (covers seal-before-start and seal-soon-after).
        if let Err(e) = coordination_compare_once_with_drand(
            client.as_ref(),
            chain.as_ref(),
            &trust,
            submit.as_ref(),
            dedupe.as_ref(),
            &ReadyDrand,
            &lkg,
        )
        .await
        {
            warn!(error = %e, "coordination compare tick failed (non-fatal)");
        }
        let mut ticker = tokio::time::interval(interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            ticker.tick().await;
            if let Err(e) = coordination_compare_once_with_drand(
                client.as_ref(),
                chain.as_ref(),
                &trust,
                submit.as_ref(),
                dedupe.as_ref(),
                &ReadyDrand,
                &lkg,
            )
            .await
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
    use chain::{ChainCall, FakeChain, FakeChainConfig};
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

    #[test]
    fn dedupe_marks_and_checks_epoch() {
        let d = EpochSubmitDedupe::new();
        assert!(!d.already_submitted(7));
        d.mark_submitted(7);
        assert!(d.already_submitted(7));
        assert!(!d.already_submitted(8));
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
        let dedupe = EpochSubmitDedupe::new();
        let out = coordination_compare_once(&client, &chain, &trust, None, &dedupe)
            .await
            .expect("soft");
        assert!(out.is_none());
    }

    #[tokio::test]
    async fn tick_burn_fallback_is_soft_none() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/weights/latest"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "epoch": null,
                "merkle_root": "",
                "sealed": false,
                "uids": [0],
                "weights": [1.0]
            })))
            .mount(&server)
            .await;
        let client = CoordinationClient::new(Some(server.uri())).unwrap();
        let chain = FakeChain::with_defaults();
        let trust = LocalTrustRoot {
            challenges: ChallengesBody::default(),
            measurements_digest: measurements_digest(&MeasurementsBody::default()),
        };
        let dedupe = EpochSubmitDedupe::new();
        let out = coordination_compare_once(&client, &chain, &trust, None, &dedupe)
            .await
            .expect("soft");
        assert!(out.is_none());
    }

    #[tokio::test]
    async fn tick_wrong_gateway_netuid_skips_before_bundle_fetch() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/weights/latest"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "epoch": 77,
                "netuid": 541,
                "merkle_root": hex::encode([0xabu8; 32]),
                "final_vector": [[0, 65535]],
                "sealed": true,
            })))
            .mount(&server)
            .await;
        let client = CoordinationClient::new(Some(server.uri())).unwrap();
        let chain = FakeChain::with_defaults();
        let trust = LocalTrustRoot {
            challenges: ChallengesBody::default(),
            measurements_digest: measurements_digest(&MeasurementsBody::default()),
        };
        let dedupe = EpochSubmitDedupe::new();
        let submit = CoordinationSubmitConfig {
            netuid: 100,
            hotkey: vec![0xBBu8; 32],
            version_key: 3,
            epoch_length: 360,
        };
        let out = coordination_compare_once(&client, &chain, &trust, Some(&submit), &dedupe)
            .await
            .expect("soft");
        assert!(out.is_none(), "wrong-netuid gateway must skip the tick");
        let received = server.received_requests().await.expect("log");
        assert_eq!(
            received.len(),
            1,
            "must not fetch the bundle from a wrong-netuid gateway"
        );
    }

    /// Seal a one-miner prism epoch and mount gateway latest + bundle routes.
    #[allow(clippy::too_many_lines)]
    async fn sealed_match_fixture(
        epoch: u64,
    ) -> (
        CoordinationClient,
        FakeChain,
        LocalTrustRoot,
        [u8; 32],
        Vec<(u16, u16)>,
        Vec<u8>,
    ) {
        let csk = sk(1);
        let gsk = sk(2);
        let cid = b"prism";
        let miner = [0xA1u8; 32];
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
            commit_reveal_enabled: true,
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
                "sealed": true,
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path(format!("/v1/bundle/{epoch}")))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "application/octet-stream")
                    .set_body_bytes(bytes.clone()),
            )
            .mount(&server)
            .await;
        let client = CoordinationClient::new(Some(server.uri())).unwrap();
        let trust = LocalTrustRoot {
            challenges: ch_body,
            measurements_digest: mdigest,
        };
        (
            client,
            chain,
            trust,
            bundle.body.merkle_root,
            bundle.body.final_vector,
            bytes,
        )
    }

    #[tokio::test]
    #[allow(clippy::too_many_lines)]
    async fn tick_sealed_latest_yields_match() {
        let epoch = 77u64;
        let (client, chain, trust, merkle_root, _, _) = sealed_match_fixture(epoch).await;
        let dedupe = EpochSubmitDedupe::new();
        let out = coordination_compare_once(&client, &chain, &trust, None, &dedupe)
            .await
            .expect("ok")
            .expect("some");
        match out {
            ComparisonOutcome::Match {
                epoch: e,
                merkle_root: root,
                ..
            } => {
                assert_eq!(e, epoch);
                assert_eq!(root, merkle_root);
                let line = format_match_line(e, &root, &[0u8; 32], 1, 1);
                assert!(line.contains("Match epoch=77"));
            }
            other => panic!("expected Match, got {other:?}"),
        }
        assert!(
            chain.call_log().is_empty(),
            "no submit when submit cfg is None"
        );
    }

    #[test]
    fn maybe_submit_match_timelocked_once_per_epoch() {
        let epoch = 88u64;
        let outcome = ComparisonOutcome::Match {
            epoch,
            local_vector: vec![(0, 65535)],
            gateway_vector: vec![(0, 65535)],
            vector_hash: [0x11; 32],
            merkle_root: [0x22; 32],
        };
        let chain = FakeChain::new(FakeChainConfig {
            current_block: 500,
            commit_reveal_enabled: true,
            ..FakeChainConfig::default()
        });
        let submit = CoordinationSubmitConfig {
            netuid: 1,
            hotkey: vec![0xBBu8; 32],
            version_key: 3,
            epoch_length: 360,
        };
        let dedupe = EpochSubmitDedupe::new();

        maybe_submit_match(&outcome, &chain, &ReadyDrand, Some(&submit), &dedupe);
        let log = chain.call_log();
        assert_eq!(log.len(), 1, "exactly one extrinsic after first Match");
        match &log[0] {
            ChainCall::SubmitTimelocked(sub) => {
                assert_eq!(sub.mecid, 0);
                assert_eq!(sub.payload.hotkey, vec![0xBBu8; 32]);
                assert_eq!(sub.payload.version_key, 3);
                assert_eq!(sub.payload.uids, vec![0]);
                assert_eq!(sub.payload.values, vec![65535]);
                assert!(
                    chain.set_weights_log().is_empty(),
                    "CR on → never set_weights"
                );
            }
            other @ ChainCall::SetWeights(_) => {
                panic!("expected SubmitTimelocked, got {other:?}")
            }
        }
        assert!(dedupe.already_submitted(epoch));

        // Second attempt: dedupe must suppress another extrinsic.
        maybe_submit_match(&outcome, &chain, &ReadyDrand, Some(&submit), &dedupe);
        assert_eq!(
            chain.call_log().len(),
            1,
            "dedupe: still one submit after second Match"
        );
    }

    #[test]
    fn maybe_submit_match_noop_without_config() {
        let outcome = ComparisonOutcome::Match {
            epoch: 1,
            local_vector: vec![(0, 1)],
            gateway_vector: vec![(0, 1)],
            vector_hash: [0; 32],
            merkle_root: [0; 32],
        };
        let chain = FakeChain::with_defaults();
        let dedupe = EpochSubmitDedupe::new();
        maybe_submit_match(&outcome, &chain, &ReadyDrand, None, &dedupe);
        assert!(chain.call_log().is_empty());
    }

    #[tokio::test]
    async fn tick_pressure_verify_allows_match_when_seal_lags_chain() {
        // Same metagraph as the sealed fixture, but chain epoch/tip far ahead —
        // pressure-verify warns (validator_seal_lag) yet Match must still proceed.
        let epoch = 77u64;
        let (client, _chain, trust, merkle_root, _, _) = sealed_match_fixture(epoch).await;
        let miner = [0xA1u8; 32];
        let chain = FakeChain::new(FakeChainConfig {
            current_block: 10_000,
            subnet_epoch_index: epoch + 11,
            hotkeys: vec![miner.to_vec()],
            owner_hotkey: miner.to_vec(),
            commit_reveal_enabled: true,
            last_epoch_block: 500,
            ..FakeChainConfig::default()
        });
        let dedupe = EpochSubmitDedupe::new();
        let submit = CoordinationSubmitConfig {
            netuid: 1,
            hotkey: vec![0xBBu8; 32],
            version_key: 3,
            epoch_length: 360,
        };
        let out = coordination_compare_once(&client, &chain, &trust, Some(&submit), &dedupe)
            .await
            .expect("ok")
            .expect("some");
        match out {
            ComparisonOutcome::Match {
                epoch: e,
                merkle_root: root,
                ..
            } => {
                assert_eq!(e, epoch);
                assert_eq!(root, merkle_root);
            }
            other => panic!("expected Match despite seal lag, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn tick_burn_after_restart_matches_lkg_and_submits() {
        let epoch = 77u64;
        let (sealed_client, chain, trust, merkle_root, _, _) = sealed_match_fixture(epoch).await;
        let dir =
            std::env::temp_dir().join(format!("base-lkg-tick-{}-{}", std::process::id(), epoch));
        let lkg = SealedBundleLkg::at(dir.join("last-sealed.bundle"));
        let submit = CoordinationSubmitConfig {
            netuid: 1,
            hotkey: vec![0xBBu8; 32],
            version_key: 3,
            epoch_length: 360,
        };
        let first_dedupe = EpochSubmitDedupe::new();
        let first = coordination_compare_once_with_drand(
            &sealed_client,
            &chain,
            &trust,
            Some(&submit),
            &first_dedupe,
            &ReadyDrand,
            &lkg,
        )
        .await
        .expect("ok")
        .expect("some");
        assert!(matches!(first, ComparisonOutcome::Match { epoch: e, .. } if e == epoch));
        assert!(lkg.load().is_some(), "Match must persist SCALE bytes");
        assert_eq!(chain.call_log().len(), 1, "first sealed tick submits");

        let burn = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/weights/latest"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "epoch": null,
                "merkle_root": "",
                "sealed": false,
                "uids": [0],
                "weights": [1.0]
            })))
            .mount(&burn)
            .await;
        let burn_client = CoordinationClient::new(Some(burn.uri())).unwrap();
        // Process restart drops in-memory submit dedupe.
        let restart_dedupe = EpochSubmitDedupe::new();
        let out = coordination_compare_once_with_drand(
            &burn_client,
            &chain,
            &trust,
            Some(&submit),
            &restart_dedupe,
            &ReadyDrand,
            &lkg,
        )
        .await
        .expect("ok")
        .expect("lkg match");
        match out {
            ComparisonOutcome::Match {
                epoch: e,
                merkle_root: root,
                ..
            } => {
                assert_eq!(e, epoch);
                assert_eq!(root, merkle_root);
            }
            other => panic!("expected LKG Match, got {other:?}"),
        }
        assert_eq!(
            chain.call_log().len(),
            2,
            "restart + burn latest must resubmit last verified weights"
        );
        let _ = std::fs::remove_dir_all(dir);
    }
}
