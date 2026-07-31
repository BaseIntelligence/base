//! Todo 33: full local challenge → grade → |E| leaves → gateway seal → validator Match.
//!
//! In-process e2e (no staging/testnet). Optional Harbor live grade when
//! `GBASE_E2E_HARBOR=1` and socket-proxy + pack image are available.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{BTreeMap, BTreeSet};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use agent_challenge::{
    emit_signed_leaf_set, hex32, intake_and_grade, public_key_from_secret, run_epoch_dispatch,
    score_map_covering_expected, submit_signed_leaf_set, ActiveSignerRegistry, EpochDispatchClient,
    EpochDispatchConfig, ExpectedParticipant, ExpectedReceiptBind, ExpectedSet, GatewayClient,
    GatewayClientConfig, HarborVerifier, HarborVerifierConfig, MinerEpochOutcome,
    NoScoreReasonCode, Reward, RunnerCapacity, ScoreOrAbsence, SubmitError, SubmitOutcome,
    Verifier, CHALLENGE_ID, SCORE_MAX, SCORING_VERSION,
};
use agent_dispatch::{
    patch_sha256, sign_work_receipt, TaskDescriptorV1, TaskResultV1, TaskStatusV1,
    WorkReceiptBodyV1, DISPATCH_PROTOCOL,
};
use agent_pack::{load_pack, HarborPack, HeldOutMaterials, PackId};
use bundle::LocalTrustRoot;
use chain::{FakeChain, FakeChainConfig};
use crypto::KEY_LEN;
use gateway::{
    build_app_with_bundles, seal_epoch, ChallengeEntry, ChallengesBody, MemoryBundleStore,
    MemoryRawWeightStore, ParticipantPolicy, RawWeightStore, Registry, RegistryConfig, SealParams,
    SharedBundleStore, TlsConfig, BPS_DENOM,
};
use schnorrkel::MiniSecretKey;
use telemetry::init_metrics;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use trustroot::{measurements_digest, MeasurementsBody};
use validator::{
    apply_three_outcome_policy, compare_bundle, recompute_view_from_comparison, ComparisonOutcome,
    CrossCheckOutcome, DissentSigner, EpochDecision,
};

const EPOCH: u64 = 33;
const BLOCK_B: u64 = 330;
const SOLVER: [u8; KEY_LEN] = [0xA1; KEY_LEN];
const ZERO: [u8; KEY_LEN] = [0xB2; KEY_LEN];
const UNREACH: [u8; KEY_LEN] = [0xC3; KEY_LEN];
const CVM_SK: [u8; KEY_LEN] = [0x7A; KEY_LEN];
const PACK_ID: &str = "realpr-more-itertools-1136";
const PATCH_FIXTURE: &str = "diff --git a/x b/x\n+todo33-e2e\n";

const EVIDENCE_MATCH_NAME: &str = "task-33-local-e2e-match.txt";
const EVIDENCE_UNTRUSTED_NAME: &str = "task-33-local-e2e-untrusted.txt";

/// Prefer operator evidence dir when writable; else temp (CI runners).
fn evidence_path(name: &str) -> PathBuf {
    let preferred = PathBuf::from("/root/.omo/evidence/gbase-agent-challenge-deepagent");
    let dir = if std::fs::create_dir_all(&preferred).is_ok()
        && std::fs::write(preferred.join(".w"), b"ok").is_ok()
    {
        let _ = std::fs::remove_file(preferred.join(".w"));
        preferred
    } else {
        let d = std::env::temp_dir().join("gbase-agent-challenge-deepagent");
        std::fs::create_dir_all(&d).expect("temp evidence dir");
        d
    };
    dir.join(name)
}

fn now_iso() -> String {
    std::process::Command::new("date")
        .arg("-Iseconds")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map_or_else(|| "unknown".into(), |s| s.trim().to_owned())
}

fn sk() -> [u8; KEY_LEN] {
    MiniSecretKey::generate_with(rand_core::OsRng).to_bytes()
}

fn cvm_pk() -> [u8; KEY_LEN] {
    MiniSecretKey::from_bytes(&CVM_SK)
        .expect("sk")
        .expand(schnorrkel::ExpansionMode::Ed25519)
        .to_public()
        .to_bytes()
}

fn e_three() -> BTreeSet<[u8; KEY_LEN]> {
    BTreeSet::from([SOLVER, ZERO, UNREACH])
}

fn expected_set() -> ExpectedSet {
    ExpectedSet {
        block_hash: {
            let mut h = [0u8; 32];
            h[0] = 0x33;
            h
        },
        participants: vec![
            ExpectedParticipant {
                hotkey: SOLVER,
                uid: 1,
            },
            ExpectedParticipant {
                hotkey: ZERO,
                uid: 2,
            },
            ExpectedParticipant {
                hotkey: UNREACH,
                uid: 3,
            },
        ],
    }
}

fn challenges(pk: [u8; KEY_LEN]) -> ChallengesBody {
    ChallengesBody {
        challenges: vec![ChallengeEntry {
            id: CHALLENGE_ID.as_bytes().to_vec(),
            public_key: pk,
            emission_share_bps: BPS_DENOM,
            policy: ParticipantPolicy::AllMetagraphHotkeys,
        }],
    }
}

async fn spawn_gateway(
    challenges: ChallengesBody,
    weights: Arc<MemoryRawWeightStore>,
    bundles: SharedBundleStore,
) -> (SocketAddr, oneshot::Sender<()>) {
    let _ = telemetry::init_tracing();
    let metrics = init_metrics().expect("metrics");
    let registry = Registry::shared(RegistryConfig::default());
    let app = build_app_with_bundles(
        metrics,
        registry,
        &TlsConfig::default(),
        Arc::new(challenges),
        weights as Arc<dyn RawWeightStore>,
        bundles,
    )
    .expect("router");
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let (tx, rx) = oneshot::channel::<()>();
    tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = rx.await;
            })
            .await
            .expect("serve");
    });
    let client = reqwest::Client::new();
    for _ in 0..80 {
        if client
            .get(format!("http://{addr}/healthz"))
            .send()
            .await
            .is_ok_and(|r| r.status().is_success())
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    (addr, tx)
}

fn gw_client(base: &str) -> GatewayClient {
    GatewayClient::new(GatewayClientConfig {
        base_url: base.into(),
        max_attempts: agent_challenge::DEFAULT_MAX_RETRIES,
        backoff: Duration::from_millis(5),
    })
    .unwrap()
}

fn dummy_pack() -> HarborPack {
    HarborPack {
        task_id: PACK_ID.into(),
        schema_version: "1".into(),
        repository_url: "https://github.com/more-itertools/more-itertools.git".into(),
        base_commit_hash: "e4d2a4a2a97246a73856754b2c4866d7f41d4875".into(),
        instruction: "fix".into(),
        dockerfile: b"FROM scratch\n".to_vec(),
        agent_timeout_sec: 60,
        verifier_timeout_sec: Some(30),
        held_out: HeldOutMaterials {
            solution_patch: Some(PATCH_FIXTURE.as_bytes().to_vec()),
            test_patch: None,
            grader_py: None,
        },
        files: vec![],
    }
}

fn signed_result(epoch: u64, miner: [u8; KEY_LEN], patch: &[u8]) -> TaskResultV1 {
    let digest = patch_sha256(patch);
    let body = WorkReceiptBodyV1 {
        challenge_id: CHALLENGE_ID.as_bytes().to_vec(),
        scoring_version: SCORING_VERSION,
        epoch,
        miner_hotkey: miner,
        pack_id: PACK_ID.as_bytes().to_vec(),
        patch_sha256: digest,
    };
    let signed = sign_work_receipt(&CVM_SK, body).expect("sign");
    TaskResultV1 {
        protocol: DISPATCH_PROTOCOL.into(),
        challenge_id: CHALLENGE_ID.into(),
        scoring_version: SCORING_VERSION,
        epoch,
        miner_hotkey_hex: hex32(&miner),
        pack_id: PACK_ID.into(),
        status: TaskStatusV1::Completed,
        model_patch: Some(String::from_utf8_lossy(patch).into_owned()),
        patch_sha256_hex: hex::encode(digest),
        receipt_sig_hex: hex::encode(signed.signature),
    }
}

fn exp_bind(epoch: u64, miner: [u8; KEY_LEN]) -> ExpectedReceiptBind {
    ExpectedReceiptBind {
        challenge_id: CHALLENGE_ID.into(),
        scoring_version: SCORING_VERSION,
        epoch,
        miner_hotkey: miner,
        pack_id: PACK_ID.into(),
        cvm_receipt_pk: cvm_pk(),
    }
}

/// Verifier that awards reward 1 for non-empty patches (fixture grade path).
struct FixtureVerifier {
    calls: Arc<AtomicU32>,
}

impl Verifier for FixtureVerifier {
    fn grade(
        &self,
        _pack: &HarborPack,
        model_patch: &[u8],
    ) -> Result<Reward, agent_challenge::VerifyError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if model_patch.is_empty() {
            Reward::try_new(0)
        } else {
            Reward::try_new(1)
        }
    }
}

struct FakeRunner;

impl EpochDispatchClient for FakeRunner {
    async fn capacity(&self, _miner: [u8; 32]) -> RunnerCapacity {
        RunnerCapacity {
            max_concurrency: 2,
            current_load: 0,
        }
    }

    async fn run_pack(
        &self,
        miner: [u8; 32],
        descriptor: TaskDescriptorV1,
    ) -> Result<TaskResultV1, String> {
        if miner == UNREACH {
            std::future::pending::<()>().await;
            unreachable!()
        }
        let patch = if miner == SOLVER { PATCH_FIXTURE } else { "" };
        Ok(TaskResultV1 {
            protocol: DISPATCH_PROTOCOL.into(),
            challenge_id: descriptor.challenge_id,
            scoring_version: descriptor.scoring_version,
            epoch: descriptor.epoch,
            miner_hotkey_hex: descriptor.miner_hotkey_hex,
            pack_id: descriptor.pack_id,
            status: TaskStatusV1::Completed,
            model_patch: Some(patch.into()),
            patch_sha256_hex: hex::encode(patch_sha256(patch.as_bytes())),
            receipt_sig_hex: "00".repeat(64),
        })
    }
}

fn harbor_enabled() -> bool {
    std::env::var("GBASE_E2E_HARBOR").ok().as_deref() == Some("1")
}

fn pack_dir() -> PathBuf {
    std::env::var("GBASE_VERIFY_PACK").map_or_else(
        |_| PathBuf::from("/tmp/da_m18c_hf_pull/tasks/realpr-more-itertools-1136"),
        PathBuf::from,
    )
}

fn docker_base() -> String {
    std::env::var("GBASE_DOCKER_BASE").unwrap_or_else(|_| "http://127.0.0.1:2375".into())
}

fn env_image() -> String {
    std::env::var("GBASE_VERIFY_IMAGE").unwrap_or_else(|_| {
        "gbase-verify-env-more-itertools-1136@sha256:462caa0ae2f4ce87509323a33c383eb6b5c364fff4350ba33c2c2bddae62537f"
            .into()
    })
}

/// Try real Harbor grade of more-itertools solution.patch. Returns (reward, note).
fn try_harbor_grade() -> Option<(u8, String)> {
    if !harbor_enabled() {
        return None;
    }
    let dir = pack_dir();
    if !dir.is_dir() {
        return Some((0, format!("pack missing at {}", dir.display())));
    }
    let pack = match load_pack(&dir) {
        Ok(p) => p,
        Err(e) => return Some((0, format!("load_pack err: {e}"))),
    };
    let solution = pack.held_out.solution_patch.clone()?;
    let work = PathBuf::from(format!("/tmp/gbase-e2e-harbor-{}", uuid::Uuid::new_v4()));
    let v = HarborVerifier::new(HarborVerifierConfig {
        docker_base: docker_base(),
        environment_image: env_image(),
        work_root: work.clone(),
        timeout_sec_override: Some(600),
        reward_zero_as_err: false,
    })
    .ok()?;
    let r = match v.grade(&pack, &solution) {
        Ok(rw) => rw.value(),
        Err(e) => {
            let _ = std::fs::remove_dir_all(&work);
            return Some((0, format!("harbor grade err: {e:?}")));
        }
    };
    let owned = v.owned_count().unwrap_or(999);
    let _ = std::fs::remove_dir_all(&work);
    Some((
        r,
        format!("harbor_reward={r} owned_containers_after={owned}"),
    ))
}

/// S1 — runner epoch → intake/grade → |E| leaves → seal → validator Match.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[allow(clippy::too_many_lines)]
async fn s1_full_local_epoch_match_and_seal() {
    // --- 1. Runner epoch dispatch (fake miners, real loop) ---
    let expected = expected_set();
    let n = expected.participants.len();
    let signers = ActiveSignerRegistry::new();
    let config = EpochDispatchConfig {
        challenge_id: CHALLENGE_ID.into(),
        scoring_version: SCORING_VERSION,
        epoch: EPOCH,
        expected: expected.clone(),
        catalog: vec![PackId::new(PACK_ID)],
        deadline: Duration::from_millis(80),
        deadline_unix_ms: 1_700_000_033_080,
    };
    let dispatch = run_epoch_dispatch(&config, Arc::new(FakeRunner), &signers)
        .await
        .expect("epoch dispatch");
    assert_eq!(dispatch.outcomes.len(), n, "exactly |E| runner outcomes");
    assert!(matches!(
        dispatch.outcomes.get(&SOLVER),
        Some(MinerEpochOutcome::Completed { .. })
    ));
    assert!(matches!(
        dispatch.outcomes.get(&UNREACH),
        Some(MinerEpochOutcome::TimedOut { .. })
    ));

    // --- 2. Intake + grade (real grade path; optional Harbor) ---
    let harbor_note = try_harbor_grade().map_or_else(
        || "harbor_skipped (set GBASE_E2E_HARBOR=1 for live pack grade)".into(),
        |(_, n)| n,
    );
    let grade_calls = Arc::new(AtomicU32::new(0));
    let verifier = FixtureVerifier {
        calls: Arc::clone(&grade_calls),
    };
    let pack = dummy_pack();
    let solution_bytes = pack
        .held_out
        .solution_patch
        .clone()
        .expect("fixture solution");
    let solver_result = signed_result(EPOCH, SOLVER, &solution_bytes);
    let intake = intake_and_grade(&exp_bind(EPOCH, SOLVER), &solver_result, &verifier, &pack)
        .expect("intake grade");
    assert_eq!(intake.leaf, ScoreOrAbsence::Score { value: SCORE_MAX });
    assert_eq!(
        grade_calls.load(Ordering::SeqCst),
        1,
        "verifier invoked once"
    );

    // Map dispatch outcomes → score map covering E (D24).
    let mut scores: BTreeMap<[u8; KEY_LEN], ScoreOrAbsence> = BTreeMap::new();
    scores.insert(SOLVER, intake.leaf.clone());
    scores.insert(ZERO, ScoreOrAbsence::Score { value: 0 });
    scores.insert(
        UNREACH,
        ScoreOrAbsence::NoScore {
            reason: NoScoreReasonCode::Timeout,
        },
    );
    let covered = score_map_covering_expected(&e_three(), &scores, &dispatch.outcomes);
    assert_eq!(covered.len(), n);

    // Prefer Score path when pack grades clean (solver has SCORE_MAX).
    assert!(matches!(
        covered.get(&SOLVER),
        Some(ScoreOrAbsence::Score { value: SCORE_MAX })
    ));

    // --- 3. Emit |E| signed leaves ---
    let challenge_secret = sk();
    let challenge_pk = public_key_from_secret(&challenge_secret).unwrap();
    let leaves =
        emit_signed_leaf_set(&challenge_secret, EPOCH, &e_three(), &covered).expect("emit |E|");
    assert_eq!(leaves.len(), n, "sealed leaf count must equal |E|");

    // --- 4. Gateway submit + seal ---
    let weights = Arc::new(MemoryRawWeightStore::new());
    let bundles = Arc::new(MemoryBundleStore::new());
    let ch_body = challenges(challenge_pk);
    let (addr, shutdown) = spawn_gateway(
        ch_body.clone(),
        Arc::clone(&weights),
        bundles.clone() as SharedBundleStore,
    )
    .await;
    let http = reqwest::Client::new();
    let pre = http
        .get(format!("http://{addr}/v1/weights/latest"))
        .send()
        .await
        .unwrap();
    assert_eq!(pre.status().as_u16(), 404, "no sealed bundle yet");

    let client = gw_client(&format!("http://{addr}"));
    let outcomes = submit_signed_leaf_set(&client, &leaves)
        .await
        .expect("submit");
    assert_eq!(outcomes.len(), n);
    assert!(outcomes.iter().all(|o| *o == SubmitOutcome::Accepted));
    assert_eq!(weights.len(), n, "store holds |E| rows");

    let chain = FakeChain::new(FakeChainConfig {
        current_block: BLOCK_B.max(10),
        hotkeys: e_three().iter().map(|h| h.to_vec()).collect(),
        owner_hotkey: vec![0xA1; 32],
        ..FakeChainConfig::default()
    });
    let gsk = sk();
    let mdigest = measurements_digest(&MeasurementsBody::default());
    let params = SealParams {
        epoch: EPOCH,
        netuid: 1,
        block_b: BLOCK_B,
        gateway_secret: gsk,
        measurements_digest: mdigest,
    };
    let bundle = seal_epoch(
        &chain,
        &ch_body,
        weights.as_ref(),
        bundles.as_ref(),
        &params,
    )
    .expect("seal");
    assert_eq!(bundle.body.leaves.len(), n, "sealed |E| leaves");
    assert_eq!(bundle.body.protocol_version, 1);
    // scoring_version lives on leaves / challenge path; bundle algorithm is separate.

    let latest = http
        .get(format!("http://{addr}/v1/weights/latest"))
        .send()
        .await
        .unwrap();
    assert_eq!(latest.status().as_u16(), 200);
    let json: serde_json::Value = latest.json().await.unwrap();
    assert_eq!(json["epoch"], EPOCH);
    let merkle_hex = json["merkle_root"].as_str().unwrap().to_owned();
    assert_eq!(merkle_hex, hex::encode(bundle.body.merkle_root));
    let fv_len = json["final_vector"].as_array().unwrap().len();
    assert!(fv_len > 0, "final_vector present");

    // --- 5. Validator recompute + Match ---
    let trust = LocalTrustRoot {
        challenges: ch_body,
        measurements_digest: mdigest,
    };
    let comparison = compare_bundle(&bundle, &chain, &trust);
    let match_line = match &comparison {
        ComparisonOutcome::Match {
            epoch,
            merkle_root,
            vector_hash,
            local_vector,
            gateway_vector,
            ..
        } => {
            assert_eq!(*epoch, EPOCH);
            assert_eq!(local_vector, gateway_vector, "dual vector equality");
            format!(
                "Match epoch={epoch} merkle_root={} vector_hash={} local_vector_len={} gateway_vector_len={}",
                hex::encode(merkle_root),
                hex::encode(vector_hash),
                local_vector.len(),
                gateway_vector.len()
            )
        }
        other => panic!("expected ComparisonOutcome::Match, got {other:?}"),
    };
    eprintln!("VALIDATOR_LOG {match_line}");

    let view = recompute_view_from_comparison(&comparison);
    let root = bundle.body.merkle_root;
    let cross = CrossCheckOutcome::Agreed {
        epoch: EPOCH,
        merkle_root: root,
        sample_size: 1,
        statements: vec![],
    };
    let vsk = sk();
    let vpk = public_key_from_secret(&vsk).unwrap();
    let decision = apply_three_outcome_policy(
        &view,
        &cross,
        Some(&bundle),
        &chain,
        &trust,
        5000,
        DissentSigner {
            secret: &vsk,
            hotkey: vpk,
        },
        None,
    )
    .expect("policy");
    assert!(
        matches!(decision, EpochDecision::Match { .. }),
        "policy Match, got {decision:?}"
    );
    assert_eq!(decision.submissions().len(), 1);
    assert!(decision.dissent().is_none());
    eprintln!("POLICY_LOG EpochDecision::Match submissions=1 dissent=none");

    // --- 6. Evidence ---
    let evidence = format!(
        "task-33 local e2e Match + sealed bundle\n\
         date: {}\n\
         protocol_version: {}\n\
         scoring_version: {}\n\
         epoch: {}\n\
         |E|: {}\n\
         runner_outcomes: {}\n\
         grade_path: intake_and_grade+FixtureVerifier (solution → SCORE_MAX)\n\
         harbor: {}\n\
         grade_calls: {}\n\
         solver_leaf: {:?}\n\
         submitted_leaves: {}\n\
         store_len: {}\n\
         sealed_leaves: {}\n\
         sealed_merkle_root: {}\n\
         latest_status: 200\n\
         latest_epoch: {}\n\
         final_vector_len: {}\n\
         validator_log: {}\n\
         policy: EpochDecision::Match submissions=1 dissent=none\n\
         orphan_check: in-process gateway stopped; no docker grade containers when harbor skipped\n",
        now_iso(),
        bundle.body.protocol_version,
        SCORING_VERSION,
        EPOCH,
        n,
        dispatch.outcomes.len(),
        harbor_note,
        grade_calls.load(Ordering::SeqCst),
        intake.leaf,
        leaves.len(),
        weights.len(),
        bundle.body.leaves.len(),
        merkle_hex,
        json["epoch"],
        fv_len,
        match_line,
    );
    let path = evidence_path(EVIDENCE_MATCH_NAME);
    std::fs::write(&path, &evidence).expect("write match evidence");
    eprintln!("EVIDENCE_MATCH written path={path:?}\n{evidence}");

    let _ = shutdown.send(());
}

/// S2 — foreign trust root / untrusted challenge key → reject; no seal.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn s2_untrusted_foreign_key_no_seal() {
    let challenge_secret = sk();
    let challenge_pk = public_key_from_secret(&challenge_secret).unwrap();
    let foreign = sk();

    let scores = BTreeMap::from([
        (SOLVER, ScoreOrAbsence::Score { value: SCORE_MAX }),
        (ZERO, ScoreOrAbsence::Score { value: 0 }),
        (
            UNREACH,
            ScoreOrAbsence::NoScore {
                reason: NoScoreReasonCode::Timeout,
            },
        ),
    ]);
    let foreign_leaves =
        emit_signed_leaf_set(&foreign, EPOCH, &e_three(), &scores).expect("foreign emit");

    let weights = Arc::new(MemoryRawWeightStore::new());
    let bundles = Arc::new(MemoryBundleStore::new());
    let ch_body = challenges(challenge_pk);
    let (addr, shutdown) = spawn_gateway(
        ch_body.clone(),
        Arc::clone(&weights),
        bundles.clone() as SharedBundleStore,
    )
    .await;
    let client = gw_client(&format!("http://{addr}"));

    let err = submit_signed_leaf_set(&client, &foreign_leaves)
        .await
        .expect_err("untrusted key must fail");
    let status = match &err {
        SubmitError::Http { status, body } => {
            assert_eq!(*status, 401, "body={body}");
            *status
        }
        other => panic!("expected Http 401, got {other:?}"),
    };
    assert_eq!(weights.len(), 0, "no rows stored for untrusted key");

    // Seal must fail (incomplete / empty set) — no bundle sealed.
    let chain = FakeChain::new(FakeChainConfig {
        current_block: BLOCK_B.max(10),
        hotkeys: e_three().iter().map(|h| h.to_vec()).collect(),
        owner_hotkey: vec![0xA1; 32],
        ..FakeChainConfig::default()
    });
    let gsk = sk();
    let mdigest = measurements_digest(&MeasurementsBody::default());
    let params = SealParams {
        epoch: EPOCH,
        netuid: 1,
        block_b: BLOCK_B,
        gateway_secret: gsk,
        measurements_digest: mdigest,
    };
    let seal_err = seal_epoch(
        &chain,
        &ch_body,
        weights.as_ref(),
        bundles.as_ref(),
        &params,
    )
    .expect_err("seal without leaves must fail");
    eprintln!("SEAL_REJECT {seal_err:?}");

    let http = reqwest::Client::new();
    let latest = http
        .get(format!("http://{addr}/v1/weights/latest"))
        .send()
        .await
        .unwrap();
    assert_eq!(
        latest.status().as_u16(),
        404,
        "no sealed bundle after untrusted reject"
    );

    let evidence = format!(
        "task-33 local e2e untrusted foreign key\n\
         date: {}\n\
         foreign_submit_status: {}\n\
         store_len_after_reject: {}\n\
         seal_error: {seal_err:?}\n\
         latest_status: 404\n\
         sealed: false\n\
         note: gateway trusts challenge_pk; leaves signed by foreign sk → 401; no bundle sealed\n",
        now_iso(),
        status,
        weights.len(),
    );
    let path = evidence_path(EVIDENCE_UNTRUSTED_NAME);
    std::fs::write(&path, &evidence).expect("write untrusted evidence");
    eprintln!("EVIDENCE_UNTRUSTED written path={path:?}\n{evidence}");

    let _ = shutdown.send(());
}
