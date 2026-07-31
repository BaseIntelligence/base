//! Todo 16: full sim E2E pipeline (no GPU / B300).
//!
//! Happy path: fixture miner fork → sealed admit → build → kernel → sim train
//! faster than champion → guards pass → promote → pay score >0 → signed leaves
//! for E → POST mock gateway.
//!
//! Noise farm: identical binary rejected by antinois (no measure).
//!
//! Artifacts: `/root/.omo/evidence/hypertraining/e2e/` and `task-16-e2e/`.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::too_many_lines)]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use chain::Metagraph;
use crypto::KEY_LEN;
use hypertraining_antinois::Sanction;
use hypertraining_challenge::{
    default_dedupe, emit_signed_leaf_set, public_key_from_secret, run_sim_pipeline,
    search_faster_compiled, submit_signed_leaf_set, verify_leaf_sig, AttestationStatus, EpochCtx,
    GatewayClient, GatewayClientConfig, Hotkey, HypertrainingChallenge, MapAttestationLookup,
    NoScoreReasonCode, PipelineOutcome, ScoreOrAbsence, SimPipelineInput, SubmitOutcome,
    CHALLENGE_ID, CHALLENGE_ID_BYTES, SCORE_MAX,
};
use hypertraining_cluster::{SegmentSeeds, SimBackend, Topology};
use hypertraining_eval::fixtures::fixture_equal_quality_pairs;
use hypertraining_promo::{
    DuelEvidence, HoldoutEvidence, PromoState, PromotionMachine, ScreenEvidence, PROMOTION_K,
    SCREEN_K,
};
use hypertraining_sealed::{
    admit, sealed_symbol_ast_hash, sha256_hex, AdmitError, AdmitInput, DatasetPin, SealedSurfaceV1,
    SegmentPin, DEFAULT_SEALED_SYMBOL_KEYS,
};
use serde_json::{json, Value};
use trustroot::{ChallengeEntry, ChallengesBody, ParticipantPolicy};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const CHAMP_SRC: &str = "def fused_gemm(a, b):\n    return a @ b\n";
const CHAMP_BIN: &[u8] =
    b".version 7.0\n.entry gemm {\nadd.u32 %r1, %r2, %r3;\nmul.f32 %f1, %f2, %f3;\n}\n";
const CAND_SRC: &str = "def pipeline_overlap(x, y):\n    return compute(x)+y\n";
const EPOCH: u64 = 16;
const BUDGET: u64 = 5_000_000;

fn evidence_root() -> PathBuf {
    let preferred = PathBuf::from("/root/.omo/evidence/hypertraining");
    if fs::create_dir_all(preferred.join("e2e")).is_ok()
        && fs::create_dir_all(preferred.join("task-16-e2e")).is_ok()
        && fs::write(preferred.join("e2e").join(".w"), b"ok").is_ok()
    {
        let _ = fs::remove_file(preferred.join("e2e").join(".w"));
        preferred
    } else {
        let d = std::env::temp_dir().join("hypertraining-e2e-evidence");
        fs::create_dir_all(d.join("e2e")).expect("temp e2e");
        fs::create_dir_all(d.join("task-16-e2e")).expect("temp task-16");
        d
    }
}

fn write_json(path: &Path, value: &Value) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("mkdir");
    }
    fs::write(path, serde_json::to_vec_pretty(value).expect("json")).expect("write json");
}

fn write_text(path: &Path, body: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("mkdir");
    }
    fs::write(path, body).expect("write text");
}

fn sealed_fixtures_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../hypertraining-sealed/tests/fixtures")
}

fn load_tree(dir: &Path) -> BTreeMap<String, Vec<u8>> {
    let mut map = BTreeMap::new();
    load_tree_rec(dir, dir, &mut map);
    map
}

fn load_tree_rec(root: &Path, dir: &Path, map: &mut BTreeMap<String, Vec<u8>>) {
    for ent in fs::read_dir(dir).expect("read_dir") {
        let ent = ent.expect("entry");
        let path = ent.path();
        if path.is_dir() {
            load_tree_rec(root, &path, map);
        } else {
            let rel = path
                .strip_prefix(root)
                .expect("strip")
                .to_string_lossy()
                .replace('\\', "/");
            map.insert(rel, fs::read(&path).expect("read"));
        }
    }
}

fn baseline_manifest(files: &BTreeMap<String, Vec<u8>>) -> SealedSurfaceV1 {
    let mut m = SealedSurfaceV1::with_pins(
        "basedeadbeef",
        DatasetPin {
            corpus: "fineweb-edu".into(),
            revision: "rev1".into(),
            order_seed: 42,
        },
        SegmentPin {
            tokens: 1_000_000,
            gbs: 8,
            seq_len: 2048,
        },
    );
    for path in [
        "megatron/core/datasets/blended.py",
        "megatron/core/num_microbatches_calculator.py",
    ] {
        let bytes = files.get(path).unwrap_or_else(|| panic!("missing {path}"));
        m.denylist_hashes.insert(path.to_owned(), sha256_hex(bytes));
    }
    let training = std::str::from_utf8(
        files
            .get("megatron/training/training.py")
            .expect("training.py"),
    )
    .expect("utf8");
    for key in DEFAULT_SEALED_SYMBOL_KEYS {
        let h = sealed_symbol_ast_hash(key, training).expect("ast hash");
        m.sealed_symbols.insert((*key).to_owned(), h);
    }
    m
}

fn mini() -> ([u8; KEY_LEN], [u8; KEY_LEN]) {
    let m = schnorrkel::MiniSecretKey::generate_with(rand_core::OsRng);
    let sk = m.to_bytes();
    let pk = m
        .expand(schnorrkel::ExpansionMode::Ed25519)
        .to_public()
        .to_bytes();
    (sk, pk)
}

fn epoch_ctx(pk: [u8; KEY_LEN], hotkeys: Vec<Vec<u8>>) -> EpochCtx {
    EpochCtx {
        netuid: 1,
        epoch: EPOCH,
        block_hash: {
            let mut h = [0u8; 32];
            h[0] = 0x16;
            h
        },
        metagraph: Metagraph {
            netuid: 1,
            hotkeys,
            owner_hotkey: vec![0u8; 32],
        },
        challenges: ChallengesBody {
            challenges: vec![ChallengeEntry {
                id: CHALLENGE_ID_BYTES.to_vec(),
                public_key: pk,
                emission_share_bps: 0,
                policy: ParticipantPolicy::AllMetagraphHotkeys,
            }],
        },
    }
}

fn leaf_json(leaf: &bundle::LeafV1) -> Value {
    let challenge_id = std::str::from_utf8(&leaf.challenge_id).expect("utf8");
    let soa = match &leaf.score_or_absence {
        ScoreOrAbsence::Score { value } => json!({ "score": { "value": value } }),
        ScoreOrAbsence::NoScore { reason } => {
            let code = match reason {
                NoScoreReasonCode::NotAttempted => 0u8,
                NoScoreReasonCode::Timeout => 1,
                NoScoreReasonCode::InvalidResponse => 2,
                NoScoreReasonCode::AttestationNotVerified => 3,
                NoScoreReasonCode::MinerError => 4,
                NoScoreReasonCode::RateLimited => 5,
                NoScoreReasonCode::ChallengeInternal => 6,
                NoScoreReasonCode::PolicySkip => 7,
            };
            json!({ "no_score": { "reason": code } })
        }
    };
    json!({
        "challenge_id": challenge_id,
        "miner_hotkey": hex::encode(leaf.miner_hotkey),
        "epoch": leaf.epoch,
        "score_or_absence": soa,
        "challenge_sig": hex::encode(leaf.challenge_sig),
    })
}

fn admitted_files_from_tree(files: &BTreeMap<String, Vec<u8>>) -> Vec<(String, Vec<u8>)> {
    // Hermetic build only needs allowlisted changed content; include fusion + training slice.
    let mut out = Vec::new();
    for key in [
        "megatron/core/fusions/softmax.py",
        "megatron/training/training.py",
    ] {
        if let Some(b) = files.get(key) {
            out.push((key.to_owned(), b.clone()));
        }
    }
    out
}

/// S1 happy: full sim path to champion + pay >0 + signed E leaves + mock gateway 202.
#[tokio::test]
async fn e2e_happy_sim_pipeline_to_gateway() {
    let root = evidence_root();
    let e2e_dir = root.join("e2e");
    let task_dir = root.join("task-16-e2e");

    // --- sealed admit (fixture miner fork) ---
    let good = load_tree(&sealed_fixtures_root().join("good_fork"));
    let manifest = baseline_manifest(&good);
    let changed = vec!["megatron/core/fusions/softmax.py".to_owned()];
    admit(&AdmitInput {
        changed_paths: &changed,
        file_contents: &good,
        manifest: &manifest,
    })
    .expect("good_fork must admit");

    // --- build → kernel → antinois → sim train faster → guards → pay ---
    let seeds = SegmentSeeds {
        run_seed: 16,
        aux_seed: 4,
    };
    let topo = Topology::new(2, 1, 1, 1);
    let faster = search_faster_compiled(CHAMP_BIN, BUDGET, &seeds, topo, 50_000).expect("faster");
    let mut dedupe = default_dedupe().expect("dedupe");
    let mut cluster = SimBackend::new();
    let (champ_loss, cand_loss) = fixture_equal_quality_pairs();
    let input = SimPipelineInput {
        cand_source: CAND_SRC,
        cand_compiled: &faster,
        champ_source: CHAMP_SRC,
        champ_compiled: CHAMP_BIN,
        miner_id: "e2e-fast",
        segment_index: 1,
        budget_tokens: BUDGET,
        seeds,
        topology: topo,
        pkey_id: 16,
        noise_ms: 0,
        validator_lock: b"validator-lock-e2e-v1\n",
        admitted_files: admitted_files_from_tree(&good),
        t_champ_ms_override: None,
        champ_loss,
        cand_loss,
    };
    let result = run_sim_pipeline(&input, &mut dedupe, &mut cluster).expect("pipeline");
    assert!(result.kernel_ok, "kernel gate");
    assert!(
        result.antinois.allows_measure(),
        "antinois must allow measure: {:?}",
        result.antinois.sanction
    );
    assert!(
        result.cand_segment.wallclock_ms < result.t_champ_ms,
        "cand {} must beat champ {}",
        result.cand_segment.wallclock_ms,
        result.t_champ_ms
    );
    assert!(
        result.eval.promote_allowed(),
        "guards must pass: {:?}",
        result.eval
    );
    assert!(
        result.score_u64 > 0 && result.score_u64 <= SCORE_MAX,
        "score {}",
        result.score_u64
    );

    // --- promote to CHAMPION ---
    let mut promo = PromotionMachine::new();
    let genesis = [0xAAu8; 32];
    promo.bootstrap_champion(genesis).expect("bootstrap");
    let cand_ck = result.cand_segment.checkpoint_hash;
    let cid = promo.admit(cand_ck);
    assert_eq!(
        promo
            .advance_screen(
                cid,
                ScreenEvidence {
                    kernel_passed: result.kernel_ok,
                    candidate_median_ms: result.cand_segment.wallclock_ms,
                    champion_median_ms: result.t_champ_ms,
                    sign_coherent: true,
                    k: SCREEN_K,
                },
            )
            .expect("screen"),
        PromoState::Screened
    );
    assert_eq!(
        promo
            .advance_duel(
                cid,
                DuelEvidence {
                    p_value: 0.01,
                    non_inferiority: result.eval.quality_ok,
                    physical_plausible: result.eval.physics_ok,
                    k: PROMOTION_K,
                },
            )
            .expect("duel"),
        PromoState::Duelled
    );
    assert_eq!(
        promo
            .advance_holdout(cid, HoldoutEvidence { sign_agrees: true })
            .expect("holdout"),
        PromoState::Confirmed
    );
    assert_eq!(promo.promote(cid).expect("promote"), PromoState::Champion);
    assert_eq!(promo.champion_hash(), Some(cand_ck));

    // --- D24 leaves for E=2 (fast miner + absent sibling) ---
    let (sk, pk) = mini();
    assert_eq!(public_key_from_secret(&sk).unwrap(), pk);
    let miner_fast: Hotkey = [0x16; KEY_LEN];
    let miner_absent: Hotkey = [0x17; KEY_LEN];
    let ch = HypertrainingChallenge::sim();
    let ctx = epoch_ctx(pk, vec![miner_fast.to_vec(), miner_absent.to_vec()]);
    let mut outcomes = BTreeMap::new();
    outcomes.insert(miner_fast, result.outcome.clone());
    let attest = MapAttestationLookup {
        default: AttestationStatus::Verified,
        ..Default::default()
    };
    let scores = ch
        .score_epoch(&ctx, &outcomes, &attest)
        .expect("score_epoch");
    assert_eq!(scores.len(), 2);
    assert!(matches!(
        &scores[&miner_fast],
        ScoreOrAbsence::Score { value } if *value > 0
    ));
    assert_eq!(
        scores[&miner_absent],
        ScoreOrAbsence::NoScore {
            reason: NoScoreReasonCode::ChallengeInternal
        }
    );
    let expected: BTreeSet<_> = scores.keys().copied().collect();
    let leaves = emit_signed_leaf_set(&sk, EPOCH, &expected, &scores).expect("emit");
    assert_eq!(leaves.len(), 2);
    for h in [miner_fast, miner_absent] {
        let leaf = leaves.get(&h).expect("leaf");
        assert_eq!(leaf.challenge_id, CHALLENGE_ID.as_bytes());
        assert_ne!(leaf.challenge_id, b"agent-v1");
        verify_leaf_sig(leaf, &pk).expect("sig");
    }

    // --- POST mock gateway ---
    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/weights/raw"))
        .respond_with(ResponseTemplate::new(202).set_body_json(json!({"status":"accepted"})))
        .expect(2)
        .mount(&mock)
        .await;
    let client = GatewayClient::new(GatewayClientConfig {
        base_url: mock.uri(),
        max_attempts: 2,
        backoff: Duration::from_millis(5),
    })
    .expect("client");
    let submit_out = submit_signed_leaf_set(&client, &leaves)
        .await
        .expect("submit");
    assert_eq!(submit_out.len(), 2);
    assert!(submit_out.iter().all(|o| *o == SubmitOutcome::Accepted));

    // --- artifacts ---
    let leaves_json: Vec<Value> = leaves.values().map(leaf_json).collect();
    write_json(&e2e_dir.join("leaves.json"), &json!(leaves_json));
    write_json(
        &e2e_dir.join("scores.json"),
        &json!({
            "fast_miner": hex::encode(miner_fast),
            "fast_score_u64": result.score_u64,
            "t_cand_ms": result.cand_segment.wallclock_ms,
            "t_champ_ms": result.t_champ_ms,
            "image_digest": result.image_digest,
            "champion_checkpoint": hex::encode(cand_ck),
            "promo_state": "CHAMPION",
            "guards": {
                "quality_ok": result.eval.quality_ok,
                "physics_ok": result.eval.physics_ok,
            },
            "kernel_ok": result.kernel_ok,
            "antinois_allows_measure": result.antinois.allows_measure(),
            "gateway_posts": 2,
            "challenge_id": CHALLENGE_ID,
        }),
    );
    write_json(
        &task_dir.join("happy-summary.json"),
        &json!({
            "scenario": "e2e_happy_sim_pipeline_to_gateway",
            "admit": "ok",
            "score_u64": result.score_u64,
            "leaves": leaves.len(),
            "gateway": "202 Accepted x2",
            "promo": "CHAMPION",
        }),
    );

    let summary = format!(
        "task-16 e2e happy PASS\n\
         admit: good_fork ok\n\
         kernel_ok: {}\n\
         t_cand_ms: {} t_champ_ms: {}\n\
         score_u64: {}\n\
         promo: CHAMPION ck={}\n\
         leaves: {} (challenge_id={})\n\
         gateway: mock POST /v1/weights/raw → 202 x{}\n\
         artifacts: {} leaves.json scores.json\n",
        result.kernel_ok,
        result.cand_segment.wallclock_ms,
        result.t_champ_ms,
        result.score_u64,
        hex::encode(cand_ck),
        leaves.len(),
        CHALLENGE_ID,
        submit_out.len(),
        e2e_dir.display(),
    );
    write_text(&task_dir.join("happy.txt"), &summary);
    write_text(&e2e_dir.join("happy.txt"), &summary);
}

/// S2 noise farm: identical binary → antinois reject, score 0, never measures.
#[tokio::test]
async fn e2e_noise_farm_identical_binary_rejected() {
    let root = evidence_root();
    let e2e_dir = root.join("e2e");
    let task_dir = root.join("task-16-e2e");

    let good = load_tree(&sealed_fixtures_root().join("good_fork"));
    let manifest = baseline_manifest(&good);
    admit(&AdmitInput {
        changed_paths: &["megatron/core/fusions/softmax.py".to_owned()],
        file_contents: &good,
        manifest: &manifest,
    })
    .expect("admit cosmetic path still ok at sealed layer");

    // Cosmetic source rewrite + identical champion binary (farm).
    let cosmetic_src = "def fused_gemm(a, b):\n    # cosmetic farm resubmit\n    return a @ b\n";
    let seeds = SegmentSeeds {
        run_seed: 99,
        aux_seed: 1,
    };
    let topo = Topology::new(2, 1, 1, 1);
    let mut dedupe = default_dedupe().expect("dedupe");
    let mut cluster = SimBackend::new();
    let (champ_loss, cand_loss) = fixture_equal_quality_pairs();
    let input = SimPipelineInput {
        cand_source: cosmetic_src,
        cand_compiled: CHAMP_BIN, // identical binary → sim > 0.85
        champ_source: CHAMP_SRC,
        champ_compiled: CHAMP_BIN,
        miner_id: "e2e-farm",
        segment_index: 2,
        budget_tokens: BUDGET,
        seeds,
        topology: topo,
        pkey_id: 99,
        noise_ms: 0,
        validator_lock: b"validator-lock-e2e-v1\n",
        admitted_files: admitted_files_from_tree(&good),
        t_champ_ms_override: None,
        champ_loss,
        cand_loss,
    };
    let result = run_sim_pipeline(&input, &mut dedupe, &mut cluster).expect("pipeline");
    assert!(
        !result.antinois.allows_measure(),
        "farm must not measure: {:?}",
        result.antinois.sanction
    );
    assert!(
        result.antinois.binary_similarity > 0.85,
        "binary sim {}",
        result.antinois.binary_similarity
    );
    assert_eq!(result.antinois.sanction, Sanction::SilentReject);
    assert_eq!(result.score_u64, 0);
    assert_eq!(result.outcome, PipelineOutcome::MinerZero);
    assert_eq!(result.cand_segment.wallclock_ms, 0, "skipped sim train");

    // Leaf path: score 0 for farm miner, still D24 cover.
    let (sk, pk) = mini();
    let miner: Hotkey = [0xFAu8; KEY_LEN];
    let ch = HypertrainingChallenge::sim();
    let ctx = epoch_ctx(pk, vec![miner.to_vec()]);
    let mut outcomes = BTreeMap::new();
    outcomes.insert(miner, result.outcome);
    let attest = MapAttestationLookup {
        default: AttestationStatus::Verified,
        ..Default::default()
    };
    let scores = ch.score_epoch(&ctx, &outcomes, &attest).expect("scores");
    assert!(matches!(scores[&miner], ScoreOrAbsence::Score { value: 0 }));
    let expected: BTreeSet<_> = scores.keys().copied().collect();
    let leaves = emit_signed_leaf_set(&sk, EPOCH, &expected, &scores).expect("emit");
    verify_leaf_sig(leaves.get(&miner).unwrap(), &pk).unwrap();

    let mock = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/weights/raw"))
        .respond_with(ResponseTemplate::new(202))
        .expect(1)
        .mount(&mock)
        .await;
    let client = GatewayClient::new(GatewayClientConfig {
        base_url: mock.uri(),
        max_attempts: 2,
        backoff: Duration::from_millis(5),
    })
    .unwrap();
    let out = submit_signed_leaf_set(&client, &leaves)
        .await
        .expect("submit");
    assert_eq!(out, vec![SubmitOutcome::Accepted]);

    write_json(
        &e2e_dir.join("farm-reject.json"),
        &json!({
            "scenario": "e2e_noise_farm_identical_binary_rejected",
            "binary_similarity": result.antinois.binary_similarity,
            "sanction": "SilentReject",
            "allows_measure": false,
            "score_u64": 0,
            "wallclock_ms": result.cand_segment.wallclock_ms,
            "leaf": leaf_json(leaves.get(&miner).unwrap()),
        }),
    );
    let summary = format!(
        "task-16 e2e farm REJECT PASS\n\
         binary_similarity: {}\n\
         sanction: SilentReject\n\
         score_u64: 0\n\
         wallclock_ms: 0 (no measure)\n",
        result.antinois.binary_similarity
    );
    write_text(&task_dir.join("farm-reject.txt"), &summary);
    write_text(&e2e_dir.join("farm-reject.txt"), &summary);
}

/// S3 adjacent: denylist fork never reaches train (admit fails).
#[test]
fn e2e_denylist_fork_never_trains() {
    let root = evidence_root();
    let task_dir = root.join("task-16-e2e");
    let good = load_tree(&sealed_fixtures_root().join("good_fork"));
    let dirty = load_tree(&sealed_fixtures_root().join("denylist_touch"));
    let manifest = baseline_manifest(&good);
    let changed = vec![
        "megatron/core/fusions/softmax.py".to_owned(),
        "megatron/core/datasets/blended.py".to_owned(),
    ];
    let err = admit(&AdmitInput {
        changed_paths: &changed,
        file_contents: &dirty,
        manifest: &manifest,
    })
    .expect_err("denylist must reject");
    match err {
        AdmitError::DenylistPathTouched { path } => {
            assert!(path.contains("datasets"), "path={path}");
        }
        other => panic!("expected DenylistPathTouched, got {other:?}"),
    }
    write_text(
        &task_dir.join("denylist-reject.txt"),
        "task-16 e2e denylist REJECT PASS\nadmit: DenylistPathTouched\nnever trains\n",
    );
}
