//! Unit tests for gbase-bundle (TDD VERIFY cases).

use super::*;
use gbase_aggregate::{
    aggregate, ScoreOrAbsence as AggScore, VerifiedLeaf, ALGORITHM_VERSION as AGG_V,
};
use gbase_chain::{FakeChain, FakeChainConfig};
use gbase_crypto::secret_from_bytes;
use gbase_trustroot::{measurements_digest, ChallengeEntry, MeasurementsBody, ParticipantPolicy};
use parity_scale_codec::{Decode, Encode};
use proptest::prelude::*;
use sha2::{Digest, Sha256};

fn sk(tag: u8) -> [u8; 32] {
    let dig = Sha256::digest([0x5A, tag, 0xA5, tag]);
    let mut seed = [0u8; 32];
    seed.copy_from_slice(&dig);
    seed
}

fn pk_of(secret: &[u8; 32]) -> [u8; 32] {
    secret_from_bytes(secret)
        .expect("sk")
        .to_public()
        .to_bytes()
}

fn hk(tag: u8) -> [u8; 32] {
    let mut h = [0u8; 32];
    h[0] = tag;
    h
}

fn to_agg(s: &ScoreOrAbsence) -> AggScore {
    match s {
        ScoreOrAbsence::Score { value } => AggScore::Score { value: *value },
        ScoreOrAbsence::NoScore { reason } => AggScore::NoScore {
            reason: *reason as u8,
        },
    }
}

fn trust_one(cid: &[u8], challenge_pk: [u8; 32], policy: ParticipantPolicy) -> LocalTrustRoot {
    LocalTrustRoot {
        challenges: ChallengesBody {
            challenges: vec![ChallengeEntry {
                id: cid.to_vec(),
                public_key: challenge_pk,
                emission_share_bps: 10_000,
                policy,
            }],
        },
        measurements_digest: measurements_digest(&MeasurementsBody::default()),
    }
}

// re-export ChallengesBody for tests
use gbase_trustroot::ChallengesBody;

fn chain_with(hotkeys: Vec<[u8; 32]>, block_b: u64) -> FakeChain {
    FakeChain::new(FakeChainConfig {
        current_block: block_b.max(10),
        hotkeys: hotkeys.into_iter().map(|h| h.to_vec()).collect(),
        owner_hotkey: vec![0xA1; 32],
        ..FakeChainConfig::default()
    })
}

fn valid_bundle(
    csk: &[u8; 32],
    gsk: &[u8; 32],
    cid: &[u8],
    miners: &[(u8, u64)],
    block_b: u64,
) -> (EpochBundleV1, LocalTrustRoot, FakeChain) {
    let cpk = pk_of(csk);
    let gpk = pk_of(gsk);
    let trust = trust_one(cid, cpk, ParticipantPolicy::AllMetagraphHotkeys);
    let hotkeys: Vec<[u8; 32]> = miners.iter().map(|(t, _)| hk(*t)).collect();
    let chain = chain_with(hotkeys.clone(), block_b);
    let block_hash = chain.block_hash(block_b).expect("hash");
    let rows = metagraph_rows_from_chain(
        &hotkeys.iter().map(|h| h.to_vec()).collect::<Vec<_>>(),
        None,
    )
    .expect("rows");
    let epoch = 7u64;
    let mut leaves = Vec::new();
    for (tag, score) in miners {
        leaves.push(
            make_signed_leaf(
                csk,
                cid,
                hk(*tag),
                epoch,
                ScoreOrAbsence::Score { value: *score },
            )
            .expect("leaf"),
        );
    }
    sort_leaves(&mut leaves);
    let merkle_root = compute_merkle_root(&leaves);
    let uid_map = uid_map_from_rows(&rows);
    let shares = trust.challenges.emission_shares();
    let verified: Vec<VerifiedLeaf> = leaves
        .iter()
        .map(|l| VerifiedLeaf {
            challenge_id: l.challenge_id.clone(),
            miner_hotkey: l.miner_hotkey,
            score_or_absence: to_agg(&l.score_or_absence),
        })
        .collect();
    let final_vector = aggregate(&verified, &shares, &uid_map, AGG_V).expect("agg");
    let body = EpochBundleBodyV1 {
        protocol_version: PROTOCOL_VERSION,
        epoch,
        netuid: 1,
        block_b,
        block_hash,
        metagraph_root: compute_metagraph_root(&rows),
        algorithm_version: ALGORITHM_VERSION,
        emission_shares: shares,
        measurements_digest: trust.measurements_digest,
        uid_map,
        leaves,
        merkle_root,
        final_vector,
        gateway_hotkey: gpk,
    };
    (sign_bundle(gsk, body).expect("sign"), trust, chain)
}

#[test]
fn s0_body_field_order_list() {
    assert_eq!(BODY_FIELD_ORDER.len(), 14);
    assert_eq!(BODY_FIELD_ORDER[0], "protocol_version");
    assert_eq!(BODY_FIELD_ORDER[3], "block_B");
    assert_eq!(BODY_FIELD_ORDER[13], "gateway_hotkey");
}

#[test]
fn s0_field_order_canary() {
    let body = field_order_canary_body();
    let enc = body.encode();
    assert_eq!(&enc[..10], &field_order_canary_prefix());
    let dec = EpochBundleBodyV1::decode(&mut &enc[..]).expect("dec");
    assert_eq!(dec.encode(), enc);
    assert_eq!(&enc[..2], &[1u8, 0]);
    let dig = Sha256::digest(&enc);
    let hex: String = dig.iter().map(|b| format!("{b:02x}")).collect();
    // Frozen sha256 of canary body — fails if §4.1 field order drifts.
    const PIN: &str = "695fee6c3e73507d2c1d3746ad0feb3011aaa3c05af2570b8607eb1f09607bb5";
    assert_eq!(hex, PIN, "field-order canary digest drifted");
}

#[test]
fn s1_happy_path_verify_ok() {
    let (bundle, trust, chain) = valid_bundle(
        &sk(1),
        &sk(2),
        b"dummy",
        &[(0xA1, 50), (0xB2, 30), (0xC3, 20)],
        100,
    );
    let v = bundle.verify(&chain, &trust).expect("verify");
    assert_eq!(v.body.epoch, 7);
    assert_eq!(v.body.leaves.len(), 3);
    let bytes = bundle.encode_bytes();
    assert_eq!(
        EpochBundleV1::decode_bytes(&bytes).unwrap().encode_bytes(),
        bytes
    );
}

#[test]
fn r1_protocol_version_rejected() {
    let (mut bundle, trust, chain) = valid_bundle(&sk(1), &sk(2), b"dummy", &[(0xA1, 1)], 50);
    bundle.body.protocol_version = 99;
    bundle = sign_bundle(&sk(2), bundle.body).unwrap();
    assert!(matches!(
        bundle.verify(&chain, &trust).unwrap_err(),
        BundleError::ProtocolVersionUnsupported(99)
    ));
}

#[test]
fn r2_block_hash_mismatch() {
    let (mut bundle, trust, chain) = valid_bundle(&sk(1), &sk(2), b"dummy", &[(0xA1, 1)], 50);
    bundle.body.block_hash[0] ^= 0xff;
    bundle = sign_bundle(&sk(2), bundle.body).unwrap();
    assert_eq!(
        bundle.verify(&chain, &trust).unwrap_err(),
        BundleError::BlockHashMismatch
    );
}

#[test]
fn r3_metagraph_root_mismatch() {
    let (mut bundle, trust, chain) = valid_bundle(&sk(1), &sk(2), b"dummy", &[(0xA1, 1)], 50);
    bundle.body.metagraph_root[0] ^= 0x01;
    bundle = sign_bundle(&sk(2), bundle.body).unwrap();
    assert_eq!(
        bundle.verify(&chain, &trust).unwrap_err(),
        BundleError::MetagraphRootMismatch
    );
}

#[test]
fn r4_emission_share_mismatch_d23() {
    let (mut bundle, trust, chain) = valid_bundle(&sk(1), &sk(2), b"dummy", &[(0xA1, 1)], 50);
    bundle.body.emission_shares = vec![(b"other".to_vec(), 10_000)];
    bundle = sign_bundle(&sk(2), bundle.body).unwrap();
    assert_eq!(
        bundle.verify(&chain, &trust).unwrap_err(),
        BundleError::EmissionShareMismatch
    );
}

#[test]
fn r5_shares_sum_invalid() {
    let csk = sk(3);
    let gsk = sk(4);
    let cid = b"x";
    let mut trust = trust_one(cid, pk_of(&csk), ParticipantPolicy::AllMetagraphHotkeys);
    trust.challenges.challenges[0].emission_share_bps = 5000;
    let (mut bundle, _, chain) = valid_bundle(&csk, &gsk, cid, &[(0xA1, 1)], 40);
    bundle.body.emission_shares = trust.challenges.emission_shares();
    bundle = sign_bundle(&gsk, bundle.body).unwrap();
    assert_eq!(
        bundle.verify(&chain, &trust).unwrap_err(),
        BundleError::EmissionSharesSumInvalid
    );
}

#[test]
fn r6_measurements_digest_mismatch() {
    let (mut bundle, trust, chain) = valid_bundle(&sk(1), &sk(2), b"dummy", &[(0xA1, 1)], 50);
    bundle.body.measurements_digest[0] ^= 0xff;
    bundle = sign_bundle(&sk(2), bundle.body).unwrap();
    assert_eq!(
        bundle.verify(&chain, &trust).unwrap_err(),
        BundleError::MeasurementsDigestMismatch
    );
}

#[test]
fn r7_merkle_root_mismatch() {
    let (mut bundle, trust, chain) = valid_bundle(&sk(1), &sk(2), b"dummy", &[(0xA1, 1)], 50);
    bundle.body.merkle_root[0] ^= 0x0f;
    bundle = sign_bundle(&sk(2), bundle.body).unwrap();
    assert_eq!(
        bundle.verify(&chain, &trust).unwrap_err(),
        BundleError::MerkleRootMismatch
    );
}

#[test]
fn r8_leaf_sig_invalid() {
    let (mut bundle, trust, chain) = valid_bundle(&sk(1), &sk(2), b"dummy", &[(0xA1, 1)], 50);
    bundle.body.leaves[0].challenge_sig[0] ^= 0xff;
    finalize_body_merkle(&mut bundle.body);
    bundle = sign_bundle(&sk(2), bundle.body).unwrap();
    assert_eq!(
        bundle.verify(&chain, &trust).unwrap_err(),
        BundleError::LeafSignatureInvalid
    );
}

#[test]
fn r9_uid_map_mismatch() {
    let (mut bundle, trust, chain) =
        valid_bundle(&sk(1), &sk(2), b"dummy", &[(0xA1, 1), (0xB2, 1)], 50);
    bundle.body.uid_map.swap(0, 1);
    bundle = sign_bundle(&sk(2), bundle.body).unwrap();
    assert_eq!(
        bundle.verify(&chain, &trust).unwrap_err(),
        BundleError::UidMapMismatch
    );
}

#[test]
fn d18_foreign_challenge_key_rejected() {
    let csk = sk(1);
    let foreign = sk(9);
    let gsk = sk(2);
    let cid = b"dummy";
    let (mut bundle, trust, chain) = valid_bundle(&csk, &gsk, cid, &[(0xA1, 10)], 60);
    let leaf = make_signed_leaf(
        &foreign,
        cid,
        hk(0xA1),
        bundle.body.epoch,
        ScoreOrAbsence::Score { value: 10 },
    )
    .unwrap();
    bundle.body.leaves = vec![leaf];
    finalize_body_merkle(&mut bundle.body);
    let verified: Vec<VerifiedLeaf> = bundle
        .body
        .leaves
        .iter()
        .map(|l| VerifiedLeaf {
            challenge_id: l.challenge_id.clone(),
            miner_hotkey: l.miner_hotkey,
            score_or_absence: to_agg(&l.score_or_absence),
        })
        .collect();
    bundle.body.final_vector = aggregate(
        &verified,
        &bundle.body.emission_shares,
        &bundle.body.uid_map,
        AGG_V,
    )
    .unwrap();
    bundle = sign_bundle(&gsk, bundle.body).unwrap();
    let err = bundle.verify(&chain, &trust).unwrap_err();
    assert!(
        matches!(
            err,
            BundleError::LeafSignatureInvalid | BundleError::LeafChallengeKeyUnknown
        ),
        "{err:?}"
    );
}

#[test]
fn d18_unknown_challenge_id_rejected() {
    let csk = sk(1);
    let gsk = sk(2);
    let (mut bundle, trust, chain) = valid_bundle(&csk, &gsk, b"dummy", &[(0xA1, 1)], 50);
    let leaf = make_signed_leaf(
        &csk,
        b"nope",
        hk(0xA1),
        bundle.body.epoch,
        ScoreOrAbsence::Score { value: 1 },
    )
    .unwrap();
    bundle.body.leaves = vec![leaf];
    finalize_body_merkle(&mut bundle.body);
    bundle = sign_bundle(&gsk, bundle.body).unwrap();
    assert_eq!(
        bundle.verify(&chain, &trust).unwrap_err(),
        BundleError::LeafChallengeKeyUnknown
    );
}

#[test]
fn d24_missing_participant_rejected() {
    let csk = sk(1);
    let gsk = sk(2);
    let (mut bundle, trust, chain) =
        valid_bundle(&csk, &gsk, b"dummy", &[(0xA1, 5), (0xB2, 5)], 70);
    bundle.body.leaves.pop();
    finalize_body_merkle(&mut bundle.body);
    let verified: Vec<VerifiedLeaf> = bundle
        .body
        .leaves
        .iter()
        .map(|l| VerifiedLeaf {
            challenge_id: l.challenge_id.clone(),
            miner_hotkey: l.miner_hotkey,
            score_or_absence: to_agg(&l.score_or_absence),
        })
        .collect();
    bundle.body.final_vector = aggregate(
        &verified,
        &bundle.body.emission_shares,
        &bundle.body.uid_map,
        AGG_V,
    )
    .unwrap();
    bundle = sign_bundle(&gsk, bundle.body).unwrap();
    assert_eq!(
        bundle.verify(&chain, &trust).unwrap_err(),
        BundleError::IncompleteParticipantSet
    );
}

#[test]
fn d24_noscore_covers_participant() {
    let csk = sk(1);
    let gsk = sk(2);
    let cid = b"dummy";
    let trust = trust_one(cid, pk_of(&csk), ParticipantPolicy::AllMetagraphHotkeys);
    let hotkeys = vec![hk(0xA1), hk(0xB2)];
    let block_b = 80u64;
    let chain = chain_with(hotkeys.clone(), block_b);
    let block_hash = chain.block_hash(block_b).unwrap();
    let rows = metagraph_rows_from_chain(
        &hotkeys.iter().map(|h| h.to_vec()).collect::<Vec<_>>(),
        None,
    )
    .unwrap();
    let epoch = 1u64;
    let mut leaves = vec![
        make_signed_leaf(
            &csk,
            cid,
            hk(0xA1),
            epoch,
            ScoreOrAbsence::Score { value: 100 },
        )
        .unwrap(),
        make_signed_leaf(
            &csk,
            cid,
            hk(0xB2),
            epoch,
            ScoreOrAbsence::NoScore {
                reason: NoScoreReasonCode::Timeout,
            },
        )
        .unwrap(),
    ];
    sort_leaves(&mut leaves);
    let uid_map = uid_map_from_rows(&rows);
    let shares = trust.challenges.emission_shares();
    let verified: Vec<VerifiedLeaf> = leaves
        .iter()
        .map(|l| VerifiedLeaf {
            challenge_id: l.challenge_id.clone(),
            miner_hotkey: l.miner_hotkey,
            score_or_absence: to_agg(&l.score_or_absence),
        })
        .collect();
    let final_vector = aggregate(&verified, &shares, &uid_map, AGG_V).unwrap();
    let body = EpochBundleBodyV1 {
        protocol_version: 1,
        epoch,
        netuid: 1,
        block_b,
        block_hash,
        metagraph_root: compute_metagraph_root(&rows),
        algorithm_version: 1,
        emission_shares: shares,
        measurements_digest: trust.measurements_digest,
        uid_map,
        merkle_root: compute_merkle_root(&leaves),
        leaves,
        final_vector,
        gateway_hotkey: pk_of(&gsk),
    };
    let bundle = sign_bundle(&gsk, body).unwrap();
    bundle.verify(&chain, &trust).expect("noscore ok");
}

#[test]
fn gateway_sig_invalid_rejected() {
    let (mut bundle, trust, chain) = valid_bundle(&sk(1), &sk(2), b"dummy", &[(0xA1, 1)], 50);
    bundle.gateway_sig[0] ^= 0xff;
    assert_eq!(
        bundle.verify(&chain, &trust).unwrap_err(),
        BundleError::BundleSignatureInvalid
    );
}

#[test]
fn reencode_decoded_bundle_byte_identical() {
    let (bundle, _, _) = valid_bundle(&sk(1), &sk(2), b"dummy", &[(0xA1, 50), (0xB2, 30)], 33);
    let bytes = bundle.encode_bytes();
    assert_eq!(
        EpochBundleV1::decode_bytes(&bytes).unwrap().encode_bytes(),
        bytes
    );
}

#[test]
fn empty_leaves_merkle_is_empty_root() {
    assert_eq!(compute_merkle_root(&[]), gbase_merkle::EMPTY_ROOT);
}

/// Build a random-ish body from a seed for round-trip property tests.
fn body_from_seed(seed: u64) -> EpochBundleBodyV1 {
    let mut h = Sha256::digest(seed.to_le_bytes());
    let mut block_hash = [0u8; 32];
    block_hash.copy_from_slice(&h);
    h = Sha256::digest(block_hash);
    let mut metagraph_root = [0u8; 32];
    metagraph_root.copy_from_slice(&h);
    h = Sha256::digest(metagraph_root);
    let mut measurements_digest = [0u8; 32];
    measurements_digest.copy_from_slice(&h);
    h = Sha256::digest(measurements_digest);
    let mut merkle_root = [0u8; 32];
    merkle_root.copy_from_slice(&h);
    h = Sha256::digest(merkle_root);
    let mut gateway_hotkey = [0u8; 32];
    gateway_hotkey.copy_from_slice(&h);
    let n_shares = (seed % 4) as usize;
    let mut emission_shares = Vec::new();
    for i in 0..n_shares {
        emission_shares.push((vec![i as u8], (seed.wrapping_add(i as u64) % 1000) as u16));
    }
    let n_uid = (seed % 3) as usize;
    let mut uid_map = Vec::new();
    for i in 0..n_uid {
        let mut hk = [0u8; 32];
        hk[0] = i as u8;
        hk[1] = (seed >> 8) as u8;
        uid_map.push((hk, i as u16));
    }
    let n_fv = (seed % 5) as usize;
    let mut final_vector = Vec::new();
    for i in 0..n_fv {
        final_vector.push((i as u16, ((seed >> i) & 0xffff) as u16));
    }
    EpochBundleBodyV1 {
        protocol_version: (seed & 0xffff) as u16,
        epoch: seed,
        netuid: ((seed >> 16) & 0xffff) as u16,
        block_b: seed.wrapping_mul(3),
        block_hash,
        metagraph_root,
        algorithm_version: ((seed >> 32) & 0xffff) as u16,
        emission_shares,
        measurements_digest,
        uid_map,
        leaves: Vec::new(),
        merkle_root,
        final_vector,
        gateway_hotkey,
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]
    #[test]
    fn prop_body_scale_roundtrip_byte_stable(seed in any::<u64>()) {
        let body = body_from_seed(seed);
        let enc = body.encode();
        let dec = EpochBundleBodyV1::decode(&mut &enc[..]).expect("decode");
        assert_eq!(dec.encode(), enc);
        assert_eq!(dec, body);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]
    #[test]
    fn prop_bundle_envelope_roundtrip(seed in any::<u64>(), sig in any::<[u8; 64]>()) {
        let bundle = EpochBundleV1 {
            body: body_from_seed(seed),
            gateway_sig: sig,
        };
        let enc = bundle.encode();
        let dec = EpochBundleV1::decode(&mut &enc[..]).expect("decode");
        assert_eq!(dec.encode(), enc);
    }
}
