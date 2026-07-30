#![allow(clippy::unwrap_used, clippy::expect_used, clippy::cast_possible_truncation)]

//! Integration tests for gbase-trustroot (task-18 VERIFY scenarios).

use std::fs;
use std::path::Path;

use gbase_crypto::KEY_LEN;
use gbase_trustroot::{
    encode_challenges_body, encode_hex, encode_measurements_body, filter_active, load_challenges_file,
    load_config_dir, load_measurements_file, sign_trust_root_raw, ChallengeEntry, ChallengesBody,
    ChallengesToml, ChallengeToml, MeasurementEntry, MeasurementsBody, MeasurementsToml,
    ParticipantPolicy, PolicyToml, TrustRootError, VerifiedRoot, BPS_DENOM,
};
use rand_core::OsRng;
use schnorrkel::MiniSecretKey;
use tempfile::tempdir;

fn gen_mini() -> ([u8; KEY_LEN], [u8; KEY_LEN]) {
    let mini = MiniSecretKey::generate_with(OsRng);
    let secret = mini.to_bytes();
    let public = mini
        .expand(schnorrkel::ExpansionMode::Ed25519)
        .to_public()
        .to_bytes();
    (secret, public)
}

fn write_signed_challenges(
    dir: &Path,
    name: &str,
    owner_secret: &[u8; KEY_LEN],
    version: u32,
    introduced_epoch: u64,
    body: &ChallengesBody,
) {
    body.validate().expect("body ok");
    let toml_doc = ChallengesToml {
        version,
        introduced_epoch,
        challenges: body
            .challenges
            .iter()
            .map(|c| ChallengeToml {
                id: String::from_utf8(c.id.clone()).expect("utf8 id"),
                public_key: encode_hex(&c.public_key),
                emission_share_bps: c.emission_share_bps,
                policy: PolicyToml::Name("all_metagraph_hotkeys".into()),
            })
            .collect(),
    };
    let text = toml::to_string_pretty(&toml_doc).expect("toml");
    let path = dir.join(name);
    fs::write(&path, text).expect("write toml");
    // Re-parse via to_body to match loader canonicalization (sort).
    let parsed: ChallengesToml = toml::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    let body2 = parsed.to_body().unwrap();
    let scale = encode_challenges_body(&body2);
    let sig = sign_trust_root_raw(owner_secret, version, introduced_epoch, &scale).unwrap();
    fs::write(
        format!("{}.sig", path.display()),
        format!("{}\n", encode_hex(&sig)),
    )
    .unwrap();
}

fn write_signed_measurements(
    dir: &Path,
    name: &str,
    owner_secret: &[u8; KEY_LEN],
    version: u32,
    introduced_epoch: u64,
    body: &MeasurementsBody,
) {
    let entries: Vec<_> = body
        .entries
        .iter()
        .map(|e| gbase_trustroot::MeasurementToml {
            mr_td: encode_hex(&e.mr_td),
            rtmr0: encode_hex(&e.rtmr0),
            rtmr1: encode_hex(&e.rtmr1),
            rtmr2: encode_hex(&e.rtmr2),
            rtmr3: encode_hex(&e.rtmr3),
            compose_hash: encode_hex(&e.compose_hash),
        })
        .collect();
    let toml_doc = MeasurementsToml {
        version,
        introduced_epoch,
        measurements: entries,
    };
    let text = toml::to_string_pretty(&toml_doc).unwrap();
    let path = dir.join(name);
    fs::write(&path, text).unwrap();
    let scale = encode_measurements_body(body);
    let sig = sign_trust_root_raw(owner_secret, version, introduced_epoch, &scale).unwrap();
    fs::write(
        format!("{}.sig", path.display()),
        format!("{}\n", encode_hex(&sig)),
    )
    .unwrap();
}

fn dummy_body(challenge_pk: [u8; KEY_LEN]) -> ChallengesBody {
    ChallengesBody {
        challenges: vec![ChallengeEntry {
            id: b"dummy".to_vec(),
            public_key: challenge_pk,
            emission_share_bps: BPS_DENOM,
            policy: ParticipantPolicy::AllMetagraphHotkeys,
        }],
    }
}

/// S1 — missing file fail-closed.
#[test]
fn s1_missing_file_fail_closed() {
    let dir = tempdir().unwrap();
    let (owner_sec, owner_pub) = gen_mini();
    let _ = owner_sec;
    let err = load_challenges_file(&dir.path().join("nope.toml"), &owner_pub).unwrap_err();
    assert!(matches!(err, TrustRootError::MissingFile { .. }), "{err:?}");
}

/// S2 — unsigned reject.
#[test]
fn s2_unsigned_reject() {
    let dir = tempdir().unwrap();
    let (owner_sec, owner_pub) = gen_mini();
    let (_, ch_pk) = gen_mini();
    let body = dummy_body(ch_pk);
    let path = dir.path().join("challenges.toml");
    let doc = ChallengesToml {
        version: 1,
        introduced_epoch: 0,
        challenges: vec![ChallengeToml {
            id: "dummy".into(),
            public_key: encode_hex(&body.challenges[0].public_key),
            emission_share_bps: BPS_DENOM,
            policy: PolicyToml::Name("all_metagraph_hotkeys".into()),
        }],
    };
    fs::write(&path, toml::to_string_pretty(&doc).unwrap()).unwrap();
    // no .sig
    let err = load_challenges_file(&path, &owner_pub).unwrap_err();
    assert!(matches!(err, TrustRootError::Unsigned { .. }), "{err:?}");
    let _ = owner_sec;
}

/// S3 — non-owner signature rejected.
#[test]
fn s3_non_owner_reject() {
    let dir = tempdir().unwrap();
    let (owner_sec, owner_pub) = gen_mini();
    let (impostor_sec, _) = gen_mini();
    let (_, ch_pk) = gen_mini();
    let body = dummy_body(ch_pk);
    // Sign with impostor, verify with owner.
    write_signed_challenges(dir.path(), "c.toml", &impostor_sec, 1, 0, &body);
    let err = load_challenges_file(&dir.path().join("c.toml"), &owner_pub).unwrap_err();
    assert!(matches!(err, TrustRootError::NonOwner), "{err:?}");
    let _ = owner_sec;
}

/// S4 — empty measurements ⇒ every quote rejected.
#[test]
fn s4_empty_measurements_reject_all_quotes() {
    let dir = tempdir().unwrap();
    let (owner_sec, owner_pub) = gen_mini();
    let empty = MeasurementsBody::default();
    write_signed_measurements(dir.path(), "m.toml", &owner_sec, 1, 0, &empty);
    let root = load_measurements_file(&dir.path().join("m.toml"), &owner_pub).unwrap();
    assert!(root.body.entries.is_empty());
    let zero48 = [0u8; 48];
    let zero32 = [0u8; 32];
    assert!(!root.body.allows_quote(
        &zero48, &zero48, &zero48, &zero48, &zero48, &zero32
    ));
    // Even a "matching" empty profile does not exist.
    let filled = MeasurementEntry {
        mr_td: [1u8; 48],
        rtmr0: [2u8; 48],
        rtmr1: [3u8; 48],
        rtmr2: [4u8; 48],
        rtmr3: [5u8; 48],
        compose_hash: [6u8; 32],
    };
    assert!(!root.body.allows_quote(
        &filled.mr_td,
        &filled.rtmr0,
        &filled.rtmr1,
        &filled.rtmr2,
        &filled.rtmr3,
        &filled.compose_hash
    ));
}

/// S4b — non-empty allowlist accepts exact match only.
#[test]
fn s4b_measurement_allowlist_exact() {
    let entry = MeasurementEntry {
        mr_td: [9u8; 48],
        rtmr0: [8u8; 48],
        rtmr1: [7u8; 48],
        rtmr2: [6u8; 48],
        rtmr3: [5u8; 48],
        compose_hash: [4u8; 32],
    };
    let body = MeasurementsBody {
        entries: vec![entry.clone()],
    };
    assert!(body.allows_quote(
        &entry.mr_td,
        &entry.rtmr0,
        &entry.rtmr1,
        &entry.rtmr2,
        &entry.rtmr3,
        &entry.compose_hash
    ));
    let mut bad = entry.mr_td;
    bad[0] ^= 1;
    assert!(!body.allows_quote(
        &bad,
        &entry.rtmr0,
        &entry.rtmr1,
        &entry.rtmr2,
        &entry.rtmr3,
        &entry.compose_hash
    ));
}

/// S5 — D21 dual-accept window then old rejected.
#[test]
fn s5_rotation_dual_accept_then_old_dropped() {
    let (owner_sec, owner_pub) = gen_mini();
    let (_, pk1) = gen_mini();
    let (_, pk2) = gen_mini();
    let body_v1 = dummy_body(pk1);
    let body_v2 = ChallengesBody {
        challenges: vec![ChallengeEntry {
            id: b"dummy".to_vec(),
            public_key: pk2,
            emission_share_bps: BPS_DENOM,
            policy: ParticipantPolicy::AllMetagraphHotkeys,
        }],
    };
    let dir = tempdir().unwrap();
    write_signed_challenges(dir.path(), "v1.toml", &owner_sec, 1, 0, &body_v1);
    // v2 introduced at epoch 10
    write_signed_challenges(dir.path(), "v2.toml", &owner_sec, 2, 10, &body_v2);

    let v1 = load_challenges_file(&dir.path().join("v1.toml"), &owner_pub).unwrap();
    let v2 = load_challenges_file(&dir.path().join("v2.toml"), &owner_pub).unwrap();
    let all = vec![v1.clone(), v2.clone()];

    let rotation_epochs = 3u32;

    // Before v2: only v1
    let a = filter_active(all.clone(), 9, rotation_epochs);
    assert_eq!(a.len(), 1);
    assert_eq!(a[0].version, 1);

    // At introduction: both (window 10..13)
    let a = filter_active(all.clone(), 10, rotation_epochs);
    assert_eq!(a.len(), 2);
    assert_eq!(a[0].version, 1);
    assert_eq!(a[1].version, 2);

    let a = filter_active(all.clone(), 12, rotation_epochs);
    assert_eq!(a.len(), 2);

    // epoch 13 = 10+3 → old dropped
    let a = filter_active(all.clone(), 13, rotation_epochs);
    assert_eq!(a.len(), 1);
    assert_eq!(a[0].version, 2);

    let _ = owner_sec;
}

/// S6 — sign/verify round-trip.
#[test]
fn s6_sign_verify_round_trip() {
    let dir = tempdir().unwrap();
    let (owner_sec, owner_pub) = gen_mini();
    let (_, ch_pk) = gen_mini();
    let body = dummy_body(ch_pk);
    write_signed_challenges(dir.path(), "c.toml", &owner_sec, 1, 0, &body);
    let loaded = load_challenges_file(&dir.path().join("c.toml"), &owner_pub).unwrap();
    assert_eq!(loaded.body, body);
    assert_eq!(loaded.version, 1);
}

/// S8 — bps must sum to 10000.
#[test]
fn s8_bps_sum_validated() {
    let bad = ChallengesBody {
        challenges: vec![ChallengeEntry {
            id: b"a".to_vec(),
            public_key: [0u8; 32],
            emission_share_bps: 9999,
            policy: ParticipantPolicy::AllMetagraphHotkeys,
        }],
    };
    let err = bad.validate().unwrap_err();
    assert!(matches!(err, TrustRootError::InvalidBody(_)));
}

/// S9 — committed repo config/ loads (if present and signed).
#[test]
fn s9_repo_config_loads_when_present() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../config");
    if !root.join("challenges.toml").is_file() {
        return;
    }
    let (ch, ms) = load_config_dir(&root, 0, 3).expect("committed config must verify");
    assert_eq!(ch.primary().unwrap().body.challenges.len(), 1);
    assert_eq!(
        ch.primary().unwrap().body.challenges[0].emission_share_bps,
        BPS_DENOM
    );
    // Task 35: allowlist frozen from real Phala fixtures (one entry).
    let entries = &ms.primary().unwrap().body.entries;
    assert_eq!(entries.len(), 1, "measurements allowlist must contain real CVM entry");
    let e = &entries[0];
    assert!(ms.allows_quote(
        &e.mr_td,
        &e.rtmr0,
        &e.rtmr1,
        &e.rtmr2,
        &e.rtmr3,
        &e.compose_hash
    ));
    // Mutated measurement must not match.
    let mut bad_td = e.mr_td;
    bad_td[0] ^= 0xff;
    assert!(!ms.allows_quote(
        &bad_td,
        &e.rtmr0,
        &e.rtmr1,
        &e.rtmr2,
        &e.rtmr3,
        &e.compose_hash
    ));
}

/// `filter_active` unit on synthetic `VerifiedRoot` without files.
#[test]
fn filter_active_ordering() {
    let mk = |version, introduced_epoch| VerifiedRoot {
        path: std::path::PathBuf::from(format!("v{version}")),
        version,
        introduced_epoch,
        body: ChallengesBody {
            challenges: vec![ChallengeEntry {
                id: b"dummy".to_vec(),
                public_key: [version as u8; 32],
                emission_share_bps: BPS_DENOM,
                policy: ParticipantPolicy::AllMetagraphHotkeys,
            }],
        },
        signature: [0u8; 64],
    };
    let roots = vec![mk(1, 0), mk(2, 100)];
    assert_eq!(filter_active(roots.clone(), 99, 3).len(), 1);
    assert_eq!(filter_active(roots.clone(), 100, 3).len(), 2);
    assert_eq!(filter_active(roots, 103, 3).len(), 1);
}
