#![allow(clippy::expect_used, clippy::unwrap_used)]
//! Serialization parity against the live reference service.
//!
//! Fixture `fixtures/reference_weights_latest.json` is a verbatim capture of
//! `GET https://chain.joinbase.ai/v1/weights/latest`. Our body must be a strict
//! superset of it: same keys, same JSON types, plus the Rust-only extras the
//! validator already reads.

use bundle::{EpochBundleBodyV1, EpochBundleV1, LeafV1, ScoreOrAbsence, ALGORITHM_VERSION};
use serde_json::Value;
use std::collections::BTreeSet;
use weights_api::{build_latest, SealRecord};

/// Keys we add on top of the reference contract.
const RUST_ONLY_KEYS: [&str; 2] = ["merkle_root", "final_vector"];

fn reference() -> Value {
    let raw = include_str!("fixtures/reference_weights_latest.json");
    serde_json::from_str(raw).expect("reference fixture parses")
}

fn leaf(challenge: &[u8], hotkey: u8, score: u64) -> LeafV1 {
    LeafV1 {
        challenge_id: challenge.to_vec(),
        miner_hotkey: [hotkey; 32],
        epoch: 4_960_002,
        score_or_absence: ScoreOrAbsence::Score { value: score },
        challenge_sig: [9u8; 64],
    }
}

fn sealed_bundle() -> EpochBundleV1 {
    EpochBundleV1 {
        body: EpochBundleBodyV1 {
            protocol_version: 1,
            epoch: 4_960_002,
            netuid: 100,
            block_b: 4_242,
            block_hash: [1u8; 32],
            metagraph_root: [2u8; 32],
            algorithm_version: ALGORITHM_VERSION,
            emission_shares: vec![
                (b"agent-challenge".to_vec(), 5_000),
                (b"prism".to_vec(), 5_000),
            ],
            measurements_digest: [3u8; 32],
            uid_map: vec![([0xAAu8; 32], 0), ([0xBBu8; 32], 157)],
            leaves: vec![leaf(b"agent-challenge", 0xBB, 10), leaf(b"prism", 0xBB, 20)],
            merkle_root: [4u8; 32],
            final_vector: vec![(0, 32_767), (157, 32_767)],
            gateway_hotkey: [5u8; 32],
        },
        gateway_sig: [6u8; 64],
    }
}

fn ours() -> Value {
    let resp = build_latest(
        &sealed_bundle(),
        SealRecord {
            sealed_at_micros: 1_785_572_565_992_448,
            revision: 1,
        },
    );
    serde_json::to_value(resp).expect("response serializes")
}

fn keys(value: &Value) -> BTreeSet<String> {
    value
        .as_object()
        .expect("json object")
        .keys()
        .cloned()
        .collect()
}

fn kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Compare `ours` against `reference` for one object: key sets must match and
/// every non-null reference value must have the same JSON type on our side.
fn assert_object_parity(label: &str, reference: &Value, ours: &Value) {
    // `int | None` in the reference schema; we have no revisioned snapshot row
    // behind a sealed leaf set, so we emit the contract-valid null.
    const NULLABLE_GAPS: [&str; 1] = ["source_outcomes.revision"];

    let expected = keys(reference);
    let actual: BTreeSet<String> = keys(ours)
        .into_iter()
        .filter(|k| !RUST_ONLY_KEYS.contains(&k.as_str()))
        .collect();
    let missing: Vec<&String> = expected.difference(&actual).collect();
    let extra: Vec<&String> = actual.difference(&expected).collect();
    assert!(
        missing.is_empty() && extra.is_empty(),
        "{label}: key diff — missing={missing:?} extra={extra:?}"
    );

    for key in &expected {
        let want = &reference[key];
        let got = &ours[key];
        if NULLABLE_GAPS.contains(&format!("{label}.{key}").as_str()) {
            assert!(got.is_null(), "{label}.{key}: expected documented null gap");
            continue;
        }
        if matches!(want, Value::Null) {
            // Reference had no value for this seal; we may carry a real one.
            continue;
        }
        assert_eq!(
            kind(want),
            kind(got),
            "{label}.{key}: type mismatch (reference {want}, ours {got})"
        );
    }
}

#[test]
fn top_level_shape_matches_reference() {
    assert_object_parity("root", &reference(), &ours());
}

#[test]
fn rust_only_extras_are_additive() {
    let ours = ours();
    for key in RUST_ONLY_KEYS {
        assert!(ours.get(key).is_some(), "missing extra {key}");
        assert!(
            reference().get(key).is_none(),
            "{key} unexpectedly present in reference"
        );
    }
}

/// Mirror of `validator::coordination::WeightsLatestView`, which must keep
/// deserializing from the widened body.
#[derive(serde::Deserialize)]
struct WeightsLatestView {
    epoch: u64,
    merkle_root: String,
}

#[test]
fn validator_view_still_deserializes() {
    let view: WeightsLatestView = serde_json::from_value(ours()).expect("validator view parses");
    assert_eq!(view.epoch, 4_960_002);
    assert_eq!(view.merkle_root.len(), 64);
}

#[test]
fn nested_shapes_match_reference() {
    let reference = reference();
    let ours = ours();
    for list in ["source_challenges", "source_snapshots", "source_outcomes"] {
        let want = reference[list].as_array().expect("reference array");
        let got = ours[list].as_array().expect("our array");
        assert!(!got.is_empty(), "{list}: empty");
        for item in got {
            assert_object_parity(list, &want[0], item);
        }
    }
    assert_object_parity(
        "metagraph_identity",
        &reference["metagraph_identity"],
        &ours["metagraph_identity"],
    );
}

#[test]
fn constant_fields_match_reference_values() {
    let reference = reference();
    let ours = ours();
    for key in [
        "protocol_version",
        "emission_policy_version",
        "burn_policy_version",
        "mapping_policy_version",
    ] {
        assert_eq!(reference[key], ours[key], "{key}");
    }
    assert_eq!(
        reference["metagraph_identity"]["burn_uid"],
        ours["metagraph_identity"]["burn_uid"]
    );
}

#[test]
fn datetime_rendering_matches_reference_format() {
    let reference = reference();
    let ours = ours();
    for key in ["computed_at", "expires_at", "metagraph_updated_at"] {
        let want = reference[key].as_str().expect("reference datetime");
        let got = ours[key].as_str().expect("our datetime");
        assert!(got.ends_with('Z'), "{key}: {got} is not Z-suffixed");
        assert_eq!(want.len(), got.len(), "{key}: {want} vs {got}");
        assert_eq!(
            want.chars().map(char::is_numeric).collect::<Vec<_>>(),
            got.chars().map(char::is_numeric).collect::<Vec<_>>(),
            "{key}: layout differs ({want} vs {got})"
        );
    }
    assert_eq!(ours["computed_at"], "2026-08-01T08:22:45.992448Z");
}

/// The 720s freshness window is what validators use to reject stale vectors.
#[test]
fn expiry_window_matches_reference_ttl() {
    let reference = reference();
    let ours = ours();
    let ref_delta = seconds(&reference["computed_at"], &reference["expires_at"]);
    let our_delta = seconds(&ours["computed_at"], &ours["expires_at"]);
    assert_eq!(ref_delta, 720);
    assert_eq!(our_delta, ref_delta);
}

fn seconds(from: &Value, to: &Value) -> i64 {
    parse_epoch_secs(to) - parse_epoch_secs(from)
}

/// Parse the fixed `YYYY-MM-DDTHH:MM:SS.ffffffZ` rendering this endpoint emits.
/// Hand-rolled because `deny.toml` bans a direct `chrono` dependency.
fn parse_epoch_secs(v: &Value) -> i64 {
    let s = v.as_str().expect("datetime string");
    let field = |a: usize, b: usize| s[a..b].parse::<i64>().expect("numeric field");
    let (year, month, day) = (field(0, 4), field(5, 7), field(8, 10));
    let (hours, minutes, secs) = (field(11, 13), field(14, 16), field(17, 19));
    assert!(s.ends_with('Z'), "expected UTC 'Z' suffix, got {s}");
    days_from_civil(year, month, day) * 86_400 + hours * 3600 + minutes * 60 + secs
}

/// Inverse of the formatter's `civil_from_days` (Howard Hinnant).
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = if year >= 0 { year } else { year - 399 } / 400;
    let yoe = year - era * 400;
    let mp = if month > 2 { month - 3 } else { month + 9 };
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}
