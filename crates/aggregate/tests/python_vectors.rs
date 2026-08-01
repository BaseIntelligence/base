//! Replay every frozen Python vector through [`aggregate::python::aggregate_python`].
//!
//! Python is the authority for the served vector (see the crate docs), so a mismatch
//! here is a Rust bug, never a "known divergence".

// Integration tests are separate crates, so clippy's allow-*-in-tests does not apply.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;

use std::fs;
use std::path::{Path, PathBuf};

use aggregate::python::{aggregate_python, to_chain_u16};
use common::{assert_bits_eq, decode_case, parse_json};

fn vector_dirs() -> Vec<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/vectors/python");
    let mut dirs: Vec<PathBuf> = fs::read_dir(&root)
        .expect("vectors/python must exist")
        .map(|e| e.expect("dir entry").path())
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    dirs
}

fn json_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = fs::read_dir(dir)
        .expect("vector dir")
        .map(|e| e.expect("dir entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    files.sort();
    files
}

#[test]
fn every_python_vector_reproduces_exactly() {
    let mut checked = 0usize;
    for dir in vector_dirs() {
        for file in json_files(&dir) {
            let name = file
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("?")
                .to_owned();
            let label = format!(
                "{}/{name}",
                dir.file_name().and_then(|n| n.to_str()).unwrap_or("?")
            );
            let text = fs::read_to_string(&file).expect("read vector");
            let case = decode_case(&name, &parse_json(&text));

            let outcome = aggregate_python(
                &case.results,
                &case.hotkey_to_uid,
                case.min_allowed_weights,
                case.max_weight_limit,
            );

            if let Some(expected_error) = &case.expected_error {
                let err = outcome
                    .as_ref()
                    .err()
                    .unwrap_or_else(|| panic!("{label}: expected error, got {outcome:?}"));
                assert_eq!(&err.to_string(), expected_error, "{label}: error text");
                checked += 1;
                continue;
            }

            let got = outcome.unwrap_or_else(|e| panic!("{label}: unexpected error {e}"));
            let want = case.expected.expect("python_float_output");

            assert_eq!(got.uids, want.uids, "{label}: uids");
            assert_bits_eq(&format!("{label}: weights"), &got.weights, &want.weights);

            let got_hk: Vec<&str> = got.hotkey_weights.iter().map(|(k, _)| k.as_str()).collect();
            let want_hk: Vec<&str> = want
                .hotkey_weights
                .iter()
                .map(|(k, _)| k.as_str())
                .collect();
            assert_eq!(got_hk, want_hk, "{label}: hotkey_weights order");
            assert_bits_eq(
                &format!("{label}: hotkey_weights"),
                &got.hotkey_weights
                    .iter()
                    .map(|(_, v)| *v)
                    .collect::<Vec<_>>(),
                &want
                    .hotkey_weights
                    .iter()
                    .map(|(_, v)| *v)
                    .collect::<Vec<_>>(),
            );

            if let Some(expected_vector) = case.expected_vector {
                let chain: Vec<(u16, u16)> = got
                    .uids
                    .iter()
                    .copied()
                    .zip(to_chain_u16(&got.weights))
                    .collect();
                assert_eq!(chain, expected_vector, "{label}: chain u16 vector");
            }
            checked += 1;
        }
    }
    assert!(checked >= 5, "expected at least the 5 original vectors");
    eprintln!("python vectors reproduced: {checked}");
}

/// Order fidelity: cases `17`–`20` are the same data in four insertion orders.
///
/// `hotkey_weights` key order must differ per case (it is first-appearance order) while
/// the uid-sorted weights stay identical. A `BTreeMap`-based port fails the first half.
#[test]
fn order_fidelity_cases_keep_python_insertion_order() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/vectors/python/8249563774ee2e71c41ae2cfac182ff32aa35dd1");

    let mut orders: Vec<Vec<String>> = Vec::new();
    let mut weights: Vec<Vec<u64>> = Vec::new();

    for file in json_files(&dir) {
        let name = file
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("?")
            .to_owned();
        if !name.contains("order_fidelity") {
            continue;
        }
        let text = fs::read_to_string(&file).expect("read vector");
        let case = decode_case(&name, &parse_json(&text));
        let got = aggregate_python(
            &case.results,
            &case.hotkey_to_uid,
            case.min_allowed_weights,
            case.max_weight_limit,
        )
        .unwrap_or_else(|e| panic!("{name}: {e}"));
        let want = case.expected.expect("python_float_output");

        // Parity with Python for this specific order is what matters.
        assert_eq!(
            got.hotkey_weights
                .iter()
                .map(|(k, _)| k.clone())
                .collect::<Vec<_>>(),
            want.hotkey_weights
                .iter()
                .map(|(k, _)| k.clone())
                .collect::<Vec<_>>(),
            "{name}: hotkey_weights order must match Python for this input order"
        );
        assert_bits_eq(&format!("{name}: weights"), &got.weights, &want.weights);

        orders.push(got.hotkey_weights.iter().map(|(k, _)| k.clone()).collect());
        weights.push(got.weights.iter().map(|w| w.to_bits()).collect());
    }

    assert_eq!(orders.len(), 4, "expected the four order-fidelity cases");
    assert!(
        orders.windows(2).any(|w| w[0] != w[1]),
        "the cases must actually differ in hotkey order, otherwise they prove nothing"
    );
    assert!(
        weights.windows(2).all(|w| w[0] == w[1]),
        "CPython 3.12 sum() is Neumaier-compensated, so these orders agree on weights"
    );
}

/// The chain u16 vector does **not** always sum to 65535 — Python never renormalises.
#[test]
fn chain_u16_sum_is_not_always_65535() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/vectors/python/8249563774ee2e71c41ae2cfac182ff32aa35dd1");
    let mut off = Vec::new();
    let mut total = 0usize;

    for file in json_files(&dir) {
        let name = file
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("?")
            .to_owned();
        let text = fs::read_to_string(&file).expect("read vector");
        let json = parse_json(&text);
        let case = decode_case(&name, &json);
        let Ok(got) = aggregate_python(
            &case.results,
            &case.hotkey_to_uid,
            case.min_allowed_weights,
            case.max_weight_limit,
        ) else {
            continue;
        };
        total += 1;
        let sum: u32 = to_chain_u16(&got.weights)
            .iter()
            .map(|w| u32::from(*w))
            .sum();
        // The vector file records Python's own sum; it must agree.
        if let Some(recorded) = json.get("chain_u16_sum") {
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let recorded = recorded.num() as u32;
            assert_eq!(sum, recorded, "{name}: chain_u16_sum");
        }
        if sum != 65_535 {
            off.push((name, sum));
        }
    }

    assert!(total > 0);
    assert!(
        !off.is_empty(),
        "expected at least one vector whose u16 encoding misses 65535"
    );
    eprintln!("u16 sums != 65535: {off:?} (of {total} non-error vectors)");
}
