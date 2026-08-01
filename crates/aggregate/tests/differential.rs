//! Differential harness: Rust vs the live Python authority, on randomized inputs.
//!
//! Ignored by default because it shells out to the BASE Python venv. Run with:
//!
//! ```text
//! cargo test -p aggregate --test differential -- --ignored --nocapture
//! ```
//!
//! Override the interpreter with `BASE_PYTHON` and the package root with `BASE_SRC`.

// Integration tests are separate crates, so clippy's allow-*-in-tests does not apply.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;

use std::path::Path;
use std::process::Command;

use aggregate::python::{aggregate_python, to_chain_u16};
use common::{assert_bits_eq, decode_case, parse_json, Json};

const DEFAULT_PYTHON: &str = "/root/prism-compute-plane/base/.venv/bin/python";
const CASES: usize = 400;
const SEED: &str = "20260101";

#[test]
#[ignore = "requires the BASE Python venv (set BASE_PYTHON / BASE_SRC)"]
fn differential_against_python_authority() {
    let python = std::env::var("BASE_PYTHON").unwrap_or_else(|_| DEFAULT_PYTHON.to_owned());
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/support/differential_gen.py");

    let output = Command::new(&python)
        .arg(&script)
        .arg(std::env::var("BASE_DIFF_CASES").unwrap_or_else(|_| CASES.to_string()))
        .arg(std::env::var("BASE_DIFF_SEED").unwrap_or_else(|_| SEED.to_owned()))
        .output()
        .unwrap_or_else(|e| panic!("spawn {python}: {e}"));
    assert!(
        output.status.success(),
        "generator failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let text = String::from_utf8(output.stdout).expect("utf8 generator output");
    let root = parse_json(&text);
    eprintln!(
        "python: {}",
        root.get("python_version").map_or("?", Json::str).trim()
    );

    let cases = root.get("cases").expect("cases").arr();
    let mut passed = 0usize;
    let mut u16_sums_off = 0usize;

    for case_json in cases {
        let name = case_json.get("name").expect("name").str().to_owned();
        let case = decode_case(&name, case_json);

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
                .unwrap_or_else(|| panic!("{name}: python raised, rust returned {outcome:?}"));
            assert_eq!(&err.to_string(), expected_error, "{name}: error text");
            passed += 1;
            continue;
        }

        let got = outcome.unwrap_or_else(|e| panic!("{name}: rust raised {e}, python did not"));
        let want = case.expected.expect("python_float_output");

        assert_eq!(got.uids, want.uids, "{name}: uids");
        assert_bits_eq(&format!("{name}: weights"), &got.weights, &want.weights);

        let got_hk: Vec<&str> = got.hotkey_weights.iter().map(|(k, _)| k.as_str()).collect();
        let want_hk: Vec<&str> = want
            .hotkey_weights
            .iter()
            .map(|(k, _)| k.as_str())
            .collect();
        assert_eq!(got_hk, want_hk, "{name}: hotkey_weights keys/order");
        assert_bits_eq(
            &format!("{name}: hotkey_weights"),
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

        let chain = to_chain_u16(&got.weights);
        let expected_vector = case.expected_vector.expect("expected_vector");
        let chain_pairs: Vec<(u16, u16)> = got
            .uids
            .iter()
            .copied()
            .zip(chain.iter().copied())
            .collect();
        assert_eq!(chain_pairs, expected_vector, "{name}: chain u16 vector");

        let sum: u32 = chain.iter().map(|w| u32::from(*w)).sum();
        if sum != u32::from(aggregate::python::CHAIN_U16_MAX) {
            u16_sums_off += 1;
        }
        passed += 1;
    }

    assert_eq!(passed, cases.len(), "every case must be checked");
    eprintln!("differential: {passed}/{} cases bit-identical", cases.len());
    eprintln!(
        "chain u16 vectors not summing to 65535: {u16_sums_off}/{} \
         (Python has no post-rounding renormalisation; neither do we)",
        cases.len()
    );
}
