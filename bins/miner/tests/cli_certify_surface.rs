//! Manual-surface QA: spawn attest server, run `miner certify --fixture-mode`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;
use tokio::process::Command;

use attest_parse::parse_tdx_quote_v4;
use attest_policy::replay_compose_hash;
use attest_replay::events_from_json;
use crypto::KEY_LEN;
use trustroot::{MeasurementEntry, MeasurementsBody};
use validator::{spawn_attest_server, AttestState};

const QUOTE: &[u8] = include_bytes!("../../../crates/attest-parse/tests/fixtures/real/quote.bin");
const EVENT_LOG: &[u8] = include_bytes!("../../../crates/attest-parse/tests/fixtures/real/event_log.json");

fn measurements() -> MeasurementsBody {
    let parsed = parse_tdx_quote_v4(QUOTE).unwrap();
    let events = events_from_json(EVENT_LOG).unwrap();
    let (compose_hash, _) = replay_compose_hash(&events).unwrap();
    MeasurementsBody {
        entries: vec![MeasurementEntry {
            mr_td: parsed.td_report.mr_td,
            rtmr0: parsed.td_report.rtmr0,
            rtmr1: parsed.td_report.rtmr1,
            rtmr2: parsed.td_report.rtmr2,
            rtmr3: parsed.td_report.rtmr3,
            compose_hash,
        }],
    }
}

fn miner_bin() -> PathBuf {
    // Same-package bin: cargo sets CARGO_BIN_EXE_miner for integration tests.
    if let Ok(p) = std::env::var("CARGO_BIN_EXE_miner") {
        return PathBuf::from(p);
    }
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("../../target/debug/miner");
    p
}

#[tokio::test]
async fn cli_fixture_mode_verified() {
    let validator = [0xcc; KEY_LEN];
    let miner = [0xaa; KEY_LEN];
    let state = AttestState::with_ok_verifier(measurements(), validator, 1);
    let (addr, shutdown, _join) = spawn_attest_server(state, SocketAddr::from(([127, 0, 0, 1], 0)))
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(30)).await;

    let bin = miner_bin();
    assert!(bin.exists(), "missing binary at {}", bin.display());

    let out = Command::new(&bin)
        .args([
            "certify",
            "--fixture-mode",
            "--validator-url",
            &format!("http://{addr}"),
            "--netuid",
            "1",
            "--epoch",
            "42",
            "--miner-hotkey-hex",
            &hex::encode(miner),
        ])
        .output()
        .await
        .expect("spawn miner");

    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "cli failed status={:?}\nstdout={stdout}\nstderr={stderr}",
        out.status
    );
    assert!(
        stdout.contains("outcome=verified"),
        "stdout missing verified: {stdout}"
    );
    assert!(stdout.contains("grants_credit=true"));
    assert!(stdout.contains("fixture_mode=true"));
    assert!(stdout.contains("carries_prior_verified=false"));

    let _ = shutdown.send(true);
}
