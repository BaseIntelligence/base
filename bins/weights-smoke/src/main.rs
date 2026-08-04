//! Smoke the gateway seal path without a gateway owner wallet.
//!
//! Flow: load challenge mini-secret → cover metagraph with signed leaves →
//! `POST /v1/weights/raw` → `POST /v1/admin/seal` → `GET /v1/weights/latest`.
//! Uses `gateway_sk` already mounted on the gateway for seal signing — not a
//! btcli wallet.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use bundle::{NoScoreReasonCode, ScoreOrAbsence};
use chain::ChainClient;
use challenge_common::{
    emit_signed_leaf_set, submit_signed_leaf_set, GatewayClient, GatewayClientConfig,
};
use challenge_keys::{load_challenge_secret, public_key_from_secret};
use clap::Parser;
use crypto::KEY_LEN;

#[derive(Debug, Parser)]
#[command(
    name = "weights-smoke",
    about = "Submit D24 leaves, admin-seal, assert GET /v1/weights/latest ≠ 404"
)]
struct Args {
    /// Gateway base URL (no trailing slash required).
    #[arg(
        long,
        env = "BASE_GATEWAY_ENDPOINT",
        default_value = "http://127.0.0.1:8080"
    )]
    gateway: String,

    /// Challenge mini-secret file (32 raw bytes or 64 hex).
    #[arg(
        long,
        env = "BASE_CHALLENGE_SK_FILE",
        default_value = "deploy/secrets/prism_sk"
    )]
    challenge_sk: PathBuf,

    /// Challenge id (must match trust root; emission > 0).
    #[arg(long, default_value = "prism")]
    challenge_id: String,

    /// Chain endpoint for metagraph hotkeys.
    #[arg(
        long,
        env = "BASE_CHAIN_ENDPOINT",
        default_value = "wss://test.finney.opentensor.ai:443"
    )]
    chain_endpoint: String,

    /// Subnet netuid.
    #[arg(long, env = "BASE_NETUID", default_value_t = 541)]
    netuid: u16,

    /// Epoch to seal (default: smoke-only range derived from chain tip).
    #[arg(long)]
    epoch: Option<u64>,

    /// Optional inclusive seal block (default: chain tip).
    #[arg(long)]
    block_b: Option<u64>,
}

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("weights-smoke: {e}");
            ExitCode::from(1)
        }
    }
}

struct TipMeta {
    tip: u64,
    expected: BTreeSet<[u8; KEY_LEN]>,
}

fn load_tip_meta(args: &Args) -> Result<TipMeta, String> {
    let mut chain = chain_live::LiveChainClient::connect(&args.chain_endpoint)
        .map_err(|e| format!("chain connect: {e}"))?;
    chain.set_netuid(args.netuid);
    let tip = args
        .block_b
        .map_or_else(|| chain.current_block().map_err(|e| e.to_string()), Ok)?;
    let hash = chain.block_hash(tip).map_err(|e| e.to_string())?;
    let meta = chain.metagraph_at(&hash).map_err(|e| e.to_string())?;
    let mut expected = BTreeSet::new();
    for hk in &meta.hotkeys {
        let arr: [u8; KEY_LEN] = hk
            .as_slice()
            .try_into()
            .map_err(|_| format!("hotkey length {} (want {KEY_LEN})", hk.len()))?;
        expected.insert(arr);
    }
    if expected.is_empty() {
        return Err(format!(
            "metagraph empty at tip={tip} netuid={}",
            args.netuid
        ));
    }
    Ok(TipMeta { tip, expected })
}

fn score_map(expected: &BTreeSet<[u8; KEY_LEN]>) -> BTreeMap<[u8; KEY_LEN], ScoreOrAbsence> {
    let mut scores = BTreeMap::new();
    for (i, hk) in expected.iter().enumerate() {
        // Mix score + absence so seal path is not score-only.
        let soa = if i == 0 {
            ScoreOrAbsence::Score {
                value: 1_000 + u64::try_from(i).unwrap_or(0),
            }
        } else {
            ScoreOrAbsence::NoScore {
                reason: NoScoreReasonCode::NotAttempted,
            }
        };
        scores.insert(*hk, soa);
    }
    scores
}

async fn admin_seal_and_check_latest(
    gateway: &str,
    epoch: u64,
    netuid: u16,
    tip: u64,
) -> Result<(), String> {
    let http = reqwest::Client::builder()
        .timeout(Duration::from_mins(1))
        .build()
        .map_err(|e| e.to_string())?;
    let base = gateway.trim_end_matches('/');
    let seal = http
        .post(format!("{base}/v1/admin/seal"))
        .json(&serde_json::json!({
            "epoch": epoch,
            "netuid": netuid,
            "block_b": tip,
        }))
        .send()
        .await
        .map_err(|e| format!("admin/seal transport: {e}"))?;
    let seal_status = seal.status().as_u16();
    let seal_text = seal.text().await.unwrap_or_default();
    if seal_status != 200 {
        return Err(format!("admin/seal HTTP {seal_status}: {seal_text}"));
    }
    eprintln!("weights-smoke: seal ok: {seal_text}");

    let latest = http
        .get(format!("{base}/v1/weights/latest"))
        .send()
        .await
        .map_err(|e| format!("weights/latest transport: {e}"))?;
    let latest_status = latest.status().as_u16();
    let latest_text = latest.text().await.unwrap_or_default();
    if latest_status != 200 {
        return Err(format!(
            "weights/latest HTTP {latest_status} (want 200; 404 means no sealed bundle): {latest_text}"
        ));
    }
    let json: serde_json::Value =
        serde_json::from_str(&latest_text).map_err(|e| format!("latest json: {e}"))?;
    let got_epoch = json.get("epoch").and_then(serde_json::Value::as_u64);
    if got_epoch != Some(epoch) {
        return Err(format!(
            "weights/latest epoch mismatch: got {got_epoch:?} want {epoch}"
        ));
    }
    eprintln!("weights-smoke: GET /v1/weights/latest OK epoch={epoch}");
    println!("{latest_text}");
    Ok(())
}

async fn run() -> Result<(), String> {
    let args = Args::parse();
    let sk = load_challenge_secret(&args.challenge_sk).map_err(|e| e.to_string())?;
    let pk = public_key_from_secret(&sk).map_err(|e| e.to_string())?;
    eprintln!(
        "weights-smoke: challenge_id={} pk={} gateway={}",
        args.challenge_id,
        hex::encode(pk),
        args.gateway
    );

    let TipMeta { tip, expected } = load_tip_meta(&args)?;
    let epoch = args.epoch.unwrap_or(8_000_000 + tip % 1_000_000);
    eprintln!(
        "weights-smoke: tip={tip} epoch={epoch} participants={}",
        expected.len()
    );

    let scores = score_map(&expected);
    let leaves = emit_signed_leaf_set(&sk, args.challenge_id.as_bytes(), epoch, &expected, &scores)
        .map_err(|e| e.to_string())?;

    let client = GatewayClient::new(GatewayClientConfig {
        base_url: args.gateway.clone(),
        ..GatewayClientConfig::default()
    })
    .map_err(|e| e.to_string())?;
    let outcomes = submit_signed_leaf_set(&client, &leaves)
        .await
        .map_err(|e| format!("submit leaves: {e}"))?;
    eprintln!(
        "weights-smoke: submitted {} leaves ({outcomes:?})",
        outcomes.len()
    );

    admin_seal_and_check_latest(&args.gateway, epoch, args.netuid, tip).await
}
