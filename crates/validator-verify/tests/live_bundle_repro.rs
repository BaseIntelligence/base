//! Live repro: verify a fetched prod bundle against Finney, printing the exact
//! block hashes involved. Ignored by default (needs network + bundle file).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use bundle::{decode_bundle, LocalTrustRoot};
use chain::ChainClient;
use chain_live::LiveChainClient;
use trustroot::{load_config_dir, measurements_digest};
use validator_verify::recompute::compare_bundle;

#[test]
#[ignore = "requires network access to finney and a fetched bundle file"]
fn live_bundle_verify_repro() {
    let path = std::env::var("REPRO_BUNDLE").unwrap_or_else(|_| "/tmp/bundle.bin".into());
    let bytes = std::fs::read(&path).expect("bundle file");
    let bundle = decode_bundle(&bytes).expect("decode");
    let mut chain =
        LiveChainClient::connect("wss://entrypoint-finney.opentensor.ai:443").expect("connect");
    chain.set_netuid(bundle.body.netuid);
    let dir = std::env::var("BASE_TRUST_ROOT_DIR").unwrap_or_else(|_| "config".into());
    let (ch, ms) = load_config_dir(std::path::Path::new(&dir), 0, 1).expect("trust root");
    let trust = LocalTrustRoot {
        challenges: ch.primary().expect("ch").body.clone(),
        measurements_digest: measurements_digest(&ms.primary().expect("ms").body),
    };
    let chain_hash = chain.block_hash(bundle.body.block_b).expect("block_hash");
    eprintln!(
        "block_b={} chain_hash=0x{} body_hash=0x{}",
        bundle.body.block_b,
        hex::encode(chain_hash),
        hex::encode(bundle.body.block_hash)
    );
    let outcome = compare_bundle(&bundle, &chain, &trust);
    eprintln!("outcome: {outcome:?}");
}
