#![allow(clippy::expect_used, clippy::unwrap_used)]
//! Unit tests for chain-live (wiremock + pure fixture tests).

use crate::{
    commit_timelocked_call, decode_bool, decode_hotkey, decode_metagraph, decode_u16, decode_u64,
    decode_vec_vec_u8, set_weights_call, storage_key, storage_map_key, storage_map_key_u16,
    ChainClient, ChainError, Era, LiveChainClient, WeightsTlockPayload,
};
use parity_scale_codec::Encode;
use serde_json::json;
use wiremock::matchers::{body_partial_json, method};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ---------------------------------------------------------------------------
// Pure: storage keys
// ---------------------------------------------------------------------------

#[test]
fn storage_key_is_32_bytes_and_deterministic() {
    let k1 = storage_key("SubtensorModule", "Tempo");
    let k2 = storage_key("SubtensorModule", "Tempo");
    assert_eq!(k1.len(), 32);
    assert_eq!(k1, k2);

    let k3 = storage_key("SubtensorModule", "Keys");
    assert_ne!(k1, k3, "different item must differ");
}

#[test]
fn storage_map_key_u16_has_correct_structure() {
    let k = storage_map_key_u16("SubtensorModule", "Tempo", 1u16);
    // Twox128(pallet) ++ Twox128(item) ++ Twox64(key) ++ key
    // 16 + 16 + 8 + 2 = 42 bytes
    assert_eq!(k.len(), 42);
    // Last 2 bytes are the LE-encoded netuid
    assert_eq!(&k[40..], &[0x01, 0x00]);
}

#[test]
fn storage_map_key_generic_has_correct_structure() {
    let key = [0xAB_u8; 4];
    let k = storage_map_key("SubtensorModule", "Keys", &key);
    // 16 + 16 + 8 + 4 = 44 bytes
    assert_eq!(k.len(), 44);
    assert_eq!(&k[40..], &key);
}

// ---------------------------------------------------------------------------
// Pure: SCALE decode
// ---------------------------------------------------------------------------

#[test]
fn decode_u64_known_vector() {
    // 600 in LE u64
    let bytes = 600u64.to_le_bytes();
    assert_eq!(decode_u64(&bytes).unwrap(), 600);
}

#[test]
fn decode_u16_known_vector() {
    let bytes = 42u16.to_le_bytes();
    assert_eq!(decode_u16(&bytes).unwrap(), 42);
}

#[test]
fn decode_bool_true_false() {
    assert!(decode_bool(&[0x01]).unwrap());
    assert!(!decode_bool(&[0x00]).unwrap());
}

#[test]
fn decode_vec_vec_u8_known_vector() {
    let data: Vec<Vec<u8>> = vec![vec![0xAA; 32], vec![0xBB; 32]];
    let encoded = data.encode();
    let decoded = decode_vec_vec_u8(&encoded).unwrap();
    assert_eq!(decoded, data);
}

#[test]
fn decode_hotkey_raw_32_bytes() {
    let key = [0x42_u8; 32];
    let result = decode_hotkey(&key).unwrap();
    assert_eq!(result, key.to_vec());
}

#[test]
fn decode_hotkey_option_some() {
    let mut bytes = vec![0x01];
    bytes.extend_from_slice(&[0x33_u8; 32]);
    let result = decode_hotkey(&bytes).unwrap();
    assert_eq!(result, vec![0x33_u8; 32]);
}

#[test]
fn decode_metagraph_builds_correctly() {
    let keys = vec![vec![0xAA; 32], vec![0xBB; 32]];
    let owner = vec![0xCC; 32];
    let mg = decode_metagraph(keys.clone(), owner.clone(), 1);
    assert_eq!(mg.netuid, 1);
    assert_eq!(mg.hotkeys, keys);
    assert_eq!(mg.owner_hotkey, owner);
}

// ---------------------------------------------------------------------------
// Pure: Era encoding
// ---------------------------------------------------------------------------

#[test]
fn era_immortal_encodes_zero() {
    assert_eq!(Era::Immortal.encode_era(), vec![0x00]);
}

#[test]
fn era_mortal_period_64_phase_0() {
    // period=64: first=32, trailing_zeros=5, factor=5
    // quantize_factor=16, phase=0 → encoded = 0x50
    let era = Era::Mortal {
        period: 64,
        phase: 0,
    };
    assert_eq!(era.encode_era(), vec![0x50]);
}

#[test]
fn era_mortal_period_128_phase_0() {
    // period=128: first=64, trailing_zeros=6, factor=6
    let era = Era::Mortal {
        period: 128,
        phase: 0,
    };
    assert_eq!(era.encode_era(), vec![0x60]);
}

#[test]
fn era_mortal_rounds_non_power_of_two() {
    // 360 → next_power_of_two = 512
    let era = Era::Mortal {
        period: 360,
        phase: 0,
    };
    // period=512: first=256, trailing_zeros=8, factor=8
    assert_eq!(era.encode_era(), vec![0x80]);
}

// ---------------------------------------------------------------------------
// Pure: extrinsic byte construction
// ---------------------------------------------------------------------------

/// Fixed test secret key (32 bytes, all 0x01 — valid schnorrkel mini-secret).
fn test_secret() -> [u8; 32] {
    [0x01_u8; 32]
}

#[test]
fn set_weights_call_bytes_known_fixture() {
    // set_weights(netuid=1, uids=[0,1], values=[100,200], version_key=0)
    let call = set_weights_call(1, &[0, 1], &[100, 200], 0);
    let expected = [
        0x07, // pallet index 7
        0x00, // call index 0
        0x01, 0x00, // netuid=1 (u16 LE)
        0x08, // Compact(2) = 2*4
        0x00, 0x00, // uid 0 (u16 LE)
        0x01, 0x00, // uid 1 (u16 LE)
        0x08, // Compact(2) = 2*4
        0x64, 0x00, // value 100 (u16 LE)
        0xc8, 0x00, // value 200 (u16 LE)
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // version_key=0 (u64 LE)
    ];
    assert_eq!(call, expected.to_vec());
}

#[test]
fn commit_timelocked_call_bytes_known_fixture() {
    let payload = WeightsTlockPayload {
        hotkey: vec![0xAA; 32],
        uids: vec![0, 1],
        values: vec![100, 200],
        version_key: 0,
    };
    let call = commit_timelocked_call(0, &payload, 99);

    // Expected structure:
    // 0x07 (pallet 7), 0x76 (call 118), 0x00 (mecid 0)
    // payload: Compact(32)=0x80, [0xAA;32], Compact(2)=0x08, [0,0,1,0], 0x08, [100,0,200,0], [0;8]
    // reveal_round: 99 LE = [0x63, 0, 0, 0, 0, 0, 0, 0]
    assert_eq!(call[0], 0x07);
    assert_eq!(call[1], 0x76);
    assert_eq!(call[2], 0x00);
    assert_eq!(call[3], 0x80); // Compact(32)
    assert_eq!(&call[4..36], &[0xAA; 32]); // hotkey
    assert_eq!(call[36], 0x08); // Compact(2) uids
    assert_eq!(&call[37..41], &[0x00, 0x00, 0x01, 0x00]); // uids
    assert_eq!(call[41], 0x08); // Compact(2) values
    assert_eq!(&call[42..46], &[0x64, 0x00, 0xc8, 0x00]); // values
    assert_eq!(&call[46..54], &[0x00; 8]); // version_key
    assert_eq!(
        &call[54..62],
        &[0x63, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
    ); // reveal_round=99
    assert_eq!(call.len(), 62);
}

#[test]
fn build_and_sign_set_weights_structure() {
    let key = test_secret();
    let genesis = [0x11_u8; 32];
    let ext = crate::build_and_sign_set_weights(
        &key,
        0,
        &Era::Immortal,
        &genesis,
        &genesis,
        440,
        1,
        1,
        &[0, 1],
        &[100, 200],
        0,
    )
    .unwrap();

    // 0x84 (version) + 0x00 (MultiAddress::Id) + 32 (pubkey) + 0x01 (Sr25519) + 64 (sig)
    // + 0x00 (era) + 0x00 (nonce) + 0x00 (tip) + 22 (call) = 124
    assert_eq!(ext.len(), 124);
    assert_eq!(ext[0], 0x84);
    assert_eq!(ext[1], 0x00);
    assert_eq!(ext[34], 0x01); // MultiSignature::Sr25519

    // Verify public key is deterministic
    let pubkey = crate::derive_public_key(&key).unwrap();
    assert_eq!(&ext[2..34], &pubkey);

    // Verify era + nonce + tip + call suffix
    assert_eq!(ext[99], 0x00); // Immortal era
    assert_eq!(ext[100], 0x00); // Compact(0) nonce
    assert_eq!(ext[101], 0x00); // Compact(0) tip
    let expected_call = set_weights_call(1, &[0, 1], &[100, 200], 0);
    assert_eq!(&ext[102..], &expected_call[..]);
}

#[test]
fn build_and_sign_commit_timelocked_structure() {
    let key = test_secret();
    let genesis = [0x22_u8; 32];
    let payload = WeightsTlockPayload {
        hotkey: vec![0xAA; 32],
        uids: vec![0, 1],
        values: vec![100, 200],
        version_key: 0,
    };
    let ext = crate::build_and_sign_commit_timelocked(
        &key,
        5,
        &Era::Immortal,
        &genesis,
        &genesis,
        440,
        1,
        0,
        &payload,
        99,
    )
    .unwrap();

    // 1 + 1 + 32 + 1 + 64 + 1 + 1(nonce=5: Compact(5)=0x14) + 1 + 62 = 164
    // Wait: Compact(5) = 5*4 = 20 = 0x14 (single byte since 5 < 64)
    assert_eq!(ext.len(), 164);
    assert_eq!(ext[0], 0x84);
    assert_eq!(ext[1], 0x00);
    assert_eq!(ext[34], 0x01); // Sr25519
    assert_eq!(ext[99], 0x00); // Immortal era
    assert_eq!(ext[100], 0x14); // Compact(5) nonce
    assert_eq!(ext[101], 0x00); // Compact(0) tip

    let expected_call = commit_timelocked_call(0, &payload, 99);
    assert_eq!(&ext[102..], &expected_call[..]);
}

#[test]
fn derive_public_key_is_deterministic() {
    let key = test_secret();
    let pk1 = crate::derive_public_key(&key).unwrap();
    let pk2 = crate::derive_public_key(&key).unwrap();
    assert_eq!(pk1, pk2);
    assert_eq!(pk1.len(), 32);
}

// ---------------------------------------------------------------------------
// wiremock: RPC read methods
// ---------------------------------------------------------------------------

#[tokio::test]
async fn mock_current_block() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(body_partial_json(json!({"method": "chain_getHeader"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "jsonrpc": "2.0", "id": 1,
            "result": {"number": "0x3e8"}
        })))
        .mount(&server)
        .await;

    let uri = server.uri();
    let block = tokio::task::spawn_blocking(move || {
        let client = LiveChainClient::connect(&uri)?;
        client.current_block()
    })
    .await
    .expect("spawn_blocking")
    .expect("current_block");
    assert_eq!(block, 1000);
}

#[tokio::test]
async fn mock_block_time() {
    let server = MockServer::start().await;
    // AuraApi_slot_duration returns SCALE u64 = 12000 ms
    let slot_hex = format!("0x{}", hex::encode(12000u64.to_le_bytes()));
    Mock::given(method("POST"))
        .and(body_partial_json(json!({
            "method": "state_call",
            "params": ["AuraApi_slot_duration"]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "jsonrpc": "2.0", "id": 1,
            "result": slot_hex
        })))
        .mount(&server)
        .await;

    let uri = server.uri();
    let bt = tokio::task::spawn_blocking(move || {
        let client = LiveChainClient::connect(&uri)?;
        client.block_time()
    })
    .await
    .expect("spawn_blocking")
    .expect("block_time");
    assert_eq!(bt, 12000);
}

#[tokio::test]
async fn mock_runtime_version_ok() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(body_partial_json(
            json!({"method": "state_getRuntimeVersion"}),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "jsonrpc": "2.0", "id": 1,
            "result": {
                "specName": "node-subtensor",
                "specVersion": 440,
                "transactionVersion": 1
            }
        })))
        .mount(&server)
        .await;

    let uri = server.uri();
    let rt = tokio::task::spawn_blocking(move || {
        let rpc = crate::LiveChainRpc::connect(&uri)?;
        rpc.state_get_runtime_version()
    })
    .await
    .expect("spawn_blocking")
    .expect("runtime version");
    assert_eq!(rt.spec_version, 440);
    assert_eq!(rt.transaction_version, 1);
}

#[tokio::test]
async fn mock_state_get_storage_u64() {
    let server = MockServer::start().await;
    // Mock any state_getStorage call with a u64=360 (tempo)
    let value_hex = format!("0x{}", hex::encode(360u64.to_le_bytes()));
    Mock::given(method("POST"))
        .and(body_partial_json(json!({"method": "state_getStorage"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "jsonrpc": "2.0", "id": 1,
            "result": value_hex
        })))
        .mount(&server)
        .await;

    let uri = server.uri();
    let tempo = tokio::task::spawn_blocking(move || {
        let client = LiveChainClient::connect(&uri)?;
        // read_netuid_u64 is private, but tempo uses u16. Let's test blocks_since_last_step (u64).
        client.blocks_since_last_step(1)
    })
    .await
    .expect("spawn_blocking")
    .expect("blocks_since_last_step");
    assert_eq!(tempo, 360);
}

#[tokio::test]
async fn mock_state_get_storage_u16() {
    let server = MockServer::start().await;
    let value_hex = format!("0x{}", hex::encode(360u16.to_le_bytes()));
    Mock::given(method("POST"))
        .and(body_partial_json(json!({"method": "state_getStorage"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "jsonrpc": "2.0", "id": 1,
            "result": value_hex
        })))
        .mount(&server)
        .await;

    let uri = server.uri();
    let tempo = tokio::task::spawn_blocking(move || {
        let client = LiveChainClient::connect(&uri)?;
        client.tempo(1)
    })
    .await
    .expect("spawn_blocking")
    .expect("tempo");
    assert_eq!(tempo, 360);
}

// ---------------------------------------------------------------------------
// wiremock: guard tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn set_weights_rejected_when_cr_enabled() {
    let server = MockServer::start().await;
    // Mock CommitRevealWeightsEnabled(1) → true (0x01)
    let key = storage_map_key_u16("SubtensorModule", "CommitRevealWeightsEnabled", 1);
    let hex_key = format!("0x{}", hex::encode(&key));
    Mock::given(method("POST"))
        .and(body_partial_json(json!({
            "method": "state_getStorage",
            "params": [hex_key]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "jsonrpc": "2.0", "id": 1,
            "result": "0x01"
        })))
        .mount(&server)
        .await;

    let uri = server.uri();
    let result = tokio::task::spawn_blocking(move || {
        let client = LiveChainClient::connect(&uri)?;
        client.set_weights(1, vec![0], vec![100], 0)
    })
    .await
    .expect("spawn_blocking");

    let err = result.expect_err("must reject");
    match err {
        ChainError::Other(msg) => {
            assert!(msg.contains("commit_reveal"), "msg: {msg}");
        }
        other => panic!("expected Other, got {other}"),
    }
}

#[tokio::test]
async fn set_weights_refuses_on_spec_version_mismatch() {
    let server = MockServer::start().await;

    // Mock CommitRevealWeightsEnabled(1) → false (0x00)
    let cr_key = storage_map_key_u16("SubtensorModule", "CommitRevealWeightsEnabled", 1);
    let cr_hex = format!("0x{}", hex::encode(&cr_key));
    Mock::given(method("POST"))
        .and(body_partial_json(json!({
            "method": "state_getStorage",
            "params": [cr_hex]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "jsonrpc": "2.0", "id": 1,
            "result": "0x00"
        })))
        .mount(&server)
        .await;

    // Mock runtime version with mismatched spec_version (441 instead of 440)
    Mock::given(method("POST"))
        .and(body_partial_json(
            json!({"method": "state_getRuntimeVersion"}),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "jsonrpc": "2.0", "id": 1,
            "result": {
                "specName": "node-subtensor",
                "specVersion": 441,
                "transactionVersion": 1
            }
        })))
        .mount(&server)
        .await;

    let uri = server.uri();
    let result = tokio::task::spawn_blocking(move || {
        let client = LiveChainClient::connect(&uri)?;
        client.set_weights(1, vec![0], vec![100], 0)
    })
    .await
    .expect("spawn_blocking");

    let err = result.expect_err("must refuse");
    match err {
        ChainError::Other(msg) => {
            assert!(msg.contains("spec_version mismatch"), "msg: {msg}");
        }
        other => panic!("expected Other, got {other}"),
    }
}

#[tokio::test]
async fn submit_timelocked_rejected_when_cr_disabled() {
    let server = MockServer::start().await;

    // Mock CommitRevealWeightsEnabled(1) → false (0x00)
    let cr_key = storage_map_key_u16("SubtensorModule", "CommitRevealWeightsEnabled", 1);
    let cr_hex = format!("0x{}", hex::encode(&cr_key));
    Mock::given(method("POST"))
        .and(body_partial_json(json!({
            "method": "state_getStorage",
            "params": [cr_hex]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "jsonrpc": "2.0", "id": 1,
            "result": "0x00"
        })))
        .mount(&server)
        .await;

    let uri = server.uri();
    let payload = WeightsTlockPayload {
        hotkey: vec![0xAA; 32],
        uids: vec![0],
        values: vec![100],
        version_key: 0,
    };
    let result = tokio::task::spawn_blocking(move || {
        let client = LiveChainClient::connect(&uri)?;
        client.submit_timelocked_weights(0, payload, 99)
    })
    .await
    .expect("spawn_blocking");

    let err = result.expect_err("must refuse");
    assert!(
        matches!(
            err,
            ChainError::CommitRevealDisabled {
                alternate: "set_weights"
            }
        ),
        "expected CommitRevealDisabled, got {err}"
    );
}

// ---------------------------------------------------------------------------
// wiremock: metagraph_at
// ---------------------------------------------------------------------------

#[tokio::test]
async fn mock_metagraph_at() {
    let server = MockServer::start().await;

    // Keys(1) → Vec<Vec<u8>> with two 32-byte hotkeys
    let keys_data: Vec<Vec<u8>> = vec![vec![0xAA; 32], vec![0xBB; 32]];
    let keys_hex = format!("0x{}", hex::encode(keys_data.encode()));

    let keys_key = storage_map_key_u16("SubtensorModule", "Keys", 1);
    let keys_key_hex = format!("0x{}", hex::encode(&keys_key));

    Mock::given(method("POST"))
        .and(body_partial_json(json!({
            "method": "state_getStorage",
            "params": [keys_key_hex]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "jsonrpc": "2.0", "id": 1,
            "result": keys_hex
        })))
        .mount(&server)
        .await;

    // SubnetOwnerHotkey(1) → 32 raw bytes
    let owner_key = storage_map_key_u16("SubtensorModule", "SubnetOwnerHotkey", 1);
    let owner_key_hex = format!("0x{}", hex::encode(&owner_key));
    let owner_hex = format!("0x{}", hex::encode([0xCC_u8; 32]));

    Mock::given(method("POST"))
        .and(body_partial_json(json!({
            "method": "state_getStorage",
            "params": [owner_key_hex]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "jsonrpc": "2.0", "id": 1,
            "result": owner_hex
        })))
        .mount(&server)
        .await;

    let uri = server.uri();
    let mg = tokio::task::spawn_blocking(move || {
        let client = LiveChainClient::connect(&uri)?;
        let hash = [0_u8; 32];
        client.metagraph_at(&hash)
    })
    .await
    .expect("spawn_blocking")
    .expect("metagraph");
    assert_eq!(mg.netuid, 1);
    assert_eq!(mg.hotkeys.len(), 2);
    assert_eq!(mg.hotkeys[0], vec![0xAA; 32]);
    assert_eq!(mg.hotkeys[1], vec![0xBB; 32]);
    assert_eq!(mg.owner_hotkey, vec![0xCC; 32]);
}

// ---------------------------------------------------------------------------
// Live testnet (ignored — requires network)
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore = "requires network access to finney testnet"]
async fn testnet_current_block_positive() {
    let uri = "wss://test.finney.opentensor.ai:443".to_owned();
    let block = tokio::task::spawn_blocking(move || {
        let client = LiveChainClient::connect(&uri)?;
        client.current_block()
    })
    .await
    .expect("spawn_blocking")
    .expect("current_block");
    assert!(block > 0, "expected tip > 0, got {block}");
}
