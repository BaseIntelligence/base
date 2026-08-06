#![allow(clippy::expect_used, clippy::unwrap_used)]
//! Unit tests for chain-live (wiremock + pure fixture tests).

use crate::{
    commit_timelocked_call, decode_axon_info, decode_bool, decode_double_map_account_k2,
    decode_double_map_k2, decode_hotkey, decode_metagraph, decode_u16, decode_u64,
    decode_vec_vec_u8, serve_axon_call, set_weights_call, storage_double_map_key_u16_account,
    storage_double_map_key_u16_u16, storage_double_map_prefix_u16, storage_key,
    storage_map_key_twox64, storage_map_key_u16, ChainClient, ChainError, Era, LiveChainClient,
    ServeAxonParams, WeightsTlockPayload,
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

/// Subtensor per-netuid maps use the `Identity` hasher: no hash prefix.
///
/// Locked against a live `state_getKeysPaged` observation on testnet, where
/// every `SubnetOwnerHotkey` / `Tempo` / `LastEpochBlock` key suffix is exactly
/// the 2-byte LE netuid.
#[test]
fn storage_map_key_u16_uses_identity_hasher() {
    let k = storage_map_key_u16("SubtensorModule", "Tempo", 1u16);
    // Twox128(pallet) ++ Twox128(item) ++ netuid_le == 16 + 16 + 2
    assert_eq!(k.len(), 34);
    assert_eq!(&k[32..], &[0x01, 0x00]);
    // Prefix must still be the plain pallet/item pair.
    assert_eq!(&k[..32], &storage_key("SubtensorModule", "Tempo")[..]);
}

/// Known-good testnet key for `SubnetOwnerHotkey(541)`, captured from a live
/// `state_getKeysPaged` page. Guards the whole prefix, not just the suffix.
#[test]
fn storage_map_key_matches_live_testnet_key() {
    let k = storage_map_key_u16("SubtensorModule", "SubnetOwnerHotkey", 541);
    assert_eq!(hex::encode(&k[32..]), "1d02", "541 LE == 0x021d");
    assert_eq!(k.len(), 34);
}

#[test]
fn storage_double_map_key_layout() {
    let prefix = storage_double_map_prefix_u16("SubtensorModule", "Keys", 541);
    assert_eq!(prefix.len(), 34);
    let full = storage_double_map_key_u16_u16("SubtensorModule", "Keys", 541, 2);
    // 16 + 16 + 2 + 2 == 36, matching the live 4-byte suffix `1d020200`.
    assert_eq!(full.len(), 36);
    assert_eq!(hex::encode(&full[32..]), "1d020200");
    assert!(full.starts_with(&prefix), "full key must extend the prefix");
    assert_eq!(decode_double_map_k2(&full).unwrap(), 2);
}

#[test]
fn decode_double_map_k2_rejects_short_key() {
    assert!(decode_double_map_k2(&[0_u8; 34]).is_err());
    assert!(decode_double_map_k2(&[]).is_err());
}

#[test]
fn storage_map_key_twox64_still_available_for_other_pallets() {
    let key = [0xAB_u8; 4];
    let k = storage_map_key_twox64("SomePallet", "SomeItem", &key);
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
    // commit blob is opaque BoundedVec bytes (encrypted in production).
    let commit = [0xABu8; 4];
    let call = commit_timelocked_call(100, 0, &commit, 99, 4);

    // 0x07 pallet, 0x76 call 118, netuid=100 LE, mecid=0,
    // Compact(4)=0x10 + 4 commit bytes, reveal_round=99 LE, version=4 LE.
    assert_eq!(call[0], 0x07);
    assert_eq!(call[1], 0x76);
    assert_eq!(&call[2..4], &[100, 0]); // netuid u16 LE
    assert_eq!(call[4], 0x00); // mecid
    assert_eq!(call[5], 0x10); // Compact(4)
    assert_eq!(&call[6..10], &[0xAB; 4]);
    assert_eq!(
        &call[10..18],
        &[0x63, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
    ); // reveal_round=99
    assert_eq!(&call[18..20], &[0x04, 0x00]); // commit_reveal_version=4
    assert_eq!(call.len(), 20);
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
        443,
        1,
        1,
        &[0, 1],
        &[100, 200],
        0,
    )
    .unwrap();

    // 0x84 (version) + 0x00 (MultiAddress::Id) + 32 (pubkey) + 0x01 (Sr25519) + 64 (sig)
    // + 0x00 (era) + 0x00 (nonce) + 0x00 (tip) + 0x00 (metadata-hash mode) + 22 (call) = 125
    assert_eq!(ext.len(), 125);
    assert_eq!(ext[0], 0x84);
    assert_eq!(ext[1], 0x00);
    assert_eq!(ext[34], 0x01); // MultiSignature::Sr25519

    // Verify public key is deterministic
    let pubkey = crate::derive_public_key(&key).unwrap();
    assert_eq!(&ext[2..34], &pubkey);

    // Verify era + nonce + tip + CheckMetadataHash mode + call suffix
    assert_eq!(ext[99], 0x00); // Immortal era
    assert_eq!(ext[100], 0x00); // Compact(0) nonce
    assert_eq!(ext[101], 0x00); // Compact(0) tip
    assert_eq!(ext[102], 0x00); // CheckMetadataHash::Disabled
    let expected_call = set_weights_call(1, &[0, 1], &[100, 200], 0);
    assert_eq!(&ext[103..], &expected_call[..]);
}

#[test]
fn build_and_sign_commit_timelocked_structure() {
    let key = test_secret();
    let genesis = [0x22_u8; 32];
    let commit = [0xABu8; 4];
    let ext = crate::build_and_sign_commit_timelocked(
        &key,
        5,
        &Era::Immortal,
        &genesis,
        &genesis,
        443,
        1,
        100,
        0,
        &commit,
        99,
        4,
    )
    .unwrap();

    // 1+1+32+1+64 + era + nonce(5=0x14) + tip + mode + 20(call) = 123
    assert_eq!(ext.len(), 123);
    assert_eq!(ext[0], 0x84);
    assert_eq!(ext[1], 0x00);
    assert_eq!(ext[34], 0x01); // Sr25519
    assert_eq!(ext[99], 0x00); // Immortal era
    assert_eq!(ext[100], 0x14); // Compact(5) nonce
    assert_eq!(ext[101], 0x00); // Compact(0) tip
    assert_eq!(ext[102], 0x00); // CheckMetadataHash::Disabled

    let expected_call = commit_timelocked_call(100, 0, &commit, 99, 4);
    assert_eq!(&ext[103..], &expected_call[..]);
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
                "specVersion": 443,
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
    assert_eq!(rt.spec_version, 443);
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

    // Mock runtime version with mismatched spec_version (441 instead of 443)
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

    // Keys is a double map: one entry per (netuid, uid), enumerated by prefix.
    let uid0_key = storage_double_map_key_u16_u16("SubtensorModule", "Keys", 1, 0);
    let uid1_key = storage_double_map_key_u16_u16("SubtensorModule", "Keys", 1, 1);
    let uid0_hex = format!("0x{}", hex::encode(&uid0_key));
    let uid1_hex = format!("0x{}", hex::encode(&uid1_key));

    Mock::given(method("POST"))
        .and(body_partial_json(json!({"method": "state_getKeysPaged"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "jsonrpc": "2.0", "id": 1,
            "result": [uid1_hex.clone(), uid0_hex.clone()]
        })))
        .mount(&server)
        .await;

    // Values come back out of uid order to prove we sort by uid, not by page order.
    Mock::given(method("POST"))
        .and(body_partial_json(json!({"method": "state_queryStorageAt"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "jsonrpc": "2.0", "id": 1,
            "result": [{
                "block": format!("0x{}", hex::encode([0u8; 32])),
                "changes": [
                    [uid1_hex, format!("0x{}", hex::encode([0xBB_u8; 32]))],
                    [uid0_hex, format!("0x{}", hex::encode([0xAA_u8; 32]))]
                ]
            }]
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
// Axons: storage key, value decode, serve_axon call
// ---------------------------------------------------------------------------

/// A real netuid-1 axon entry, captured from testnet via `substrateinterface`.
///
/// `Axons(1, 5Gui9iTEmkGcDUqYMM8V2jhGTuudoZYGKNM3FZSaG53rCJTa)`.
const AXON_HOTKEY_HEX: &str = "d650f2b10830144abe932c82d01bdc359fb6461c4d9ed6527b83aa197d75e90f";
const AXON_KEY_HEX: &str = "658faa385070e074c85bf6b568cf0555b4e0c7b1d5f74994fcd58c28ab601fec\
01000462dd252e492bb60bcd4be55cc02581\
d650f2b10830144abe932c82d01bdc359fb6461c4d9ed6527b83aa197d75e90f";
const AXON_VALUE_HEX: &str = "d4ce3e0000000000285889001de59add0000000000000000000000009b1f04040000";

fn axon_hotkey() -> [u8; 32] {
    let bytes = hex::decode(AXON_HOTKEY_HEX).expect("hotkey hex");
    bytes.try_into().expect("32 bytes")
}

/// `Axons` uses `Blake2_128Concat` on the hotkey, unlike the all-`Identity`
/// per-netuid maps. Locked to the key `substrateinterface` produced for the
/// same `(netuid, hotkey)` pair against live testnet.
#[test]
fn axons_storage_key_matches_substrateinterface() {
    let key = storage_double_map_key_u16_account("SubtensorModule", "Axons", 1, &axon_hotkey());
    assert_eq!(hex::encode(&key), AXON_KEY_HEX);
    // Twox128 ++ Twox128 ++ netuid ++ blake2_128 ++ AccountId32.
    assert_eq!(key.len(), 16 + 16 + 2 + 16 + 32);
    // The enumeration prefix must be a prefix of the full key.
    let prefix = storage_double_map_prefix_u16("SubtensorModule", "Axons", 1);
    assert_eq!(&key[..prefix.len()], &prefix[..]);
}

#[test]
fn axons_key_round_trips_the_hotkey() {
    let hk = axon_hotkey();
    let key = storage_double_map_key_u16_account("SubtensorModule", "Axons", 1, &hk);
    assert_eq!(decode_double_map_account_k2(&key).expect("k2"), hk.to_vec());
    assert!(decode_double_map_account_k2(&key[..40]).is_err());
}

#[test]
fn axon_info_decodes_live_testnet_value() {
    let raw = hex::decode(AXON_VALUE_HEX).expect("value hex");
    let axon = decode_axon_info(&raw).expect("decode AxonInfo");
    assert_eq!(axon.block, 4_116_180);
    assert_eq!(axon.version, 9_001_000);
    assert_eq!(axon.ip, 3_717_915_933);
    assert_eq!(axon.port, 8091);
    assert_eq!(axon.ip_type, 4);
    assert_eq!(axon.protocol, 4);
    assert_eq!(axon.placeholder1, 0);
    assert_eq!(axon.placeholder2, 0);
    assert_eq!(
        axon.base_url().as_deref(),
        Some("http://221.154.229.29:8091")
    );
}

/// `serve_axon` is `SubtensorModule` (pallet 7) call index 4, args in the order
/// `netuid, version, ip, port, ip_type, protocol, placeholder1, placeholder2`
/// (testnet metadata `get_metadata_call_function('SubtensorModule','serve_axon')`).
#[test]
fn serve_axon_call_encodes_pallet_7_call_4() {
    let params = ServeAxonParams::ipv4(541, 9_012_002, std::net::Ipv4Addr::new(1, 2, 3, 4), 8091);
    let call = serve_axon_call(&params);
    let mut want = vec![7_u8, 4];
    541_u16.encode_to(&mut want);
    9_012_002_u32.encode_to(&mut want);
    u128::from(u32::from_be_bytes([1, 2, 3, 4])).encode_to(&mut want);
    8091_u16.encode_to(&mut want);
    want.extend_from_slice(&[4, 0, 0, 0]);
    assert_eq!(call, want);
    assert_eq!(call.len(), 2 + 2 + 4 + 16 + 2 + 4);
}

#[test]
fn serve_axon_extrinsic_is_signed_and_carries_the_call() {
    let params = ServeAxonParams::ipv4(541, 9_012_002, std::net::Ipv4Addr::new(1, 2, 3, 4), 8091);
    let ext = crate::build_and_sign_serve_axon(
        &[7_u8; 32],
        0,
        &Era::Immortal,
        &[0x11; 32],
        &[0x11; 32],
        443,
        1,
        &params,
    )
    .expect("sign");
    assert_eq!(ext[0], 0x84, "V4 signed extrinsic");
    let call = serve_axon_call(&params);
    assert!(
        ext.ends_with(&call),
        "signed extrinsic must end with the call bytes"
    );
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

/// Testnet endpoint and the netuid we actually own there.
const TESTNET: &str = "wss://test.finney.opentensor.ai:443";
const OUR_NETUID: u16 = 541;
/// The on-chain `SubnetOwnerHotkey` for netuid 541: the `base-owner` wallet,
/// SS58 `5CfjVGG7DaagMUuABNnqQJygLV2xtn3AQ7LnPeFoc5gVK9xo`.
const OUR_OWNER_HEX: &str = "1ab7145525140560cb64e1e89fae8258e813ba12d9c20faaeabc17f95ba5fe7e";

/// Every read path against the real chain in one pass.
///
/// This is the regression guard for the `Twox64Concat`-vs-`Identity` hasher
/// bug: with the wrong hasher every one of these reads returns `None` and the
/// test fails on the very first assertion.
struct LiveReads {
    owner: Vec<u8>,
    tempo: u64,
    crv: u16,
    block_time: u64,
    last_epoch: u64,
    reveal: u64,
    mg: chain::Metagraph,
}

#[tokio::test]
#[ignore = "requires network access to finney testnet"]
async fn testnet_all_reads_resolve() {
    let reads = tokio::task::spawn_blocking(|| -> Result<LiveReads, ChainError> {
        let mut client = LiveChainClient::connect(TESTNET)?;
        client.set_netuid(OUR_NETUID);
        // commit_reveal_enabled must not error even though the key is absent.
        let _cr = client.commit_reveal_enabled(OUR_NETUID)?;
        Ok(LiveReads {
            owner: client.subnet_owner_hotkey(OUR_NETUID)?,
            tempo: client.tempo(OUR_NETUID)?,
            crv: client.commit_reveal_version(OUR_NETUID)?,
            block_time: client.block_time()?,
            last_epoch: client.last_epoch_block(OUR_NETUID)?,
            reveal: client.reveal_period_epochs(OUR_NETUID)?,
            mg: client.metagraph_at(&[0_u8; 32])?,
        })
    })
    .await
    .expect("spawn_blocking")
    .expect("all reads must resolve against the live chain");

    let LiveReads {
        owner,
        tempo,
        crv,
        block_time,
        last_epoch,
        reveal,
        mg,
    } = reads;

    assert_eq!(owner.len(), 32, "owner hotkey must be a 32-byte AccountId");
    assert!(tempo > 0, "tempo must be positive, got {tempo}");
    assert_eq!(crv, 4, "testnet commit-reveal version should be CRV4");
    assert_eq!(block_time, 12_000, "Aura slot duration is 12s in ms");
    assert!(last_epoch > 0, "last epoch block must be set");
    assert!(reveal > 0, "reveal period must be positive");

    // The metagraph must come back non-empty and owner-consistent. With the old
    // single-key `Vec<Vec<u8>>` decode this was always an error.
    assert_eq!(mg.netuid, OUR_NETUID);
    assert!(
        !mg.hotkeys.is_empty(),
        "metagraph must enumerate at least one neuron"
    );
    for hk in &mg.hotkeys {
        assert_eq!(hk.len(), 32, "each hotkey must be a 32-byte AccountId");
    }
    assert_eq!(mg.owner_hotkey, owner, "metagraph owner must match");
}

/// Netuid 541 is ours; assert the owner hotkey has not silently changed.
#[tokio::test]
#[ignore = "requires network access to finney testnet"]
async fn testnet_541_owner_is_base_owner() {
    let owner = tokio::task::spawn_blocking(|| {
        LiveChainClient::connect(TESTNET).and_then(|c| c.subnet_owner_hotkey(OUR_NETUID))
    })
    .await
    .expect("spawn_blocking")
    .expect("owner");
    assert_eq!(
        hex::encode(&owner),
        OUR_OWNER_HEX,
        "netuid {OUR_NETUID} owner changed"
    );
}

/// A subnet that does not exist must be a clean error, never a silent default.
#[tokio::test]
#[ignore = "requires network access to finney testnet"]
async fn testnet_absent_subnet_errors() {
    let err = tokio::task::spawn_blocking(|| {
        LiveChainClient::connect(TESTNET).and_then(|c| c.subnet_owner_hotkey(60000))
    })
    .await
    .expect("spawn_blocking")
    .expect_err("absent subnet must error");
    let msg = err.to_string();
    assert!(msg.contains("SubnetOwnerHotkey"), "unexpected: {msg}");
}

/// Netuid 1 has thousands of served axons; a wrong hasher yields an empty set.
#[tokio::test]
#[ignore = "requires network access to finney testnet"]
async fn testnet_netuid1_axons_enumerate() {
    let axons = tokio::task::spawn_blocking(|| {
        LiveChainClient::connect(TESTNET).and_then(|c| c.enumerate_axons(1))
    })
    .await
    .expect("spawn_blocking")
    .expect("enumerate axons");
    assert!(!axons.is_empty(), "netuid 1 must have served axons");
    let reachable = axons.iter().filter(|(_, a)| a.base_url().is_some()).count();
    println!(
        "netuid 1: {} axons, {reachable} with a reachable base_url",
        axons.len()
    );
    for (hk, axon) in axons.iter().take(3) {
        println!("  {} -> {:?} {:?}", hex::encode(hk), axon, axon.base_url());
    }
    assert!(reachable > 0, "at least one axon must publish ip+port");
    for (hk, axon) in &axons {
        assert_eq!(hk.len(), 32, "axon key must be a 32-byte AccountId");
        assert!(
            axon.ip_type == 4 || axon.ip_type == 6,
            "unexpected ip_type {} for {}",
            axon.ip_type,
            hex::encode(hk)
        );
    }
}

/// Our subnet has neurons but nobody has served an axon yet: empty, not an error.
#[tokio::test]
#[ignore = "requires network access to finney testnet"]
async fn testnet_541_axons_empty_without_error() {
    let axons = tokio::task::spawn_blocking(|| {
        LiveChainClient::connect(TESTNET).and_then(|c| c.enumerate_axons(OUR_NETUID))
    })
    .await
    .expect("spawn_blocking")
    .expect("enumerate axons must not error on an empty map");
    println!("netuid {OUR_NETUID}: {} axons {axons:?}", axons.len());
    for (hk, axon) in &axons {
        assert_eq!(hk.len(), 32);
        assert!(axon.ip_type == 4 || axon.ip_type == 6);
    }
}

/// The single-key read must return exactly what `substrateinterface` decoded
/// for the same `(netuid, hotkey)` — the fixture in [`AXON_VALUE_HEX`].
#[tokio::test]
#[ignore = "requires network access to finney testnet"]
async fn testnet_single_axon_read_matches_reference() {
    let hk = axon_hotkey();
    let live = tokio::task::spawn_blocking(move || {
        LiveChainClient::connect(TESTNET).and_then(|c| c.read_axon(1, &hk))
    })
    .await
    .expect("spawn_blocking")
    .expect("read_axon")
    .expect("hotkey must have a served axon");
    let reference = decode_axon_info(&hex::decode(AXON_VALUE_HEX).expect("hex")).expect("decode");
    println!("live      : {live:?} url={:?}", live.base_url());
    println!("reference : {reference:?} url={:?}", reference.base_url());
    // `block`/`version` advance whenever the miner re-serves; the endpoint is
    // the part the operator dispatches to, so compare that exactly.
    assert_eq!(live.ip, reference.ip);
    assert_eq!(live.port, reference.port);
    assert_eq!(live.ip_type, reference.ip_type);
    assert_eq!(live.base_url(), reference.base_url());

    // An unregistered hotkey on the same netuid must be absent, not defaulted.
    let absent = tokio::task::spawn_blocking(|| {
        LiveChainClient::connect(TESTNET).and_then(|c| c.read_axon(1, &[0xEE_u8; 32]))
    })
    .await
    .expect("spawn_blocking")
    .expect("read_axon");
    assert_eq!(absent, None);
}

#[test]
fn parse_commit_reveal_enabled_v3_bool() {
    // Minimal synthetic fragment: name + Bool(true/false).
    let mut true_bytes = b"commit_reveal_weights_enabled".to_vec();
    true_bytes.extend_from_slice(&[0, 1]);
    assert_eq!(
        crate::parse_commit_reveal_enabled_v3(&true_bytes),
        Some(true)
    );
    let mut false_bytes = b"commit_reveal_weights_enabled".to_vec();
    false_bytes.extend_from_slice(&[0, 0]);
    assert_eq!(
        crate::parse_commit_reveal_enabled_v3(&false_bytes),
        Some(false)
    );
    assert_eq!(crate::parse_commit_reveal_enabled_v3(b"nope"), None);
}

// ---------------------------------------------------------------------------
// wiremock: weight-submit dispatch confirmation via LastUpdate
// ---------------------------------------------------------------------------

async fn mount_storage(server: &MockServer, key: &[u8], value: serde_json::Value) {
    let key_hex = format!("0x{}", hex::encode(key));
    Mock::given(method("POST"))
        .and(body_partial_json(json!({
            "method": "state_getStorage",
            "params": [key_hex]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "jsonrpc": "2.0", "id": 1,
            "result": value
        })))
        .mount(server)
        .await;
}

async fn mount_tip(server: &MockServer, tip: u64) {
    Mock::given(method("POST"))
        .and(body_partial_json(json!({"method": "chain_getHeader"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "jsonrpc": "2.0", "id": 1,
            "result": { "number": format!("0x{tip:x}") }
        })))
        .mount(server)
        .await;
}

fn uids_key(netuid: u16, hotkey: &[u8; 32]) -> Vec<u8> {
    storage_double_map_key_u16_account("SubtensorModule", "Uids", netuid, hotkey)
}

fn last_update_key(netuid: u16) -> Vec<u8> {
    storage_map_key_u16("SubtensorModule", "LastUpdate", netuid)
}

fn vec_u64_hex(v: &[u64]) -> String {
    format!("0x{}", hex::encode(v.encode()))
}

#[tokio::test]
async fn last_weight_update_reads_uid_entry() {
    let server = MockServer::start().await;
    let hotkey = [0xAA_u8; 32];
    mount_storage(&server, &uids_key(1, &hotkey), json!("0x0200")).await;
    mount_storage(
        &server,
        &last_update_key(1),
        json!(vec_u64_hex(&[10, 20, 30])),
    )
    .await;

    let uri = server.uri();
    let got = tokio::task::spawn_blocking(move || {
        LiveChainClient::connect(&uri)?.last_weight_update(1, &[0xAA_u8; 32])
    })
    .await
    .expect("spawn_blocking")
    .expect("read");
    assert_eq!(got, Some(30));
}

#[tokio::test]
async fn last_weight_update_none_when_unregistered() {
    let server = MockServer::start().await;
    let hotkey = [0xEE_u8; 32];
    mount_storage(&server, &uids_key(1, &hotkey), serde_json::Value::Null).await;

    let uri = server.uri();
    let got = tokio::task::spawn_blocking(move || {
        LiveChainClient::connect(&uri)?.last_weight_update(1, &[0xEE_u8; 32])
    })
    .await
    .expect("spawn_blocking")
    .expect("read");
    assert_eq!(got, None);
}

#[tokio::test]
async fn confirm_weight_update_ok_when_last_update_advances() {
    let server = MockServer::start().await;
    let hotkey = [0xAA_u8; 32];
    mount_tip(&server, 1000).await;
    mount_storage(&server, &uids_key(1, &hotkey), json!("0x0000")).await;
    mount_storage(&server, &last_update_key(1), json!(vec_u64_hex(&[999]))).await;

    let uri = server.uri();
    tokio::task::spawn_blocking(move || {
        LiveChainClient::connect(&uri)?.confirm_weight_update(1, &[0xAA_u8; 32], Some(900), 999)
    })
    .await
    .expect("spawn_blocking")
    .expect("advanced LastUpdate must confirm");
}

#[tokio::test]
async fn confirm_weight_update_rate_limited_inside_window() {
    let server = MockServer::start().await;
    let hotkey = [0xAA_u8; 32];
    mount_tip(&server, 1000).await;
    mount_storage(&server, &uids_key(1, &hotkey), json!("0x0000")).await;
    mount_storage(&server, &last_update_key(1), json!(vec_u64_hex(&[950]))).await;
    // WeightsSetRateLimit absent → default 100.
    mount_storage(
        &server,
        &storage_map_key_u16("SubtensorModule", "WeightsSetRateLimit", 1),
        serde_json::Value::Null,
    )
    .await;

    let uri = server.uri();
    let err = tokio::task::spawn_blocking(move || {
        LiveChainClient::connect(&uri)?.confirm_weight_update(1, &[0xAA_u8; 32], Some(950), 996)
    })
    .await
    .expect("spawn_blocking")
    .expect_err("static LastUpdate inside the window must be RateLimited");
    assert_eq!(
        err,
        ChainError::RateLimited {
            retry_after_blocks: Some(50)
        }
    );
}

#[tokio::test]
async fn confirm_weight_update_unconfirmed_outside_window_is_transient() {
    let server = MockServer::start().await;
    let hotkey = [0xAA_u8; 32];
    mount_tip(&server, 2000).await;
    mount_storage(&server, &uids_key(1, &hotkey), json!("0x0000")).await;
    mount_storage(&server, &last_update_key(1), json!(vec_u64_hex(&[950]))).await;
    mount_storage(
        &server,
        &storage_map_key_u16("SubtensorModule", "WeightsSetRateLimit", 1),
        serde_json::Value::Null,
    )
    .await;

    let uri = server.uri();
    let err = tokio::task::spawn_blocking(move || {
        LiveChainClient::connect(&uri)?.confirm_weight_update(1, &[0xAA_u8; 32], Some(950), 1996)
    })
    .await
    .expect("spawn_blocking")
    .expect_err("static LastUpdate outside the window is not RateLimited");
    match err {
        ChainError::Other(msg) => assert!(msg.contains("unconfirmed"), "msg: {msg}"),
        other => panic!("expected Other, got {other}"),
    }
}

/// Full submit path: pool acceptance is not enough — the method must keep
/// reading `LastUpdate` until it advances, and only then report success.
#[tokio::test]
async fn submit_timelocked_ok_only_after_dispatch_confirmation() {
    let server = MockServer::start().await;
    let sk = [0x42_u8; 32];
    let pk = crate::derive_public_key(&sk).expect("pk");

    // CR on via sparse storage (hyperparams state_call is unmocked → falls back).
    mount_storage(
        &server,
        &storage_map_key_u16("SubtensorModule", "CommitRevealWeightsEnabled", 1),
        json!("0x01"),
    )
    .await;
    Mock::given(method("POST"))
        .and(body_partial_json(
            json!({"method": "state_getRuntimeVersion"}),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "jsonrpc": "2.0", "id": 1,
            "result": {
                "specName": "node-subtensor",
                "specVersion": 443,
                "transactionVersion": 1
            }
        })))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(body_partial_json(json!({"method": "chain_getBlockHash"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "jsonrpc": "2.0", "id": 1,
            "result": format!("0x{}", hex::encode([0x11_u8; 32]))
        })))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(body_partial_json(
            json!({"method": "system_accountNextIndex"}),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "jsonrpc": "2.0", "id": 1,
            "result": 0
        })))
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(body_partial_json(
            json!({"method": "author_submitExtrinsic"}),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "jsonrpc": "2.0", "id": 1,
            "result": format!("0x{}", hex::encode([0xEE_u8; 32]))
        })))
        .mount(&server)
        .await;
    mount_tip(&server, 500).await;
    mount_storage(&server, &uids_key(1, &pk), json!("0x0000")).await;
    // Pre-submit read sees the old block; confirmation reads see the advance.
    Mock::given(method("POST"))
        .and(body_partial_json(json!({
            "method": "state_getStorage",
            "params": [format!("0x{}", hex::encode(last_update_key(1)))]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "jsonrpc": "2.0", "id": 1,
            "result": vec_u64_hex(&[400])
        })))
        .up_to_n_times(1)
        .with_priority(1)
        .mount(&server)
        .await;
    Mock::given(method("POST"))
        .and(body_partial_json(json!({
            "method": "state_getStorage",
            "params": [format!("0x{}", hex::encode(last_update_key(1)))]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "jsonrpc": "2.0", "id": 1,
            "result": vec_u64_hex(&[501])
        })))
        .with_priority(5)
        .mount(&server)
        .await;

    let uri = server.uri();
    let payload = WeightsTlockPayload {
        hotkey: pk.to_vec(),
        uids: vec![0],
        values: vec![65535],
        version_key: 0,
    };
    tokio::task::spawn_blocking(move || {
        let mut client = LiveChainClient::connect(&uri)?;
        client.set_signing_key(sk);
        client.submit_timelocked_weights(0, payload, 99)
    })
    .await
    .expect("spawn_blocking")
    .expect("dispatch-confirmed submit must succeed");
}

// ---------------------------------------------------------------------------
// wiremock: ordered endpoint failover (BASE_CHAIN_ENDPOINTS)
// ---------------------------------------------------------------------------

/// Mount a `chain_getHeader` responder returning block 1000.
async fn mount_header_ok(server: &MockServer) {
    Mock::given(method("POST"))
        .and(body_partial_json(json!({"method": "chain_getHeader"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "jsonrpc": "2.0", "id": 1,
            "result": {"number": "0x3e8"}
        })))
        .mount(server)
        .await;
}

async fn request_count(server: &MockServer) -> usize {
    server
        .received_requests()
        .await
        .expect("request recording")
        .len()
}

#[tokio::test]
async fn failover_http_429_primary_cools_and_fallback_serves() {
    let bad = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(429).set_body_json(json!({
            "error": "Too many requests from this source.",
            "policy": "http_60s",
            "reason": "rate_exceeded",
            "retry_after_seconds": 60
        })))
        .mount(&bad)
        .await;
    let good = MockServer::start().await;
    mount_header_ok(&good).await;

    let uri = format!("{},{}", bad.uri(), good.uri());
    tokio::task::spawn_blocking(move || {
        let client = LiveChainClient::connect(&uri)?;
        assert_eq!(client.current_block()?, 1000);
        // Second call must skip the still-cooling primary entirely.
        assert_eq!(client.current_block()?, 1000);
        Ok::<_, ChainError>(())
    })
    .await
    .expect("spawn_blocking")
    .expect("failover to secondary");
    assert_eq!(request_count(&bad).await, 1, "primary cooled after 429");
    assert_eq!(request_count(&good).await, 2, "fallback served both calls");
}

#[tokio::test]
async fn failover_rpc_error_string_too_many_requests() {
    let bad = MockServer::start().await;
    // Finney entrypoint also returns HTTP 200 with a bare-string error payload.
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "jsonrpc": "2.0", "id": 1,
            "error": "Too many requests from this source."
        })))
        .mount(&bad)
        .await;
    let good = MockServer::start().await;
    mount_header_ok(&good).await;

    let uri = format!("{},{}", bad.uri(), good.uri());
    let block =
        tokio::task::spawn_blocking(move || LiveChainClient::connect(&uri)?.current_block())
            .await
            .expect("spawn_blocking")
            .expect("string rate-limit error must fail over");
    assert_eq!(block, 1000);
    assert_eq!(request_count(&good).await, 1);
}

#[tokio::test]
async fn failover_rpc_error_object_minus_32005() {
    let bad = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "jsonrpc": "2.0", "id": 1,
            "error": {"code": -32005, "message": "rate limit exceeded"}
        })))
        .mount(&bad)
        .await;
    let good = MockServer::start().await;
    mount_header_ok(&good).await;

    let uri = format!("{},{}", bad.uri(), good.uri());
    let block =
        tokio::task::spawn_blocking(move || LiveChainClient::connect(&uri)?.current_block())
            .await
            .expect("spawn_blocking")
            .expect("-32005 must fail over");
    assert_eq!(block, 1000);
    assert_eq!(request_count(&good).await, 1);
}

#[tokio::test]
async fn no_failover_on_request_level_error() {
    let bad = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "jsonrpc": "2.0", "id": 1,
            "error": {"code": -32602, "message": "Invalid params"}
        })))
        .mount(&bad)
        .await;
    let good = MockServer::start().await;
    mount_header_ok(&good).await;

    let uri = format!("{},{}", bad.uri(), good.uri());
    let err = tokio::task::spawn_blocking(move || LiveChainClient::connect(&uri)?.current_block())
        .await
        .expect("spawn_blocking")
        .expect_err("request-level error must not fail over");
    assert!(err.to_string().contains("Invalid params"), "err: {err}");
    assert_eq!(
        request_count(&good).await,
        0,
        "fallback untouched on request-level error"
    );
}

#[tokio::test]
async fn all_endpoints_faulting_returns_last_error() {
    let bad = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(429).set_body_json(json!({
            "error": "Too many requests from this source."
        })))
        .mount(&bad)
        .await;
    let uri = bad.uri();
    let err = tokio::task::spawn_blocking(move || LiveChainClient::connect(&uri)?.current_block())
        .await
        .expect("spawn_blocking")
        .expect_err("single faulting endpoint must error");
    assert!(err.to_string().contains("http 429"), "err: {err}");
}

#[test]
fn connect_rejects_empty_endpoint_list() {
    for raw in ["", " , ", ","] {
        let err = crate::LiveChainRpc::connect(raw).expect_err("empty list must fail");
        assert!(err.to_string().contains("no chain endpoint"), "err: {err}");
    }
}

// ---------------------------------------------------------------------------
// Pre-submit rate-limit window check (no doomed extrinsic, no confirm wait)
// ---------------------------------------------------------------------------

/// Mount the common signing-path mocks (runtime version, genesis hash, nonce).
async fn mount_signing_path(server: &MockServer) {
    Mock::given(method("POST"))
        .and(body_partial_json(
            json!({"method": "state_getRuntimeVersion"}),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "jsonrpc": "2.0", "id": 1,
            "result": {"specName": "node-subtensor", "specVersion": 443, "transactionVersion": 1}
        })))
        .mount(server)
        .await;
    Mock::given(method("POST"))
        .and(body_partial_json(json!({"method": "chain_getBlockHash"})))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "jsonrpc": "2.0", "id": 1,
            "result": format!("0x{}", hex::encode([0x11_u8; 32]))
        })))
        .mount(server)
        .await;
    Mock::given(method("POST"))
        .and(body_partial_json(
            json!({"method": "system_accountNextIndex"}),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "jsonrpc": "2.0", "id": 1,
            "result": 0
        })))
        .mount(server)
        .await;
}

async fn assert_no_extrinsic_submitted(server: &MockServer) {
    let reqs = server.received_requests().await.expect("requests");
    let submits = reqs
        .iter()
        .filter(|r| String::from_utf8_lossy(&r.body).contains("author_submitExtrinsic"))
        .count();
    assert_eq!(
        submits, 0,
        "no extrinsic may reach the pool inside the window"
    );
}

#[tokio::test]
async fn submit_timelocked_skips_pool_inside_rate_limit_window() {
    let server = MockServer::start().await;
    let sk = [0x42_u8; 32];
    let pk = crate::derive_public_key(&sk).expect("pk");
    // CR on via sparse storage (hyperparams state_call unmocked → falls back).
    mount_storage(
        &server,
        &storage_map_key_u16("SubtensorModule", "CommitRevealWeightsEnabled", 1),
        json!("0x01"),
    )
    .await;
    mount_signing_path(&server).await;
    mount_tip(&server, 1000).await;
    mount_storage(&server, &uids_key(1, &pk), json!("0x0000")).await;
    // Last update at block 990, tip 1000 → 10 blocks into the 100-block window.
    mount_storage(&server, &last_update_key(1), json!(vec_u64_hex(&[990]))).await;
    mount_storage(
        &server,
        &storage_map_key_u16("SubtensorModule", "WeightsSetRateLimit", 1),
        serde_json::Value::Null,
    )
    .await;

    let uri = server.uri();
    let payload = WeightsTlockPayload {
        hotkey: pk.to_vec(),
        uids: vec![0],
        values: vec![65535],
        version_key: 0,
    };
    let err = tokio::task::spawn_blocking(move || {
        let mut client = LiveChainClient::connect(&uri)?;
        client.set_signing_key(sk);
        client.submit_timelocked_weights(0, payload, 99)
    })
    .await
    .expect("spawn_blocking")
    .expect_err("inside the window must fail fast with RateLimited");
    assert_eq!(
        err,
        ChainError::RateLimited {
            retry_after_blocks: Some(90)
        }
    );
    assert_no_extrinsic_submitted(&server).await;
}

#[tokio::test]
async fn set_weights_skips_pool_inside_rate_limit_window() {
    let server = MockServer::start().await;
    let sk = [0x42_u8; 32];
    let pk = crate::derive_public_key(&sk).expect("pk");
    // CR off via sparse storage → set_weights path allowed.
    mount_storage(
        &server,
        &storage_map_key_u16("SubtensorModule", "CommitRevealWeightsEnabled", 1),
        json!("0x00"),
    )
    .await;
    mount_signing_path(&server).await;
    mount_tip(&server, 1000).await;
    mount_storage(&server, &uids_key(1, &pk), json!("0x0000")).await;
    mount_storage(&server, &last_update_key(1), json!(vec_u64_hex(&[990]))).await;
    mount_storage(
        &server,
        &storage_map_key_u16("SubtensorModule", "WeightsSetRateLimit", 1),
        serde_json::Value::Null,
    )
    .await;

    let uri = server.uri();
    let err = tokio::task::spawn_blocking(move || {
        let mut client = LiveChainClient::connect(&uri)?;
        client.set_signing_key(sk);
        client.set_weights(1, vec![0], vec![65535], 0)
    })
    .await
    .expect("spawn_blocking")
    .expect_err("inside the window must fail fast with RateLimited");
    assert_eq!(
        err,
        ChainError::RateLimited {
            retry_after_blocks: Some(90)
        }
    );
    assert_no_extrinsic_submitted(&server).await;
}
