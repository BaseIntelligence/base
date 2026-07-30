//! RTMR3 event-log replay for dstack / Phala TDX attestation.
//!
//! Digest and extend match BASE `phala_quote.replay_rtmr3` / cc-eventlog:
//!
//! ```text
//! digest = SHA384( event_type_le32 ‖ b":" ‖ name_utf8 ‖ b":" ‖ payload )
//! mr     = SHA384( mr ‖ digest )   // mr starts as 48 zero bytes
//! ```
//!
//! Only IMR index 3 (RTMR3) is folded. Unknown `event_type` values are
//! surfaced as errors and never silently skipped.

#![forbid(unsafe_code)]

use sha2::{Digest, Sha384};
use thiserror::Error;

/// SHA-384 digest / RTMR register length in bytes.
pub const REGISTER_LEN: usize = 48;

/// dstack runtime event type (RTMR3 app events). Not a TCG-defined value.
pub const DSTACK_RUNTIME_EVENT_TYPE: u32 = 0x0800_0001;

/// RTMR index whose event log binds the app compose (RTMR3).
pub const APP_IMR: u32 = 3;

/// Event name carrying the normalized compose hash payload.
pub const COMPOSE_HASH_EVENT: &str = "compose-hash";

/// Event name carrying the KMS key-provider identity payload.
pub const KEY_PROVIDER_EVENT: &str = "key-provider";

/// One entry from a dstack / cc-eventlog RTMR event log.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Event {
    /// PCR/IMR index. Only [`APP_IMR`] (3) is applied to RTMR3.
    pub imr: u32,
    /// Event type; must be [`DSTACK_RUNTIME_EVENT_TYPE`] for RTMR3 app events.
    pub event_type: u32,
    /// Event name (UTF-8), e.g. [`COMPOSE_HASH_EVENT`].
    pub name: String,
    /// Raw event payload bytes (already decoded from hex if sourced as hex).
    pub payload: Vec<u8>,
    /// Optional logged digest; when present must equal the recomputed digest.
    pub digest: Option<[u8; REGISTER_LEN]>,
}

/// Result of replaying an event log into RTMR3.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Replay {
    /// Final RTMR3 register (48 bytes).
    pub rtmr3: [u8; REGISTER_LEN],
    /// Payload of the last `compose-hash` event, if any.
    pub compose_hash: Option<Vec<u8>>,
    /// Payload of the last `key-provider` event, if any.
    pub key_provider: Option<Vec<u8>>,
}

impl Replay {
    /// Lowercase hex encoding of [`Self::rtmr3`].
    #[must_use]
    pub fn rtmr3_hex(&self) -> String {
        hex_encode(&self.rtmr3)
    }

    /// Lowercase hex encoding of the compose-hash payload, if present.
    #[must_use]
    pub fn compose_hash_hex(&self) -> Option<String> {
        self.compose_hash.as_ref().map(|p| hex_encode(p))
    }
}

/// Failures while replaying an RTMR3 event log (fail closed).
#[derive(Debug, Error, Eq, PartialEq)]
pub enum ReplayError {
    /// An RTMR3 event used an `event_type` this crate does not accept.
    #[error("unknown RTMR3 event_type 0x{event_type:08x} (not silently skipped)")]
    UnknownEventType { event_type: u32 },
    /// Logged digest hex/bytes do not match the digest recomputed from contents.
    #[error("event '{name}' digest does not match its payload")]
    DigestMismatch { name: String },
}

/// SHA-384 digest of one dstack runtime event (cc-eventlog format).
///
/// `SHA384( event_type(le32) ‖ b":" ‖ event_name ‖ b":" ‖ payload )`.
#[must_use]
pub fn event_digest(event_type: u32, name: &str, payload: &[u8]) -> [u8; REGISTER_LEN] {
    let mut hasher = Sha384::new();
    hasher.update(event_type.to_le_bytes());
    hasher.update(b":");
    hasher.update(name.as_bytes());
    hasher.update(b":");
    hasher.update(payload);
    let out = hasher.finalize();
    let mut digest = [0_u8; REGISTER_LEN];
    digest.copy_from_slice(&out);
    digest
}

/// Fold one event digest into an RTMR: `SHA384(mr ‖ digest)`.
#[must_use]
pub fn rtmr_extend(mr: &[u8; REGISTER_LEN], digest: &[u8; REGISTER_LEN]) -> [u8; REGISTER_LEN] {
    let mut hasher = Sha384::new();
    hasher.update(mr);
    hasher.update(digest);
    let out = hasher.finalize();
    let mut next = [0_u8; REGISTER_LEN];
    next.copy_from_slice(&out);
    next
}

/// Replay RTMR3 (`imr == 3`) events from a synthetic or real event log.
///
/// - Non-RTMR3 entries (`imr != 3`) are ignored.
/// - Runtime events must use [`DSTACK_RUNTIME_EVENT_TYPE`]; any other type on
///   IMR 3 yields [`ReplayError::UnknownEventType`] (never skipped).
/// - Digest is always recomputed from `(event_type, name, payload)`.
/// - When `event.digest` is `Some`, it must equal the recomputed digest.
/// - Surfaces the `compose-hash` and `key-provider` payloads when present.
///
/// # Errors
///
/// Returns [`ReplayError::UnknownEventType`] when an IMR-3 event uses a type
/// other than [`DSTACK_RUNTIME_EVENT_TYPE`].
/// Returns [`ReplayError::DigestMismatch`] when a logged digest is present and
/// does not equal the recomputed digest.
pub fn replay_rtmr3(events: &[Event]) -> Result<Replay, ReplayError> {
    let mut mr = [0_u8; REGISTER_LEN];
    let mut compose_hash: Option<Vec<u8>> = None;
    let mut key_provider: Option<Vec<u8>> = None;

    for event in events {
        if event.imr != APP_IMR {
            continue;
        }

        if event.event_type != DSTACK_RUNTIME_EVENT_TYPE {
            return Err(ReplayError::UnknownEventType {
                event_type: event.event_type,
            });
        }

        let digest = event_digest(event.event_type, &event.name, &event.payload);

        if let Some(logged) = event.digest {
            if logged != digest {
                return Err(ReplayError::DigestMismatch {
                    name: event.name.clone(),
                });
            }
        }

        mr = rtmr_extend(&mr, &digest);

        if event.name == COMPOSE_HASH_EVENT {
            compose_hash = Some(event.payload.clone());
        } else if event.name == KEY_PROVIDER_EVENT {
            key_provider = Some(event.payload.clone());
        }
    }

    Ok(Replay {
        rtmr3: mr,
        compose_hash,
        key_provider,
    })
}

/// Failures while decoding a dstack / Phala event-log JSON document.
#[derive(Debug, Error, Eq, PartialEq)]
pub enum EventLogError {
    /// JSON is not a valid array of event objects.
    #[error("invalid event log JSON: {0}")]
    InvalidJson(String),
    /// A required field is missing or has the wrong type.
    #[error("event log entry invalid: {0}")]
    InvalidEntry(String),
    /// Hex field is malformed.
    #[error("event log hex decode failed for field '{field}': {detail}")]
    BadHex { field: String, detail: String },
}

#[derive(Debug, serde::Deserialize)]
struct WireEvent {
    imr: u32,
    event_type: u32,
    #[serde(default)]
    digest: Option<String>,
    #[serde(default)]
    event: String,
    #[serde(default)]
    event_payload: String,
}

/// Parse a dstack / Phala `event_log.json` array into [`Event`] values.
///
/// Wire fields: `imr`, `event_type`, `event` (name), `event_payload` (hex),
/// optional `digest` (hex). Empty payload/digest strings become empty / `None`.
///
/// # Errors
/// Returns [`EventLogError`] on JSON or hex failures.
pub fn events_from_json(bytes: &[u8]) -> Result<Vec<Event>, EventLogError> {
    let wire: Vec<WireEvent> =
        serde_json::from_slice(bytes).map_err(|e| EventLogError::InvalidJson(e.to_string()))?;
    let mut out = Vec::with_capacity(wire.len());
    for (i, w) in wire.into_iter().enumerate() {
        let payload = decode_hex_field("event_payload", &w.event_payload, i)?;
        let digest = match w.digest.as_deref() {
            None | Some("") => None,
            Some(h) => {
                let raw = decode_hex_field("digest", h, i)?;
                if raw.len() != REGISTER_LEN {
                    return Err(EventLogError::InvalidEntry(format!(
                        "entry {i}: digest must be {REGISTER_LEN} bytes, got {}",
                        raw.len()
                    )));
                }
                let mut reg = [0_u8; REGISTER_LEN];
                reg.copy_from_slice(&raw);
                Some(reg)
            }
        };
        out.push(Event {
            imr: w.imr,
            event_type: w.event_type,
            name: w.event,
            payload,
            digest,
        });
    }
    Ok(out)
}

fn decode_hex_field(field: &str, hex: &str, index: usize) -> Result<Vec<u8>, EventLogError> {
    if hex.is_empty() {
        return Ok(Vec::new());
    }
    if !hex.len().is_multiple_of(2) {
        return Err(EventLogError::BadHex {
            field: field.into(),
            detail: format!("entry {index}: odd length {}", hex.len()),
        });
    }
    let mut out = Vec::with_capacity(hex.len() / 2);
    let bytes = hex.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let hi = hex_nibble(bytes[i]).map_err(|d| EventLogError::BadHex {
            field: field.into(),
            detail: format!("entry {index}: {d}"),
        })?;
        let lo = hex_nibble(bytes[i + 1]).map_err(|d| EventLogError::BadHex {
            field: field.into(),
            detail: format!("entry {index}: {d}"),
        })?;
        out.push((hi << 4) | lo);
        i += 2;
    }
    Ok(out)
}

fn hex_nibble(b: u8) -> Result<u8, String> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(format!("invalid hex byte {b:?}")),
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len().saturating_mul(2));
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    /// Hand-computed: single compose-hash event with payload `0xc3` × 32.
    ///
    /// ```text
    /// digest = SHA384(le32(0x08000001) || ":" || "compose-hash" || ":" || payload)
    /// rtmr3  = SHA384(zeros48 || digest)
    /// ```
    const HAND_COMPOSE_PAYLOAD_HEX: &str =
        "c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3c3";
    const HAND_SINGLE_DIGEST_HEX: &str = "cd215d106dbab980241ba1081057cf9afe26f730b7b937e1e5cbb30a552881348eb5f55bf538035d9bd325b113eca998";
    const HAND_SINGLE_RTMR3_HEX: &str = "d66b966a576eea8267dcf6b7e2e6114ff6d32fb244e1dd1ce1911704a7fd8e6d39a757c1b935df5f39e141fd6dbca73e";

    /// Two-event order A then B (compose-hash, then key-provider `kms-provider-v1`).
    const HAND_AB_RTMR3_HEX: &str = "fa36681261bffb2f3434ad18ef698e14d937ec4761a1b4a60b107e18b330a3c025a184c9400000c3d316310f32a2f0b6";
    /// Two-event order B then A — must differ from AB.
    const HAND_BA_RTMR3_HEX: &str = "901607d82dc09d7fcdba0051422f62fd4f866a06540c0d29d90eccc611dd9671c70bba7aab409edf2589c6c81b112348";

    fn hex_to_bytes(hex: &str) -> Vec<u8> {
        assert!(hex.len().is_multiple_of(2), "hex length");
        (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex nibble"))
            .collect()
    }

    fn hex_to_reg(hex: &str) -> [u8; REGISTER_LEN] {
        let v = hex_to_bytes(hex);
        assert_eq!(v.len(), REGISTER_LEN);
        let mut out = [0_u8; REGISTER_LEN];
        out.copy_from_slice(&v);
        out
    }

    fn runtime_event(name: &str, payload: Vec<u8>) -> Event {
        Event {
            imr: APP_IMR,
            event_type: DSTACK_RUNTIME_EVENT_TYPE,
            name: name.to_owned(),
            payload,
            digest: None,
        }
    }

    #[test]
    fn event_digest_matches_hand_computed_compose_hash() {
        let payload = hex_to_bytes(HAND_COMPOSE_PAYLOAD_HEX);
        let d = event_digest(DSTACK_RUNTIME_EVENT_TYPE, COMPOSE_HASH_EVENT, &payload);
        assert_eq!(hex_encode(&d), HAND_SINGLE_DIGEST_HEX);
    }

    #[test]
    fn synthetic_single_event_replays_to_hand_computed_rtmr3() {
        let payload = hex_to_bytes(HAND_COMPOSE_PAYLOAD_HEX);
        let log = [runtime_event(COMPOSE_HASH_EVENT, payload.clone())];
        let replay = replay_rtmr3(&log).expect("replay");
        assert_eq!(replay.rtmr3_hex(), HAND_SINGLE_RTMR3_HEX);
        assert_eq!(replay.compose_hash.as_deref(), Some(payload.as_slice()));
        assert_eq!(
            replay.compose_hash_hex().as_deref(),
            Some(HAND_COMPOSE_PAYLOAD_HEX)
        );
        assert!(replay.key_provider.is_none());
    }

    #[test]
    fn empty_log_yields_zero_rtmr3_and_no_compose_hash() {
        let replay = replay_rtmr3(&[]).expect("empty");
        assert_eq!(replay.rtmr3, [0_u8; REGISTER_LEN]);
        assert!(replay.compose_hash.is_none());
        assert!(replay.key_provider.is_none());
    }

    #[test]
    fn reordering_two_events_changes_rtmr3() {
        let compose = hex_to_bytes(HAND_COMPOSE_PAYLOAD_HEX);
        let key = b"kms-provider-v1".to_vec();

        let ab = [
            runtime_event(COMPOSE_HASH_EVENT, compose.clone()),
            runtime_event(KEY_PROVIDER_EVENT, key.clone()),
        ];
        let ba = [
            runtime_event(KEY_PROVIDER_EVENT, key),
            runtime_event(COMPOSE_HASH_EVENT, compose),
        ];

        let replay_ab = replay_rtmr3(&ab).expect("ab");
        let replay_ba = replay_rtmr3(&ba).expect("ba");

        assert_eq!(replay_ab.rtmr3_hex(), HAND_AB_RTMR3_HEX);
        assert_eq!(replay_ba.rtmr3_hex(), HAND_BA_RTMR3_HEX);
        assert_ne!(
            replay_ab.rtmr3, replay_ba.rtmr3,
            "event order must bind into RTMR3"
        );
        // Both still surface compose-hash payload (last wins for each name).
        assert!(replay_ab.compose_hash.is_some());
        assert!(replay_ba.compose_hash.is_some());
        assert!(replay_ab.key_provider.is_some());
        assert!(replay_ba.key_provider.is_some());
    }

    #[test]
    fn unknown_event_type_is_surfaced_not_skipped() {
        let payload = hex_to_bytes(HAND_COMPOSE_PAYLOAD_HEX);
        let log = [Event {
            imr: APP_IMR,
            event_type: 0x0000_0099,
            name: COMPOSE_HASH_EVENT.to_owned(),
            payload,
            digest: None,
        }];
        let err = replay_rtmr3(&log).expect_err("unknown type must fail");
        assert_eq!(
            err,
            ReplayError::UnknownEventType {
                event_type: 0x0000_0099
            }
        );
    }

    #[test]
    fn unknown_event_type_does_not_leave_partial_mr_as_success() {
        // A valid event followed by an unknown type must error — never return
        // a half-applied RTMR3 as Ok.
        let payload = hex_to_bytes(HAND_COMPOSE_PAYLOAD_HEX);
        let log = [
            runtime_event(COMPOSE_HASH_EVENT, payload.clone()),
            Event {
                imr: APP_IMR,
                event_type: 0xDEAD_BEEF,
                name: "evil".to_owned(),
                payload: vec![1, 2, 3],
                digest: None,
            },
        ];
        assert!(matches!(
            replay_rtmr3(&log),
            Err(ReplayError::UnknownEventType {
                event_type: 0xDEAD_BEEF
            })
        ));
    }

    #[test]
    fn non_app_imr_entries_are_ignored() {
        let payload = hex_to_bytes(HAND_COMPOSE_PAYLOAD_HEX);
        let log = [
            Event {
                imr: 0,
                event_type: DSTACK_RUNTIME_EVENT_TYPE,
                name: "noise".to_owned(),
                payload: b"skip-me".to_vec(),
                digest: None,
            },
            runtime_event(COMPOSE_HASH_EVENT, payload),
        ];
        let replay = replay_rtmr3(&log).expect("replay");
        assert_eq!(replay.rtmr3_hex(), HAND_SINGLE_RTMR3_HEX);
    }

    #[test]
    fn logged_digest_mismatch_is_rejected() {
        let payload = hex_to_bytes(HAND_COMPOSE_PAYLOAD_HEX);
        let mut bad = [0_u8; REGISTER_LEN];
        bad[0] = 0xff;
        let log = [Event {
            imr: APP_IMR,
            event_type: DSTACK_RUNTIME_EVENT_TYPE,
            name: COMPOSE_HASH_EVENT.to_owned(),
            payload,
            digest: Some(bad),
        }];
        assert!(matches!(
            replay_rtmr3(&log),
            Err(ReplayError::DigestMismatch { .. })
        ));
    }

    #[test]
    fn logged_digest_match_is_accepted() {
        let payload = hex_to_bytes(HAND_COMPOSE_PAYLOAD_HEX);
        let digest = hex_to_reg(HAND_SINGLE_DIGEST_HEX);
        let log = [Event {
            imr: APP_IMR,
            event_type: DSTACK_RUNTIME_EVENT_TYPE,
            name: COMPOSE_HASH_EVENT.to_owned(),
            payload,
            digest: Some(digest),
        }];
        let replay = replay_rtmr3(&log).expect("match");
        assert_eq!(replay.rtmr3_hex(), HAND_SINGLE_RTMR3_HEX);
    }

    #[test]
    fn rtmr_extend_matches_hand_fold() {
        let digest = hex_to_reg(HAND_SINGLE_DIGEST_HEX);
        let mr = rtmr_extend(&[0_u8; REGISTER_LEN], &digest);
        assert_eq!(hex_encode(&mr), HAND_SINGLE_RTMR3_HEX);
    }

    /// Task-34 real Phala event log → RTMR3 must match quote.bin register.
    #[test]
    fn real_event_log_replays_quote_rtmr3() {
        // RTMR3 absolute offset in v4 quote (header 48 + 472).
        const RTMR3_OFF: usize = 48 + 472;
        let log = include_bytes!("../../gbase-attest-parse/tests/fixtures/real/event_log.json");
        let quote = include_bytes!("../../gbase-attest-parse/tests/fixtures/real/quote.bin");
        let events = events_from_json(log).expect("parse event log");
        let replay = replay_rtmr3(&events).expect("replay");
        let mut expected = [0_u8; REGISTER_LEN];
        expected.copy_from_slice(&quote[RTMR3_OFF..RTMR3_OFF + REGISTER_LEN]);
        assert_eq!(
            replay.rtmr3, expected,
            "replayed RTMR3 must equal quote RTMR3"
        );
        let compose = replay.compose_hash.expect("compose-hash event");
        assert_eq!(compose.len(), 32);
        assert_eq!(
            hex_encode(&compose),
            "1b8a63efb0f7afda1e52c823f39cc0a79d6d75c2e7a086b58e0e6a2db548524b"
        );
    }
}
