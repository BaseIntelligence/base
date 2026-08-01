//! HTTP dispatch client against miner agent-runners.
//!
//! One hop per miner: `GET /v1/capacity` → signed `POST /v1/task` →
//! `GET /v1/task/{id}` until terminal. Every failure mode maps onto an existing
//! [`agent_challenge::MinerEpochOutcome`]; nothing here can invent a result.

use std::collections::BTreeMap;
use std::time::Duration;

use agent_challenge::{EpochDispatchClient, RunnerCapacity, KEY_LEN};
use agent_dispatch::{TaskDescriptorV1, TaskResultV1};
use agent_runner::{
    sign_dispatch_request, unix_now_ms, CapacityResponse, TaskAccepted, TaskLifecycle, TaskView,
    DEFAULT_DISPATCH_NONCE_TTL,
};
use rand_core::{OsRng, RngCore};

/// Per-request HTTP budget for a single capacity / submit / poll hop.
const HTTP_TIMEOUT: Duration = Duration::from_secs(10);
/// Gap between `GET /v1/task/{id}` polls.
const POLL_INTERVAL: Duration = Duration::from_secs(2);
/// Dispatch-auth lifetime, kept strictly under the runner's accepted maximum.
const AUTH_TTL: Duration = Duration::from_mins(2);

/// Zero capacity: the miner cannot be dispatched to this tick.
const NO_CAPACITY: RunnerCapacity = RunnerCapacity {
    max_concurrency: 0,
    current_load: 0,
};

/// Dispatch client that talks to the base URL each miner published on chain.
#[derive(Debug)]
pub struct HttpDispatchClient {
    http: reqwest::Client,
    endpoints: BTreeMap<[u8; KEY_LEN], String>,
    secret: [u8; KEY_LEN],
    public: [u8; KEY_LEN],
}

impl HttpDispatchClient {
    /// Build a client over already-resolved axon endpoints.
    ///
    /// # Errors
    ///
    /// TLS / HTTP client construction failure.
    pub fn new(
        endpoints: BTreeMap<[u8; KEY_LEN], String>,
        secret: [u8; KEY_LEN],
        public: [u8; KEY_LEN],
    ) -> Result<Self, String> {
        let http = reqwest::Client::builder()
            .timeout(HTTP_TIMEOUT)
            .build()
            .map_err(|e| format!("dispatch http client: {e}"))?;
        Ok(Self {
            http,
            endpoints,
            secret,
            public,
        })
    }

    async fn submit(&self, base: &str, descriptor: TaskDescriptorV1) -> Result<String, String> {
        let mut nonce = [0_u8; KEY_LEN];
        OsRng.fill_bytes(&mut nonce);
        let ttl = AUTH_TTL.min(DEFAULT_DISPATCH_NONCE_TTL);
        let expires_at = unix_now_ms()
            .saturating_add(u64::try_from(ttl.as_millis()).unwrap_or(u64::from(u32::MAX)));
        let signed =
            sign_dispatch_request(&self.secret, &self.public, descriptor, nonce, expires_at)
                .map_err(|e| format!("sign dispatch: {e}"))?;

        let resp = self
            .http
            .post(format!("{base}/v1/task"))
            .json(&signed)
            .send()
            .await
            .map_err(|e| format!("POST /v1/task: {e}"))?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("POST /v1/task refused: {status} {body}"));
        }
        let accepted: TaskAccepted = resp
            .json()
            .await
            .map_err(|e| format!("POST /v1/task body: {e}"))?;
        Ok(accepted.task_id)
    }

    async fn poll(&self, base: &str, task_id: &str) -> Result<TaskResultV1, String> {
        loop {
            let view: TaskView = self
                .http
                .get(format!("{base}/v1/task/{task_id}"))
                .send()
                .await
                .map_err(|e| format!("GET /v1/task/{task_id}: {e}"))?
                .error_for_status()
                .map_err(|e| format!("GET /v1/task/{task_id}: {e}"))?
                .json()
                .await
                .map_err(|e| format!("GET /v1/task/{task_id} body: {e}"))?;
            match view.status {
                TaskLifecycle::Pending | TaskLifecycle::Running => {
                    tokio::time::sleep(POLL_INTERVAL).await;
                }
                // A terminal lifecycle with no envelope is a broken runner, not a
                // score: surface it so the miner gets MinerError, never a zero.
                TaskLifecycle::Completed | TaskLifecycle::TimedOut | TaskLifecycle::Failed => {
                    return view
                        .result
                        .ok_or_else(|| format!("terminal task {task_id} carried no result"));
                }
            }
        }
    }
}

impl EpochDispatchClient for HttpDispatchClient {
    async fn capacity(&self, miner: [u8; KEY_LEN]) -> RunnerCapacity {
        let Some(base) = self.endpoints.get(&miner) else {
            return NO_CAPACITY;
        };
        let probe = self
            .http
            .get(format!("{base}/v1/capacity"))
            .send()
            .await
            .and_then(reqwest::Response::error_for_status);
        match probe {
            Ok(r) => match r.json::<CapacityResponse>().await {
                Ok(c) => RunnerCapacity {
                    max_concurrency: c.max_concurrency,
                    current_load: c.current_load,
                },
                Err(e) => {
                    tracing::warn!(
                        event = "capacity_body_invalid",
                        hotkey = %hex::encode(miner),
                        error = %e,
                        "runner capacity body unreadable"
                    );
                    NO_CAPACITY
                }
            },
            Err(e) => {
                tracing::info!(
                    event = "capacity_unreachable",
                    hotkey = %hex::encode(miner),
                    error = %e,
                    "runner did not answer /v1/capacity"
                );
                NO_CAPACITY
            }
        }
    }

    async fn run_pack(
        &self,
        miner: [u8; KEY_LEN],
        descriptor: TaskDescriptorV1,
    ) -> Result<TaskResultV1, String> {
        let base = self
            .endpoints
            .get(&miner)
            .ok_or_else(|| "miner published no axon".to_owned())?
            .clone();
        let task_id = self.submit(&base, descriptor).await?;
        self.poll(&base, &task_id).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_challenge::{
        public_key_from_secret, run_epoch_dispatch, ActiveSignerRegistry, EpochDispatchConfig,
        ExpectedParticipant, ExpectedSet, MinerEpochOutcome,
    };
    use agent_dispatch::{TaskStatusV1, DISPATCH_PROTOCOL};
    use agent_pack::PackId;
    use std::sync::Arc;
    use wiremock::matchers::{method, path, path_regex};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    const MINER: [u8; KEY_LEN] = [0xB1; KEY_LEN];

    fn keypair() -> ([u8; KEY_LEN], [u8; KEY_LEN]) {
        let sk = [0x42_u8; KEY_LEN];
        let pk = public_key_from_secret(&sk).expect("pk");
        (sk, pk)
    }

    fn result_json(patch: &str) -> serde_json::Value {
        serde_json::json!({
            "protocol": DISPATCH_PROTOCOL,
            "challenge_id": "agent-v1",
            "scoring_version": 2,
            "epoch": 7,
            "miner_hotkey_hex": hex::encode(MINER),
            "pack_id": "pack-a",
            "status": "completed",
            "model_patch": patch,
            "patch_sha256_hex": hex::encode(agent_dispatch::patch_sha256(patch.as_bytes())),
            "receipt_sig_hex": hex::encode([0_u8; 64]),
        })
    }

    async fn runner_with_capacity(max: u32, load: u32, patch: &str) -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/capacity"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "max_concurrency": max,
                "current_load": load,
            })))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/task"))
            .respond_with(
                ResponseTemplate::new(202).set_body_json(serde_json::json!({"task_id": "t-1"})),
            )
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path_regex(r"^/v1/task/.+$"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "task_id": "t-1",
                "status": "completed",
                "result": result_json(patch),
            })))
            .mount(&server)
            .await;
        server
    }

    fn dispatch_cfg(expected: ExpectedSet) -> EpochDispatchConfig {
        EpochDispatchConfig {
            challenge_id: "agent-v1".into(),
            scoring_version: 2,
            epoch: 7,
            expected,
            catalog: vec![PackId::new("pack-a")],
            deadline: Duration::from_secs(20),
            deadline_unix_ms: unix_now_ms().saturating_add(20_000),
        }
    }

    fn expected_set() -> ExpectedSet {
        ExpectedSet {
            block_hash: [0x11; 32],
            participants: vec![ExpectedParticipant {
                hotkey: MINER,
                uid: 1,
            }],
        }
    }

    /// Regression: the old client advertised zero capacity, so no miner was
    /// ever contacted. A reachable runner must actually receive the dispatch.
    #[tokio::test]
    async fn reachable_miner_receives_a_dispatch_and_returns_its_patch() {
        let server = runner_with_capacity(2, 0, "diff --git a/x b/x").await;
        let (sk, pk) = keypair();
        let client = Arc::new(
            HttpDispatchClient::new(BTreeMap::from([(MINER, server.uri())]), sk, pk)
                .expect("client"),
        );

        let out = run_epoch_dispatch(
            &dispatch_cfg(expected_set()),
            client,
            &ActiveSignerRegistry::new(),
        )
        .await
        .expect("dispatch");

        match out.outcomes.get(&MINER) {
            Some(MinerEpochOutcome::Completed { pack_id, result }) => {
                assert_eq!(pack_id, "pack-a");
                assert_eq!(result.status, TaskStatusV1::Completed);
                assert_eq!(result.model_patch.as_deref(), Some("diff --git a/x b/x"));
            }
            other => panic!("expected Completed, got {other:?}"),
        }

        // The runner saw a real, signed submission.
        let posts: Vec<_> = server
            .received_requests()
            .await
            .expect("requests")
            .into_iter()
            .filter(|r| r.url.path() == "/v1/task")
            .collect();
        assert_eq!(posts.len(), 1);
        let envelope: serde_json::Value = posts[0].body_json().expect("json");
        assert_eq!(
            envelope["signer_pubkey_hex"].as_str(),
            Some(hex::encode(pk).as_str())
        );
        assert_eq!(envelope["signature_hex"].as_str().map(str::len), Some(128));
        assert_eq!(envelope["descriptor"]["epoch"].as_u64(), Some(7));
    }

    /// No published axon must be an absence, never a fabricated dispatch.
    #[tokio::test]
    async fn miner_without_an_axon_is_capacity_exhausted() {
        let (sk, pk) = keypair();
        let client = Arc::new(HttpDispatchClient::new(BTreeMap::new(), sk, pk).expect("client"));
        let out = run_epoch_dispatch(
            &dispatch_cfg(expected_set()),
            client,
            &ActiveSignerRegistry::new(),
        )
        .await
        .expect("dispatch");
        assert!(matches!(
            out.outcomes.get(&MINER),
            Some(MinerEpochOutcome::CapacityExhausted { .. })
        ));
    }

    /// A runner that reports itself full is not dispatched to.
    #[tokio::test]
    async fn full_runner_is_not_dispatched_to() {
        let server = runner_with_capacity(1, 1, "x").await;
        let (sk, pk) = keypair();
        let client = Arc::new(
            HttpDispatchClient::new(BTreeMap::from([(MINER, server.uri())]), sk, pk)
                .expect("client"),
        );
        let out = run_epoch_dispatch(
            &dispatch_cfg(expected_set()),
            client,
            &ActiveSignerRegistry::new(),
        )
        .await
        .expect("dispatch");
        assert!(matches!(
            out.outcomes.get(&MINER),
            Some(MinerEpochOutcome::CapacityExhausted { .. })
        ));
        assert!(server
            .received_requests()
            .await
            .expect("requests")
            .iter()
            .all(|r| r.url.path() != "/v1/task"));
    }

    /// A refused submit is a miner-attributable failure, not a score.
    #[tokio::test]
    async fn refused_submit_is_a_failure_outcome() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/capacity"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"max_concurrency": 1, "current_load": 0})),
            )
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/v1/task"))
            .respond_with(ResponseTemplate::new(401).set_body_json(
                serde_json::json!({"error": "unauthorized", "code": "unauthorized"}),
            ))
            .mount(&server)
            .await;
        let (sk, pk) = keypair();
        let client = Arc::new(
            HttpDispatchClient::new(BTreeMap::from([(MINER, server.uri())]), sk, pk)
                .expect("client"),
        );
        let out = run_epoch_dispatch(
            &dispatch_cfg(expected_set()),
            client,
            &ActiveSignerRegistry::new(),
        )
        .await
        .expect("dispatch");
        match out.outcomes.get(&MINER) {
            Some(MinerEpochOutcome::Failed { reason, .. }) => {
                assert!(reason.contains("401"), "{reason}");
            }
            other => panic!("expected Failed, got {other:?}"),
        }
    }
}
