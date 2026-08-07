//! Multi-step tool-calling loop (OpenRouter-compatible).

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::llm::ChatClient;
use crate::prompts::system_prompt;
use crate::tools::{dispatch, tool_schemas, ToolContext};
use crate::types::{AgenticBackend, AgenticError, AgenticVerdict, ReviewRequest};

/// Loop budgets.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Max assistant turns (each may include multiple `tool_calls`).
    pub max_turns: u32,
    /// Max completion tokens per provider call.
    pub max_tokens_per_turn: u32,
    /// Soft total token budget across the loop. Each turn resends the full
    /// conversation, so `usage.total_tokens` accumulates superlinearly: v3
    /// source trees + a ≤24KiB metrics blob need ~60k for a healthy review
    /// (the old 24k default died mid-loop at ~40k on real submissions).
    pub token_budget: u64,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_turns: 12,
            max_tokens_per_turn: 1_800,
            token_budget: 120_000,
        }
    }
}

/// Live `OpenRouter` agentic verifier.
pub struct OpenRouterAgent {
    client: ChatClient,
    config: AgentConfig,
}

impl std::fmt::Debug for OpenRouterAgent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenRouterAgent")
            .field("client", &self.client)
            .field("config", &self.config)
            .finish()
    }
}

impl OpenRouterAgent {
    /// Build with default loop config.
    ///
    /// # Errors
    /// Empty key / HTTP client build.
    pub fn new(api_key: impl Into<String>) -> Result<Self, AgenticError> {
        Ok(Self {
            client: ChatClient::new(api_key)?,
            config: AgentConfig::default(),
        })
    }

    /// Explicit client config (wiremock) + loop budgets.
    ///
    /// # Errors
    /// Empty key.
    pub fn with_config(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
        model: impl Into<String>,
        config: AgentConfig,
    ) -> Result<Self, AgenticError> {
        Ok(Self {
            client: ChatClient::with_config(api_key, base_url, model)?,
            config,
        })
    }

    /// Provider base URL (key never exposed).
    #[must_use]
    pub fn base_url(&self) -> &str {
        self.client.base_url()
    }

    /// Configured model id.
    #[must_use]
    pub fn model(&self) -> &str {
        self.client.model()
    }

    async fn run_loop(&self, req: &ReviewRequest) -> Result<AgenticVerdict, AgenticError> {
        let ctx = ToolContext::from_request(req)?;
        let tools = tool_schemas();
        let mut messages = vec![
            json!({
                "role": "system",
                "content": system_prompt(&req.domain_rules, &req.primary_relpaths),
            }),
            json!({
                "role": "user",
                "content": format!(
                    "Review the submission in the mounted workdir. Primary: {:?}. Call submit_verdict when done.",
                    req.primary_relpaths
                ),
            }),
        ];
        let mut spent_tokens: u64 = 0;
        let mut forced = false;

        for _turn in 0..self.config.max_turns {
            if spent_tokens >= self.config.token_budget {
                // Budget death gets exactly one forced-verdict turn: the model
                // must convert evidence already gathered into submit_verdict
                // instead of dying mid-investigation with no outcome.
                if forced {
                    return Err(AgenticError::NoVerdict(format!(
                        "token budget exhausted ({spent_tokens}) without submit_verdict"
                    )));
                }
                forced = true;
                messages.push(json!({
                    "role": "user",
                    "content": "Token budget exhausted. Do NOT call any inspection tool. Call submit_verdict now with your best judgment from the evidence already gathered."
                }));
            }
            let turn = self
                .client
                .chat_tools(&messages, &tools, self.config.max_tokens_per_turn)
                .await?;
            spent_tokens = spent_tokens.saturating_add(turn.tokens);

            let tool_calls = turn
                .message
                .get("tool_calls")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();

            if tool_calls.is_empty() {
                // Prose-only — fail closed (must use submit_verdict).
                return Err(AgenticError::NoVerdict(
                    "assistant returned no tool_calls".into(),
                ));
            }

            messages.push(turn.message);

            for call in &tool_calls {
                let id = call
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or("call")
                    .to_owned();
                let name = call
                    .pointer("/function/name")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_owned();
                let args_raw = call
                    .pointer("/function/arguments")
                    .and_then(Value::as_str)
                    .unwrap_or("{}");
                let args: Value = serde_json::from_str(args_raw).unwrap_or_else(|_| json!({}));

                match dispatch(&ctx, &name, &args)? {
                    Ok(content) => {
                        messages.push(json!({
                            "role": "tool",
                            "tool_call_id": id,
                            "content": content,
                        }));
                    }
                    Err(verdict) => return Ok(verdict),
                }
            }
        }

        Err(AgenticError::NoVerdict(format!(
            "max turns ({}) without submit_verdict",
            self.config.max_turns
        )))
    }
}

#[async_trait]
impl AgenticBackend for OpenRouterAgent {
    async fn review(&self, req: &ReviewRequest) -> Result<AgenticVerdict, AgenticError> {
        self.run_loop(req).await
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use crate::types::VerdictKind;
    use std::fs;
    use tempfile::tempdir;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn loop_accepts_submit_verdict() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "submit_verdict",
                                "arguments": "{\"verdict\":\"clean\",\"cheat_codes\":[],\"nearest_id\":null,\"similarity_bps\":1200,\"rationale\":\"ok\"}"
                            }
                        }]
                    }
                }],
                "usage": {"total_tokens": 40}
            })))
            .mount(&server)
            .await;

        let dir = tempdir().unwrap();
        fs::write(dir.path().join("agent.py"), "def f():\n    return 1\n").unwrap();
        let agent = OpenRouterAgent::with_config(
            "test-key-123456",
            server.uri(),
            "m",
            AgentConfig::default(),
        )
        .unwrap();
        let req = ReviewRequest {
            workdir: dir.path().to_path_buf(),
            primary_relpaths: vec!["agent.py".into()],
            corpus: vec![],
            metrics_relpath: None,
            pages_relpath: None,
            sanitize_report_relpath: None,
            domain_rules: "design".into(),
        };
        let v = agent.review(&req).await.unwrap();
        assert_eq!(v.verdict, VerdictKind::Clean);
        assert_eq!(v.similarity_bps, 1200);
    }

    /// Sequential responses + capture (budget tests need distinct turns).
    struct Seq {
        bodies: Vec<Value>,
        idx: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        saw_forced: std::sync::Arc<std::sync::atomic::AtomicBool>,
    }

    impl wiremock::Respond for Seq {
        fn respond(&self, req: &wiremock::Request) -> ResponseTemplate {
            if String::from_utf8_lossy(&req.body).contains("Token budget exhausted") {
                self.saw_forced
                    .store(true, std::sync::atomic::Ordering::SeqCst);
            }
            let i = self
                .idx
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                .min(self.bodies.len() - 1);
            ResponseTemplate::new(200).set_body_json(&self.bodies[i])
        }
    }

    fn tool_call_body(name: &str, args: &str, tokens: u64) -> Value {
        json!({
            "choices": [{"message": {"role": "assistant", "content": null, "tool_calls": [{
                "id": "c", "type": "function", "function": {"name": name, "arguments": args}}]}}],
            "usage": {"total_tokens": tokens}
        })
    }

    #[test]
    fn default_budget_covers_v3_source_tree_reviews() {
        // Regression pin (E12): the old 24k default died mid-review at ~40k
        // accumulated tokens on real v3 submissions.
        assert_eq!(AgentConfig::default().token_budget, 120_000);
    }

    #[tokio::test]
    async fn budget_exhaustion_forces_final_verdict_turn() {
        let server = MockServer::start().await;
        let saw_forced = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let seq = Seq {
            bodies: vec![
                tool_call_body("read_file", "{\"path\":\"agent.py\"}", 500),
                tool_call_body(
                    "submit_verdict",
                    "{\"verdict\":\"clean\",\"cheat_codes\":[],\"nearest_id\":null,\"similarity_bps\":100,\"rationale\":\"best judgment from gathered evidence\"}",
                    900,
                ),
            ],
            idx: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            saw_forced: std::sync::Arc::clone(&saw_forced),
        };
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(seq)
            .mount(&server)
            .await;

        let dir = tempdir().unwrap();
        fs::write(dir.path().join("agent.py"), "x = 1\n").unwrap();
        let agent = OpenRouterAgent::with_config(
            "test-key-123456",
            server.uri(),
            "m",
            AgentConfig {
                token_budget: 100,
                ..AgentConfig::default()
            },
        )
        .unwrap();
        let req = ReviewRequest {
            workdir: dir.path().to_path_buf(),
            primary_relpaths: vec!["agent.py".into()],
            corpus: vec![],
            metrics_relpath: None,
            pages_relpath: None,
            sanitize_report_relpath: None,
            domain_rules: String::new(),
        };
        // Budget died after turn 1; the forced turn still yields a verdict.
        let v = agent.review(&req).await.unwrap();
        assert_eq!(v.verdict, VerdictKind::Clean);
        assert_eq!(v.similarity_bps, 100);
        assert!(saw_forced.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[tokio::test]
    async fn budget_exhaustion_without_verdict_is_defined_noverdict() {
        let server = MockServer::start().await;
        let saw_forced = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let seq = Seq {
            bodies: vec![tool_call_body("read_file", "{\"path\":\"agent.py\"}", 500)],
            idx: std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            saw_forced: std::sync::Arc::clone(&saw_forced),
        };
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(seq)
            .mount(&server)
            .await;

        let dir = tempdir().unwrap();
        fs::write(dir.path().join("agent.py"), "x = 1\n").unwrap();
        let agent = OpenRouterAgent::with_config(
            "test-key-123456",
            server.uri(),
            "m",
            AgentConfig {
                token_budget: 100,
                ..AgentConfig::default()
            },
        )
        .unwrap();
        let req = ReviewRequest {
            workdir: dir.path().to_path_buf(),
            primary_relpaths: vec!["agent.py".into()],
            corpus: vec![],
            metrics_relpath: None,
            pages_relpath: None,
            sanitize_report_relpath: None,
            domain_rules: String::new(),
        };
        // The model ignored the forced turn (kept inspecting): defined
        // NoVerdict error — the orchestrator maps this to a bounded
        // no-retrain retry, then terminal NoScore(ChallengeInternal).
        let err = agent.review(&req).await.unwrap_err();
        match err {
            AgenticError::NoVerdict(m) => assert!(m.contains("token budget exhausted"), "{m}"),
            other => panic!("expected NoVerdict, got {other}"),
        }
        assert!(saw_forced.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[tokio::test]
    async fn prose_only_fail_closed() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "looks clean to me"
                    }
                }],
                "usage": {"total_tokens": 10}
            })))
            .mount(&server)
            .await;

        let dir = tempdir().unwrap();
        fs::write(dir.path().join("agent.py"), "x = 1\n").unwrap();
        let agent = OpenRouterAgent::with_config(
            "test-key-123456",
            server.uri(),
            "m",
            AgentConfig {
                max_turns: 2,
                ..AgentConfig::default()
            },
        )
        .unwrap();
        let req = ReviewRequest {
            workdir: dir.path().to_path_buf(),
            primary_relpaths: vec!["agent.py".into()],
            corpus: vec![],
            metrics_relpath: None,
            pages_relpath: None,
            sanitize_report_relpath: None,
            domain_rules: String::new(),
        };
        let err = agent.review(&req).await.unwrap_err();
        assert!(matches!(err, AgenticError::NoVerdict(_)));
    }
}
