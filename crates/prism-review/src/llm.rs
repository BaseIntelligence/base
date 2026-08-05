//! `OpenRouter` chat client (master-side; pod never sees a key).

use std::path::PathBuf;

use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderValue, USER_AGENT};

use crate::prompts::{
    REVIEW_PROMPT_V3, REVIEW_PROMPT_VERSION, SIMILARITY_PROMPT_V2, SIMILARITY_PROMPT_VERSION,
};
use crate::types::{
    truncate_source, ReviewError, ReviewVerdict, SimilarityKind, SimilarityVerdict, SourceSnippet,
    MAX_CORPUS_SNIPPETS, MAX_REVIEW_TOKENS, OPENROUTER_API_BASE,
};
use crate::ReviewBackend;

/// Default chat model (cheap, deterministic-ish for reviews).
pub const DEFAULT_MODEL: &str = "openai/gpt-4o-mini";

/// Read the `OpenRouter` API key from a file (mode 0400), never from env text.
///
/// # Errors
/// Missing/empty file.
pub fn load_api_key_file(path: &std::path::Path) -> Result<String, ReviewError> {
    let text = std::fs::read_to_string(path).map_err(|_| ReviewError::NoKey)?;
    let key = text.trim().to_owned();
    if key.is_empty() || key.len() < 12 {
        return Err(ReviewError::NoKey);
    }
    Ok(key)
}

/// `OpenRouter` chat client. Key is never `Debug`/`Display`'d.
pub struct OpenRouterClient {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
    model: String,
}

impl std::fmt::Debug for OpenRouterClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenRouterClient")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("api_key", &"<redacted>")
            .finish_non_exhaustive()
    }
}

impl OpenRouterClient {
    /// Build with default base + model.
    ///
    /// # Errors
    /// Empty key.
    pub fn new(api_key: impl Into<String>) -> Result<Self, ReviewError> {
        Self::with_config(api_key, OPENROUTER_API_BASE, DEFAULT_MODEL)
    }

    /// Build with explicit base + model (wiremock tests).
    ///
    /// # Errors
    /// Empty key.
    pub fn with_config(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
        model: impl Into<String>,
    ) -> Result<Self, ReviewError> {
        let api_key = api_key.into();
        if api_key.trim().is_empty() {
            return Err(ReviewError::NoKey);
        }
        let mut headers = HeaderMap::new();
        let mut hv = HeaderValue::from_str(&format!("Bearer {api_key}"))
            .map_err(|e| ReviewError::Transport(e.to_string()))?;
        hv.set_sensitive(true);
        headers.insert(reqwest::header::AUTHORIZATION, hv);
        headers.insert(
            USER_AGENT,
            HeaderValue::from_static("base-prism-review/0.1"),
        );
        let http = reqwest::Client::builder()
            .default_headers(headers)
            .timeout(std::time::Duration::from_secs(90))
            .build()
            .map_err(|e| ReviewError::Transport(e.to_string()))?;
        Ok(Self {
            http,
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            api_key,
            model: model.into(),
        })
    }

    /// Never expose key material.
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Never expose key material.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }

    async fn chat(&self, prompt: &str) -> Result<String, ReviewError> {
        let url = format!("{}/chat/completions", self.base_url);
        let body = serde_json::json!({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": MAX_REVIEW_TOKENS,
        });
        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| ReviewError::Transport(sanitize(&e.to_string(), &self.api_key)))?;
        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| ReviewError::Transport(sanitize(&e.to_string(), &self.api_key)))?;
        if !status.is_success() {
            return Err(ReviewError::Provider(format!(
                "{status}: {}",
                sanitize(&truncate_str(&text, 240), &self.api_key)
            )));
        }
        let v: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| ReviewError::Parse(format!("envelope json: {e}")))?;
        v.pointer("/choices/0/message/content")
            .and_then(|c| c.as_str())
            .map(str::to_owned)
            .ok_or_else(|| ReviewError::Parse("missing choices[0].message.content".into()))
    }
}

fn truncate_str(s: &str, n: usize) -> String {
    if s.len() <= n {
        s.to_owned()
    } else {
        format!("{}…", &s[..n])
    }
}

fn sanitize(msg: &str, key: &str) -> String {
    if key.is_empty() {
        msg.to_owned()
    } else {
        msg.replace(key, "<redacted>")
    }
}

/// Extract the first {...} JSON object from an LLM answer (tolerates leading
/// prose the template forbids but models occasionally emit).
fn extract_json(text: &str) -> Result<serde_json::Value, ReviewError> {
    let start = text
        .find('{')
        .ok_or_else(|| ReviewError::Parse("no json object".into()))?;
    let mut depth = 0i32;
    let mut in_str = false;
    let mut esc = false;
    let mut end = None;
    for (i, c) in text[start..].char_indices() {
        if esc {
            esc = false;
            continue;
        }
        match c {
            '\\' if in_str => esc = true,
            '"' => in_str = !in_str,
            '{' if !in_str => depth += 1,
            '}' if !in_str => {
                depth -= 1;
                if depth == 0 {
                    end = Some(start + i + 1);
                    break;
                }
            }
            _ => {}
        }
    }
    let end = end.ok_or_else(|| ReviewError::Parse("unbalanced json".into()))?;
    serde_json::from_str(&text[start..end]).map_err(|e| ReviewError::Parse(format!("json: {e}")))
}

fn parse_review(text: &str) -> Result<ReviewVerdict, ReviewError> {
    let v = extract_json(text)?;
    let qs = v
        .get("quality_score")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| ReviewError::Parse("quality_score missing/non-int".into()))?;
    if qs > 1000 {
        return Err(ReviewError::Parse("quality_score > 1000".into()));
    }
    let issues = v
        .get("issues")
        .and_then(|x| x.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|i| i.as_str().map(|s| s.chars().take(160).collect::<String>()))
                .take(5)
                .collect::<Vec<String>>()
        })
        .unwrap_or_default();
    Ok(ReviewVerdict {
        quality_score: u16::try_from(qs.min(u64::from(u16::MAX))).unwrap_or(0),
        issues,
        prompt_version: REVIEW_PROMPT_VERSION,
    })
}

fn parse_similarity(text: &str) -> Result<SimilarityVerdict, ReviewError> {
    let v = extract_json(text)?;
    let kind = match v.get("kind").and_then(|x| x.as_str()).unwrap_or("") {
        "original" => SimilarityKind::Original,
        "suspicious" => SimilarityKind::Suspicious,
        "copied" => SimilarityKind::Copied,
        other => return Err(ReviewError::Parse(format!("kind: {other:?}"))),
    };
    let score = v
        .get("score")
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| ReviewError::Parse("score missing/non-float".into()))?;
    if !(0.0..=1.0).contains(&score) {
        return Err(ReviewError::Parse("score out of range".into()));
    }
    let closest = v
        .get("closest")
        .and_then(|x| x.as_str())
        .filter(|s| s != &"null")
        .map(|s| s.chars().take(64).collect());
    let evidence = v
        .get("evidence")
        .and_then(|x| x.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|i| i.as_str().map(|s| s.chars().take(160).collect::<String>()))
                .take(3)
                .collect::<Vec<String>>()
        })
        .unwrap_or_default();
    Ok(SimilarityVerdict {
        kind,
        score,
        closest,
        evidence,
        prompt_version: SIMILARITY_PROMPT_VERSION,
    })
}

/// Corpus rendering for the similarity prompt: **architectures only**
/// (similarity v2 — training.py is exempt from the corpus).
fn corpus_block(corpus: &[SourceSnippet]) -> String {
    corpus
        .iter()
        .take(MAX_CORPUS_SNIPPETS)
        .map(|s| {
            format!(
                "--- {label} architecture.py ---\n{arch}\n",
                label = s.label,
                arch = truncate_source(&s.architecture_py),
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[async_trait]
impl ReviewBackend for OpenRouterClient {
    async fn review(
        &self,
        architecture_py: &str,
        training_py: &str,
    ) -> Result<ReviewVerdict, ReviewError> {
        let prompt = REVIEW_PROMPT_V3
            .replace("{ARCH}", &truncate_source(architecture_py))
            .replace("{TRAIN}", &truncate_source(training_py));
        let answer = self.chat(&prompt).await?;
        parse_review(&answer)
    }

    async fn similarity(
        &self,
        architecture_py: &str,
        corpus: &[SourceSnippet],
    ) -> Result<SimilarityVerdict, ReviewError> {
        let prompt = SIMILARITY_PROMPT_V2
            .replace("{ARCH}", &truncate_source(architecture_py))
            .replace("{CORPUS}", &corpus_block(corpus));
        let answer = self.chat(&prompt).await?;
        parse_similarity(&answer)
    }
}

/// Default key path (documented for deployments).
#[must_use]
#[allow(dead_code)] // used by operators reading the crate public surface
pub fn default_key_path() -> PathBuf {
    PathBuf::from("/run/base/openrouter/api_key")
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    #[test]
    fn parse_review_ok() {
        let v = parse_review(r#"{"quality_score": 640, "issues": ["lr too high"]}"#).unwrap();
        assert_eq!(v.quality_score, 640);
        assert_eq!(v.issues.len(), 1);
    }

    #[test]
    fn parse_review_bounds() {
        assert!(parse_review(r#"{"quality_score": 1001}"#).is_err());
        assert!(parse_review("not json").is_err());
    }

    #[test]
    fn parse_similarity_ok() {
        let v = parse_similarity(
            r#"{"kind": "copied", "score": 0.97, "closest": "baseline", "evidence": ["same TinyGPT"]}"#,
        )
        .unwrap();
        assert!(matches!(v.kind, SimilarityKind::Copied));
        assert!((v.score - 0.97).abs() < 1e-9);
        assert_eq!(v.closest.as_deref(), Some("baseline"));
    }

    #[test]
    fn parse_similarity_rejects_garbage() {
        assert!(parse_similarity(r#"{"kind": "unknown"}"#).is_err());
        assert!(parse_similarity(r#"{"kind": "copied"}"#).is_err());
    }

    #[test]
    fn extract_json_tolerant() {
        let v = extract_json("The answer is:{\"a\": 1} done.").unwrap();
        assert_eq!(v["a"], 1);
    }

    #[tokio::test]
    async fn chat_wiremock_roundtrip() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{"message": {"content": "{\"quality_score\": 700, \"issues\": []}"}}]
            })))
            .mount(&server)
            .await;
        let c = OpenRouterClient::with_config("test-key", server.uri(), "m").unwrap();
        let v = c
            .review(
                "def build_model(ctx): pass",
                "def train(model, ctx): return {}",
            )
            .await
            .unwrap();
        assert_eq!(v.quality_score, 700);
    }

    #[tokio::test]
    async fn provider_error_sanitized_and_typed() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .respond_with(
                ResponseTemplate::new(401)
                    .set_body_string("User not found. Key test-secret-key-123 is invalid"),
            )
            .mount(&server)
            .await;
        let c = OpenRouterClient::with_config("test-secret-key-123", server.uri(), "m").unwrap();
        let err = c.review("a", "b").await.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("<redacted>"));
        assert!(!msg.contains("test-secret-key-123"));
    }
}
