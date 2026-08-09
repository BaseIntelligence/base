//! Deterministic offline agentic verifier (CI / local sim, no network).
//!
//! Heuristics: metrics forge / `METRICS_JSON` / Prism static source cheats
//! (hooks, non-causal `TokenMix`) → cheat; scrape-style famous-site clones →
//! cheat; byte-identical corpus / extreme AST → cheat/suspicious; else clean.

use std::fs;
use std::path::Path;

use async_trait::async_trait;
use challenge_ast::{
    fingerprint_source, similarity_bps, static_source_cheat, top_k_nearest, SourceCheatKind,
    AST_CHEAT_BPS, AST_SUSPICIOUS_BPS, BASELINE_CORPUS_PREFIX,
};
use serde_json::Value;

use crate::tools::{load_primary_sources, resolve_rel, source_hash_hex};
use challenge_agentic_types::{
    AgenticBackend, AgenticError, AgenticVerdict, CheatCode, ReviewRequest, VerdictKind,
};

/// BPB below this with near-zero tokens is treated as forged (sim heuristic).
const SIM_IMPOSSIBLE_BPB: f64 = 0.2;
/// Recipe-v1: BPB below this ⇒ label-leak / forge on the frozen val cut.
const SIM_ABSURD_LOW_BPB: f64 = 1.0;

/// AST similarity ≥ this → `cheat` (with corpus nearest).
pub const SIM_CHEAT_BPS: u16 = AST_CHEAT_BPS;
/// AST similarity ≥ this (and < cheat) → `suspicious`.
pub const SIM_SUSPICIOUS_BPS: u16 = AST_SUSPICIOUS_BPS;

fn is_baseline_entry(id: &str) -> bool {
    id.starts_with(BASELINE_CORPUS_PREFIX)
}

fn cheat_verdict(code: CheatCode, rationale: impl Into<String>) -> AgenticVerdict {
    AgenticVerdict {
        verdict: VerdictKind::Cheat,
        cheat_codes: vec![code],
        nearest_id: None,
        similarity_bps: 10_000,
        rationale: rationale.into(),
    }
}

/// Offline deterministic agentic backend.
#[derive(Debug, Default)]
pub struct SimAgent {
    _private: (),
}

impl SimAgent {
    /// New sim backend.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Primaries in scope for corpus comparison. Prism compares `architecture.py`
/// ONLY (similarity v2: training.py is exempt from corpus/candidate); other
/// domains use every primary.
fn scoped_primaries<'a>(req: &ReviewRequest, primaries: &'a [(String, String)]) -> Vec<&'a str> {
    if req.domain_rules.contains("Prism domain") {
        let arch: Vec<&str> = primaries
            .iter()
            .filter(|(p, _)| p.ends_with("architecture.py"))
            .map(|(_, s)| s.as_str())
            .collect();
        if !arch.is_empty() {
            return arch;
        }
    }
    primaries.iter().map(|(_, s)| s.as_str()).collect()
}

#[async_trait]
impl AgenticBackend for SimAgent {
    async fn review(&self, req: &ReviewRequest) -> Result<AgenticVerdict, AgenticError> {
        let primaries = load_primary_sources(req)?;
        if primaries.is_empty() {
            return Err(AgenticError::NoVerdict("no primary files".into()));
        }

        if let Some(v) = metrics_cheat_verdict(req, &primaries)? {
            return Ok(v);
        }

        if let Some(v) = static_source_cheat_verdict(req, &primaries) {
            return Ok(v);
        }

        if let Some(v) = pages_scrape_cheat_verdict(req)? {
            return Ok(v);
        }

        // Concatenate the in-scope primaries for hash / fingerprint (stable
        // order from request).
        let joined = scoped_primaries(req, &primaries).join("\n#--\n");
        let cand_hash = source_hash_hex(&joined);

        for entry in &req.corpus {
            if is_baseline_entry(&entry.id) {
                continue;
            }
            if source_hash_hex(&entry.source) == cand_hash
                || primaries
                    .iter()
                    .any(|(_, s)| source_hash_hex(s) == source_hash_hex(&entry.source))
            {
                return Ok(AgenticVerdict {
                    verdict: VerdictKind::Cheat,
                    cheat_codes: vec![CheatCode::NearIdenticalHarnessCopy],
                    nearest_id: Some(entry.id.clone()),
                    similarity_bps: 10_000,
                    rationale: "sim: byte-identical corpus source".into(),
                });
            }
        }

        let Ok(query_fp) = fingerprint_source(&joined) else {
            // Unparseable primary → fail closed (operator), not silent clean.
            return Err(AgenticError::NoVerdict(
                "sim: primary python did not parse".into(),
            ));
        };

        let mut corpus_fps = Vec::new();
        for entry in &req.corpus {
            if is_baseline_entry(&entry.id) {
                continue;
            }
            if let Ok(fp) = fingerprint_source(&entry.source) {
                corpus_fps.push((entry.id.clone(), fp));
            }
        }

        if corpus_fps.is_empty() {
            return Ok(AgenticVerdict {
                verdict: VerdictKind::Clean,
                cheat_codes: vec![],
                nearest_id: None,
                similarity_bps: 0,
                rationale: "sim: empty usable corpus".into(),
            });
        }

        let nearest = top_k_nearest(&query_fp, &corpus_fps, 1);
        let n = nearest
            .into_iter()
            .next()
            .ok_or_else(|| AgenticError::NoVerdict("sim: empty nearest".into()))?;

        // Recompute for clarity (same as neighbor.similarity_bps).
        let bps = corpus_fps
            .iter()
            .find(|(id, _)| id == &n.id)
            .map_or(n.similarity_bps, |(_, fp)| similarity_bps(&query_fp, fp));

        if bps >= SIM_CHEAT_BPS {
            return Ok(AgenticVerdict {
                verdict: VerdictKind::Cheat,
                cheat_codes: vec![CheatCode::AstArchitectureCopy],
                nearest_id: Some(n.id),
                similarity_bps: bps,
                rationale: format!("sim: ast similarity_bps={bps} >= {SIM_CHEAT_BPS}"),
            });
        }
        if bps >= SIM_SUSPICIOUS_BPS {
            return Ok(AgenticVerdict {
                verdict: VerdictKind::Suspicious,
                cheat_codes: vec![CheatCode::NearIdenticalHarnessCopy],
                nearest_id: Some(n.id),
                similarity_bps: bps,
                rationale: format!("sim: ast similarity_bps={bps} >= {SIM_SUSPICIOUS_BPS}"),
            });
        }

        Ok(AgenticVerdict {
            verdict: VerdictKind::Clean,
            cheat_codes: vec![],
            nearest_id: Some(n.id),
            similarity_bps: bps,
            rationale: format!("sim: ast similarity_bps={bps} below thresholds"),
        })
    }
}

/// Prism pre-pod static source cheats (`challenge_ast::static_source_cheat`).
fn static_source_cheat_verdict(
    req: &ReviewRequest,
    primaries: &[(String, String)],
) -> Option<AgenticVerdict> {
    if !req.domain_rules.contains("Prism domain") {
        return None;
    }
    let pick = |suf: &str| {
        primaries
            .iter()
            .find(|(p, _)| p.ends_with(suf))
            .map_or("", |(_, s)| s.as_str())
    };
    let hit = static_source_cheat(pick("architecture.py"), pick("training.py"))?;
    let code = match hit.kind {
        SourceCheatKind::EvalShortCircuit => CheatCode::EvalShortCircuit,
        SourceCheatKind::MissingTelemetryHooks => CheatCode::MissingTelemetryHooks,
        SourceCheatKind::NonCausalLabelLeak => CheatCode::NonCausalLabelLeak,
    };
    Some(cheat_verdict(code, format!("sim: {}", hit.rationale)))
}

/// Brand clusters for scrape-style famous-site clones (≥2 hits in one cluster).
const FAMOUS_SITE_CLUSTERS: &[&[&str]] = &[
    &[
        "twitter.com",
        // Survives sanitize when og:/data-testid markers are stripped.
        "x.com",
        "property=\"og:site_name\" content=\"twitter\"",
        "property=\"og:site_name\" content=\"x\"",
        "data-testid=\"tweet\"",
        "aria-label=\"twitter\"",
    ],
    &[
        "facebook.com",
        "property=\"og:site_name\" content=\"facebook\"",
        "fbcdn.net",
        "data-testid=\"facebook",
    ],
    &[
        "instagram.com",
        "property=\"og:site_name\" content=\"instagram\"",
        "cdninstagram.com",
    ],
    &[
        "youtube.com",
        "property=\"og:site_name\" content=\"youtube\"",
        "ytimg.com",
        "ytd-app",
    ],
    &[
        "linkedin.com",
        "property=\"og:site_name\" content=\"linkedin\"",
        "licdn.com",
    ],
    &[
        "netflix.com",
        "property=\"og:site_name\" content=\"netflix\"",
        "nflxext.com",
    ],
    &[
        "amazon.com",
        "property=\"og:site_name\" content=\"amazon\"",
        "amazon-adsystem.com",
    ],
    &[
        "github.com",
        "property=\"og:site_name\" content=\"github\"",
        "github.githubassets.com",
    ],
];

/// Ultra-strong single markers (`og:site_name` of a famous property).
const FAMOUS_OG_SITE_NAMES: &[&str] = &[
    "property=\"og:site_name\" content=\"twitter\"",
    "property=\"og:site_name\" content=\"x\"",
    "property=\"og:site_name\" content=\"facebook\"",
    "property=\"og:site_name\" content=\"instagram\"",
    "property=\"og:site_name\" content=\"youtube\"",
    "property=\"og:site_name\" content=\"linkedin\"",
    "property=\"og:site_name\" content=\"netflix\"",
    "property=\"og:site_name\" content=\"amazon\"",
    "property=\"og:site_name\" content=\"github\"",
];

/// Design pages that look like an untransformed famous-site scrape → cheat.
fn pages_scrape_cheat_verdict(req: &ReviewRequest) -> Result<Option<AgenticVerdict>, AgenticError> {
    let Some(rel) = &req.pages_relpath else {
        return Ok(None);
    };
    let workdir = req
        .workdir
        .canonicalize()
        .map_err(|e| AgenticError::Tool(format!("workdir: {e}")))?;
    let pages_dir = resolve_rel(&workdir, rel)?;
    if !pages_dir.is_dir() {
        return Ok(None);
    }
    let mut blob = String::new();
    for ent in fs::read_dir(&pages_dir)
        .map_err(|e| AgenticError::Tool(format!("pages: {e}")))?
        .flatten()
    {
        let path = ent.path();
        if path.extension().and_then(|e| e.to_str()) != Some("html") {
            continue;
        }
        if let Ok(t) = fs::read_to_string(&path) {
            blob.push_str(&t);
            blob.push('\n');
        }
    }
    if blob.is_empty() {
        return Ok(None);
    }
    let low = blob.to_ascii_lowercase();
    if FAMOUS_OG_SITE_NAMES.iter().any(|m| low.contains(m)) {
        return Ok(Some(AgenticVerdict {
            verdict: VerdictKind::Cheat,
            cheat_codes: vec![CheatCode::ScrapedSiteClone],
            nearest_id: None,
            similarity_bps: 10_000,
            rationale: "sim: famous-site og:site_name in sanitized pages".into(),
        }));
    }
    for cluster in FAMOUS_SITE_CLUSTERS {
        let hits = cluster.iter().filter(|m| low.contains(*m)).count();
        if hits >= 2 {
            return Ok(Some(AgenticVerdict {
                verdict: VerdictKind::Cheat,
                cheat_codes: vec![CheatCode::ScrapedSiteClone],
                nearest_id: None,
                similarity_bps: 10_000,
                rationale: format!("sim: famous-site scrape markers ({hits} hits)"),
            }));
        }
    }
    // Thin republish wrapper: agent fetches a known site and dumps HTML.
    if agent_fetches_famous_site(req, &workdir) && low.contains("<html") {
        return Ok(Some(AgenticVerdict {
            verdict: VerdictKind::Cheat,
            cheat_codes: vec![CheatCode::ScrapedSiteClone],
            nearest_id: None,
            similarity_bps: 10_000,
            rationale: "sim: agent fetches famous site + emits HTML pages".into(),
        }));
    }
    Ok(None)
}

fn agent_fetches_famous_site(req: &ReviewRequest, workdir: &Path) -> bool {
    let domains = [
        "twitter.com",
        "x.com",
        "facebook.com",
        "instagram.com",
        "youtube.com",
        "linkedin.com",
        "netflix.com",
        "amazon.com",
        "github.com",
    ];
    for rel in &req.primary_relpaths {
        let Ok(path) = resolve_rel(workdir, rel) else {
            continue;
        };
        let Ok(src) = fs::read_to_string(&path) else {
            continue;
        };
        let low = src.to_ascii_lowercase();
        let fetches = low.contains("requests.get")
            || low.contains("urllib.request")
            || low.contains("httpx.get")
            || low.contains("aiohttp");
        // Also catch hardcoded HTML clones embedded in agent.py (no network fetch).
        let embeds_html = low.contains("<html") || low.contains("<!doctype");
        if !(fetches || embeds_html) {
            continue;
        }
        if domains.iter().any(|d| low.contains(d)) {
            return true;
        }
    }
    false
}

/// Hardcoded `METRICS_JSON` in miner sources or impossible metrics payload.
fn metrics_cheat_verdict(
    req: &ReviewRequest,
    primaries: &[(String, String)],
) -> Result<Option<AgenticVerdict>, AgenticError> {
    for (path, src) in primaries {
        if src.contains("METRICS_JSON=") {
            return Ok(Some(cheat_verdict(
                CheatCode::EvalShortCircuit,
                format!("sim: hardcoded METRICS_JSON in {path}"),
            )));
        }
    }
    let Some(rel) = &req.metrics_relpath else {
        return Ok(None);
    };
    let workdir = req
        .workdir
        .canonicalize()
        .map_err(|e| AgenticError::Tool(format!("workdir: {e}")))?;
    let path = resolve_rel(&workdir, rel)?;
    let Ok(text) = fs::read_to_string(&path) else {
        return Ok(None);
    };
    let Ok(v) = serde_json::from_str::<Value>(&text) else {
        return Ok(Some(cheat_verdict(
            CheatCode::InconsistentMetrics,
            "sim: metrics json unparseable",
        )));
    };
    let bpb = v.get("bpb").and_then(Value::as_f64);
    let tokens = v.get("tokens_seen").and_then(Value::as_u64).unwrap_or(0);
    let wall = v
        .get("wall_clock_seconds")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let Some(bpb) = bpb else {
        return Ok(Some(cheat_verdict(
            CheatCode::InconsistentMetrics,
            "sim: metrics missing bpb",
        )));
    };
    if !bpb.is_finite() || bpb <= 0.0 {
        return Ok(Some(cheat_verdict(
            CheatCode::InconsistentMetrics,
            format!("sim: non-finite/non-positive bpb={bpb}"),
        )));
    }
    if tokens == 0 || (bpb < SIM_IMPOSSIBLE_BPB && tokens < 64) {
        return Ok(Some(cheat_verdict(
            CheatCode::InconsistentMetrics,
            format!("sim: impossible bpb={bpb} tokens_seen={tokens}"),
        )));
    }
    if bpb < SIM_ABSURD_LOW_BPB {
        return Ok(Some(cheat_verdict(
            CheatCode::NonCausalLabelLeak,
            format!("sim: absurd bpb={bpb} (<{SIM_ABSURD_LOW_BPB}) label leak/forge"),
        )));
    }
    if wall < 0.01 && tokens > 10_000 {
        return Ok(Some(cheat_verdict(
            CheatCode::InconsistentMetrics,
            format!("sim: impossible wall_clock={wall} tokens_seen={tokens}"),
        )));
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use challenge_agentic_types::CorpusEntry;
    use std::fs;
    use tempfile::tempdir;

    fn write_primary(dir: &std::path::Path, name: &str, body: &str) -> ReviewRequest {
        fs::write(dir.join(name), body).unwrap();
        ReviewRequest {
            workdir: dir.to_path_buf(),
            primary_relpaths: vec![name.into()],
            corpus: vec![],
            metrics_relpath: None,
            pages_relpath: None,
            sanitize_report_relpath: None,
            domain_rules: "design".into(),
        }
    }

    const BASELINE: &str = r"
import json

def build_pages(ctx):
    return {'index.html': '<html>ok</html>'}

class Agent:
    def run(self):
        return build_pages({})
";

    const UNRELATED: &str = r"
import math

def train(model, data):
    total = 0.0
    for x in data:
        total += math.sin(x)
    return total
";

    #[tokio::test]
    async fn sim_byte_copy_is_cheat() {
        let dir = tempdir().unwrap();
        let mut req = write_primary(dir.path(), "agent.py", BASELINE);
        req.corpus = vec![CorpusEntry {
            id: "harness:victim".into(),
            source: BASELINE.into(),
        }];
        let v = SimAgent::new().review(&req).await.unwrap();
        assert_eq!(v.verdict, VerdictKind::Cheat);
        assert_eq!(v.similarity_bps, 10_000);
        assert_eq!(v.nearest_id.as_deref(), Some("harness:victim"));
    }

    #[tokio::test]
    async fn sim_baseline_byte_copy_is_clean() {
        // Baseline-zeroing fix: starting from the published baseline is not a
        // cheat, even byte-for-byte vs the reference corpus entry.
        let dir = tempdir().unwrap();
        let mut req = write_primary(dir.path(), "agent.py", BASELINE);
        req.corpus = vec![CorpusEntry {
            id: "baseline".into(),
            source: BASELINE.into(),
        }];
        let v = SimAgent::new().review(&req).await.unwrap();
        assert_eq!(v.verdict, VerdictKind::Clean);
    }

    #[tokio::test]
    async fn sim_unrelated_is_clean() {
        let dir = tempdir().unwrap();
        let mut req = write_primary(dir.path(), "agent.py", UNRELATED);
        req.corpus = vec![CorpusEntry {
            id: "baseline".into(),
            source: BASELINE.into(),
        }];
        let v = SimAgent::new().review(&req).await.unwrap();
        assert_eq!(v.verdict, VerdictKind::Clean);
        assert!(v.similarity_bps < SIM_SUSPICIOUS_BPS);
    }

    #[tokio::test]
    async fn sim_no_primary_fail_closed() {
        let dir = tempdir().unwrap();
        let req = ReviewRequest {
            workdir: dir.path().to_path_buf(),
            primary_relpaths: vec![],
            corpus: vec![],
            metrics_relpath: None,
            pages_relpath: None,
            sanitize_report_relpath: None,
            domain_rules: String::new(),
        };
        let err = SimAgent::new().review(&req).await.unwrap_err();
        assert!(matches!(err, AgenticError::NoVerdict(_)));
    }

    #[tokio::test]
    async fn sim_hardcoded_metrics_json_is_cheat() {
        let dir = tempdir().unwrap();
        let body = r#"
def train(model, ctx):
    print('METRICS_JSON={"bpb":0.01,"tokens_seen":0}')
    return {}
"#;
        let mut req = write_primary(dir.path(), "training.py", body);
        req.domain_rules = "prism".into();
        let v = SimAgent::new().review(&req).await.unwrap();
        assert_eq!(v.verdict, VerdictKind::Cheat);
        assert!(v.cheat_codes.contains(&CheatCode::EvalShortCircuit));
    }

    #[tokio::test]
    async fn sim_impossible_bpb_metrics_is_cheat() {
        let dir = tempdir().unwrap();
        let mut req = write_primary(dir.path(), "architecture.py", UNRELATED);
        fs::write(
            dir.path().join("metrics.json"),
            r#"{"bpb":0.01,"tokens_seen":0,"wall_clock_seconds":0.0,"notes":"forged"}"#,
        )
        .unwrap();
        req.metrics_relpath = Some("metrics.json".into());
        req.domain_rules = "prism".into();
        let v = SimAgent::new().review(&req).await.unwrap();
        assert_eq!(v.verdict, VerdictKind::Cheat);
        assert!(v.cheat_codes.contains(&CheatCode::InconsistentMetrics));
    }

    #[tokio::test]
    async fn sim_famous_site_scrape_pages_is_cheat() {
        let dir = tempdir().unwrap();
        let agent = r#"
def run():
    return {"index.html": "<html></html>"}
"#;
        let mut req = write_primary(dir.path(), "agent.py", agent);
        fs::create_dir_all(dir.path().join("pages")).unwrap();
        fs::write(
            dir.path().join("pages/index.html"),
            r#"<!DOCTYPE html>
<html>
<head>
  <meta property="og:site_name" content="Twitter">
  <title>Twitter</title>
</head>
<body>
  <div data-testid="tweet">Hello from twitter.com clone</div>
</body>
</html>"#,
        )
        .unwrap();
        req.pages_relpath = Some("pages".into());
        req.domain_rules = "design".into();
        let v = SimAgent::new().review(&req).await.unwrap();
        assert_eq!(v.verdict, VerdictKind::Cheat);
        assert!(v.cheat_codes.contains(&CheatCode::ScrapedSiteClone));
    }

    #[tokio::test]
    async fn sim_sanitized_twitter_x_links_are_cheat() {
        // After sanitize, og:/data-testid markers are often stripped; twitter.com + x.com remain.
        let dir = tempdir().unwrap();
        let agent = r#"
def run(task, llm, out):
    html = "<html><a href='https://twitter.com'>t</a><a href='https://x.com'>x</a></html>"
    out.write_page("index.html", html)
"#;
        let mut req = write_primary(dir.path(), "agent.py", agent);
        fs::create_dir_all(dir.path().join("pages")).unwrap();
        fs::write(
            dir.path().join("pages/index.html"),
            r#"<!DOCTYPE html><html><body>
<a href="https://twitter.com/home" rel="noopener noreferrer">twitter.com</a>
<a href="https://x.com" rel="noopener noreferrer">x.com</a>
<div>timeline clone</div>
</body></html>"#,
        )
        .unwrap();
        req.pages_relpath = Some("pages".into());
        req.domain_rules = "design".into();
        let v = SimAgent::new().review(&req).await.unwrap();
        assert_eq!(v.verdict, VerdictKind::Cheat);
        assert!(v.cheat_codes.contains(&CheatCode::ScrapedSiteClone));
    }

    #[tokio::test]
    async fn sim_prism_missing_telemetry_hooks_is_cheat() {
        let dir = tempdir().unwrap();
        let body = r#"
def train(model, ctx):
    model.zero_grad()
    return {"loss": 1.0}
"#;
        let mut req = write_primary(dir.path(), "training.py", body);
        req.domain_rules = crate::PRISM_DOMAIN_RULES.into();
        let v = SimAgent::new().review(&req).await.unwrap();
        assert_eq!(v.verdict, VerdictKind::Cheat);
        assert!(v.cheat_codes.contains(&CheatCode::MissingTelemetryHooks));
    }

    #[tokio::test]
    async fn sim_prism_with_telemetry_hooks_not_flagged() {
        let dir = tempdir().unwrap();
        let body = r#"
import prism_telemetry

def train(model, ctx):
    prism_telemetry.report(loss=1.0, step=1)
    prism_telemetry.finish_evaluation()
    return {"loss": 1.0}
"#;
        let mut req = write_primary(dir.path(), "training.py", body);
        req.domain_rules = crate::PRISM_DOMAIN_RULES.into();
        let v = SimAgent::new().review(&req).await.unwrap();
        assert_eq!(v.verdict, VerdictKind::Clean);
    }

    #[tokio::test]
    async fn sim_prism_tokenmix_label_leak_is_cheat() {
        // Sanitized prod class (`b99a7047`): dense `TokenMix` over time, no causal mask.
        let dir = tempdir().unwrap();
        let arch = r"
import torch.nn as nn
class TokenMix(nn.Module):
    def __init__(self, seq, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(seq, hidden), nn.GELU(), nn.Linear(hidden, seq))
    def forward(self, x):
        return self.net(x.transpose(1, 2)).transpose(1, 2)
def build_model(ctx):
    return TokenMix(512, 1024)
";
        let train = r"
import prism_telemetry
def train(model, ctx):
    prism_telemetry.report(loss=1.0, step=1)
    prism_telemetry.finish_evaluation()
    return {'loss': 1.0}
";
        fs::write(dir.path().join("architecture.py"), arch).unwrap();
        fs::write(dir.path().join("training.py"), train).unwrap();
        let req = ReviewRequest {
            workdir: dir.path().to_path_buf(),
            primary_relpaths: vec!["architecture.py".into(), "training.py".into()],
            corpus: vec![],
            metrics_relpath: None,
            pages_relpath: None,
            sanitize_report_relpath: None,
            domain_rules: crate::PRISM_DOMAIN_RULES.into(),
        };
        let v = SimAgent::new().review(&req).await.unwrap();
        assert_eq!(v.verdict, VerdictKind::Cheat);
        assert!(v.cheat_codes.contains(&CheatCode::NonCausalLabelLeak));
    }

    #[tokio::test]
    async fn sim_absurd_low_bpb_is_label_leak() {
        let dir = tempdir().unwrap();
        let arch = "import torch\ndef build_model(ctx):\n    return torch.nn.Linear(8, 8)\n";
        let train = r"
import prism_telemetry
def train(model, ctx):
    prism_telemetry.report(loss=1.0, step=1)
    prism_telemetry.finish_evaluation()
    return {'loss': 1.0}
";
        fs::write(dir.path().join("architecture.py"), arch).unwrap();
        fs::write(dir.path().join("training.py"), train).unwrap();
        fs::write(
            dir.path().join("metrics.json"),
            r#"{"bpb":0.23,"tokens_seen":2048,"wall_clock_seconds":540.0,"notes":"recipe-v1"}"#,
        )
        .unwrap();
        let req = ReviewRequest {
            workdir: dir.path().to_path_buf(),
            primary_relpaths: vec!["architecture.py".into(), "training.py".into()],
            corpus: vec![],
            metrics_relpath: Some("metrics.json".into()),
            pages_relpath: None,
            sanitize_report_relpath: None,
            domain_rules: crate::PRISM_DOMAIN_RULES.into(),
        };
        let v = SimAgent::new().review(&req).await.unwrap();
        assert_eq!(v.verdict, VerdictKind::Cheat);
        assert!(v.cheat_codes.contains(&CheatCode::NonCausalLabelLeak));
    }

    #[tokio::test]
    async fn sim_design_training_py_not_hooks_checked() {
        // Hooks rule is Prism-only; design primaries never trip it.
        let dir = tempdir().unwrap();
        let mut req = write_primary(dir.path(), "training.py", "def train():\n    pass\n");
        req.domain_rules = "design".into();
        let v = SimAgent::new().review(&req).await.unwrap();
        assert_eq!(v.verdict, VerdictKind::Clean);
    }

    #[tokio::test]
    async fn sim_baseline_style_pages_are_clean() {
        let dir = tempdir().unwrap();
        let mut req = write_primary(dir.path(), "agent.py", BASELINE);
        fs::create_dir_all(dir.path().join("pages")).unwrap();
        fs::write(
            dir.path().join("pages/index.html"),
            r#"<!DOCTYPE html>
<html data-agent="design-baseline">
<head><title>Home</title><style>.hero{color:#111}</style></head>
<body><header class="hero"><h1>Product</h1></header></body>
</html>"#,
        )
        .unwrap();
        req.pages_relpath = Some("pages".into());
        req.corpus = vec![CorpusEntry {
            id: "other".into(),
            source: UNRELATED.into(),
        }];
        req.domain_rules = "design".into();
        let v = SimAgent::new().review(&req).await.unwrap();
        assert_eq!(v.verdict, VerdictKind::Clean);
    }
}
