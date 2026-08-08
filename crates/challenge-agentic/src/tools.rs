//! Read-only workdir tools + AST helpers for the agentic loop.

use std::fmt::Write as _;
use std::fs;
use std::path::{Component, Path, PathBuf};

use challenge_ast::{
    fingerprint_source, structural_diff_summary, summarize_fingerprint, top_k_nearest, Fingerprint,
    AST_CHEAT_BPS, AST_SUSPICIOUS_BPS,
};
use serde_json::{json, Value};

use crate::types::{
    AgenticError, AgenticVerdict, CheatCode, CorpusEntry, ReviewRequest, VerdictKind,
};

/// Precomputed corpus fingerprints for AST tools.
#[derive(Debug, Clone)]
pub(crate) struct ToolContext {
    pub workdir: PathBuf,
    pub corpus: Vec<(String, Fingerprint)>,
    pub metrics_relpath: Option<String>,
    pub pages_relpath: Option<String>,
    pub sanitize_report_relpath: Option<String>,
    pub enable_run_command: bool,
}

impl ToolContext {
    pub(crate) fn from_request(req: &ReviewRequest) -> Result<Self, AgenticError> {
        let workdir = req
            .workdir
            .canonicalize()
            .map_err(|e| AgenticError::Tool(format!("workdir: {e}")))?;
        if !workdir.is_dir() {
            return Err(AgenticError::Tool("workdir is not a directory".into()));
        }
        let mut corpus = Vec::with_capacity(req.corpus.len());
        for entry in &req.corpus {
            // Skip unparseable corpus entries; still usable for byte-hash sim.
            if let Ok(fp) = fingerprint_source(&entry.source) {
                corpus.push((entry.id.clone(), fp));
            }
        }
        // `run_command` exists only inside the hardened review container (the
        // `challenge-review` image sets this); never on the master process.
        let enable_run_command = matches!(
            std::env::var("AGENTIC_ENABLE_RUN_COMMAND").as_deref(),
            Ok("1" | "true" | "TRUE" | "yes" | "YES")
        );
        Ok(Self {
            workdir,
            corpus,
            metrics_relpath: req.metrics_relpath.clone(),
            pages_relpath: req.pages_relpath.clone(),
            sanitize_report_relpath: req.sanitize_report_relpath.clone(),
            enable_run_command,
        })
    }
}

/// OpenAI/OpenRouter tool schemas for the chat loop.
#[must_use]
#[allow(clippy::too_many_lines)] // JSON schema literals for every tool.
pub(crate) fn tool_schemas() -> Value {
    let list_dir = json!({
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative directory path (default \".\")."}
        },
        "required": []
    });
    let read_file = json!({
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "max_bytes": {"type": "integer", "minimum": 1, "maximum": 65536}
        },
        "required": ["path"]
    });
    let grep_files = json!({
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "description": "Relative root (default \".\")."},
            "max_hits": {"type": "integer", "minimum": 1, "maximum": 50}
        },
        "required": ["pattern"]
    });
    let file_stat = json!({
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"]
    });
    let ast_summary = json!({
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"]
    });
    let ast_diff = json!({
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "k": {"type": "integer", "minimum": 1, "maximum": 10}
        },
        "required": ["path"]
    });
    let run_command = json!({
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command run by bash inside the hardened review sandbox (cwd = workdir, scrubbed env, 15s cap, truncated output)."},
            "path": {"type": "string", "description": "Relative cwd under the workdir (default \".\")."}
        },
        "required": ["command"]
    });
    let empty = json!({"type": "object", "properties": {}, "required": []});
    let submit = json!({
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["clean", "suspicious", "cheat"]},
            "cheat_codes": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "near_identical_harness_copy",
                        "trivial_republish_wrapper",
                        "scraped_site_clone",
                        "sanitize_bypass",
                        "obfuscation_to_hide_copy",
                        "inconsistent_metrics",
                        "eval_short_circuit",
                        "ast_architecture_copy"
                    ]
                }
            },
            "nearest_id": {"type": ["string", "null"]},
            "similarity_bps": {"type": "integer", "minimum": 0, "maximum": 10000},
            "rationale": {"type": "string"}
        },
        "required": ["verdict", "cheat_codes", "similarity_bps", "rationale"]
    });
    json!([
        fn_tool(
            "list_dir",
            "List entries in a directory under the workdir.",
            &list_dir
        ),
        fn_tool(
            "read_file",
            "Read a UTF-8 text file (truncated).",
            &read_file
        ),
        fn_tool(
            "grep_files",
            "Search a regex-ish substring under a relative root.",
            &grep_files
        ),
        fn_tool(
            "file_stat",
            "Stat a file or directory (size, is_dir).",
            &file_stat
        ),
        fn_tool(
            "ast_summary",
            "Fingerprint + summarize a Python file under workdir.",
            &ast_summary
        ),
        fn_tool(
            "ast_diff_nearest",
            "Top-k AST neighbors vs corpus for a Python file.",
            &ast_diff
        ),
        fn_tool(
            "read_metrics",
            "Read Prism metrics/receipt JSON if configured.",
            &empty
        ),
        fn_tool(
            "read_pages",
            "List Design sanitized pages + optional sanitize report.",
            &empty
        ),
        fn_tool(
            "run_command",
            "Run a short shell command inside the review sandbox (both agents mounted). Use for diffs, grep, python AST probes.",
            &run_command
        ),
        fn_tool(
            "submit_verdict",
            "MANDATORY final call. Emit the anti-cheat verdict. No further tools after this.",
            &submit,
        )
    ])
}

fn fn_tool(name: &str, description: &str, parameters: &Value) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters
        }
    })
}

/// Dispatch one tool call. `Ok(Err(verdict))` means `submit_verdict` succeeded.
pub(crate) fn dispatch(
    ctx: &ToolContext,
    name: &str,
    args: &Value,
) -> Result<Result<String, AgenticVerdict>, AgenticError> {
    match name {
        "list_dir" => Ok(Ok(tool_list_dir(ctx, args)?)),
        "read_file" => Ok(Ok(tool_read_file(ctx, args)?)),
        "grep_files" => Ok(Ok(tool_grep(ctx, args)?)),
        "file_stat" => Ok(Ok(tool_stat(ctx, args)?)),
        "ast_summary" => Ok(Ok(tool_ast_summary(ctx, args)?)),
        "ast_diff_nearest" => Ok(Ok(tool_ast_diff(ctx, args)?)),
        "read_metrics" => Ok(Ok(tool_read_metrics(ctx)?)),
        "read_pages" => Ok(Ok(tool_read_pages(ctx)?)),
        "run_command" => Ok(Ok(tool_run_command(ctx, args)?)),
        "submit_verdict" => Ok(Err(parse_verdict(args)?)),
        other => Err(AgenticError::Tool(format!("unknown tool: {other}"))),
    }
}

fn tool_list_dir(ctx: &ToolContext, args: &Value) -> Result<String, AgenticError> {
    let rel = args.get("path").and_then(Value::as_str).unwrap_or(".");
    let path = resolve_rel(&ctx.workdir, rel)?;
    let mut names = Vec::new();
    let rd = fs::read_dir(&path).map_err(|e| AgenticError::Tool(format!("list_dir: {e}")))?;
    for ent in rd.flatten().take(200) {
        let name = ent.file_name().to_string_lossy().into_owned();
        let suffix = if ent.file_type().is_ok_and(|t| t.is_dir()) {
            "/"
        } else {
            ""
        };
        names.push(format!("{name}{suffix}"));
    }
    names.sort();
    Ok(names.join("\n"))
}

fn tool_read_file(ctx: &ToolContext, args: &Value) -> Result<String, AgenticError> {
    let rel = args
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| AgenticError::Tool("read_file: path required".into()))?;
    let max = usize::try_from(
        args.get("max_bytes")
            .and_then(Value::as_u64)
            .map_or(24_576, |n| n.min(65_536)),
    )
    .unwrap_or(24_576);
    let path = resolve_rel(&ctx.workdir, rel)?;
    let bytes = fs::read(&path).map_err(|e| AgenticError::Tool(format!("read_file: {e}")))?;
    let slice = if bytes.len() > max {
        &bytes[..max]
    } else {
        &bytes
    };
    let mut text = String::from_utf8_lossy(slice).into_owned();
    if bytes.len() > max {
        text.push_str("\n…[truncated]");
    }
    Ok(text)
}

fn tool_grep(ctx: &ToolContext, args: &Value) -> Result<String, AgenticError> {
    let pattern = args
        .get("pattern")
        .and_then(Value::as_str)
        .ok_or_else(|| AgenticError::Tool("grep_files: pattern required".into()))?;
    if pattern.is_empty() || pattern.len() > 128 {
        return Err(AgenticError::Tool("grep_files: bad pattern".into()));
    }
    let rel = args.get("path").and_then(Value::as_str).unwrap_or(".");
    let max_hits = usize::try_from(
        args.get("max_hits")
            .and_then(Value::as_u64)
            .map_or(20, |n| n.min(50)),
    )
    .unwrap_or(20);
    let root = resolve_rel(&ctx.workdir, rel)?;
    let mut hits = Vec::new();
    grep_walk(&root, &ctx.workdir, pattern, &mut hits, max_hits)?;
    if hits.is_empty() {
        Ok("(no hits)".into())
    } else {
        Ok(hits.join("\n"))
    }
}

fn grep_walk(
    path: &Path,
    workdir: &Path,
    pattern: &str,
    hits: &mut Vec<String>,
    max_hits: usize,
) -> Result<(), AgenticError> {
    if hits.len() >= max_hits {
        return Ok(());
    }
    let meta = fs::metadata(path).map_err(|e| AgenticError::Tool(format!("grep: {e}")))?;
    if meta.is_dir() {
        for ent in fs::read_dir(path)
            .map_err(|e| AgenticError::Tool(format!("grep: {e}")))?
            .flatten()
        {
            grep_walk(&ent.path(), workdir, pattern, hits, max_hits)?;
            if hits.len() >= max_hits {
                break;
            }
        }
        return Ok(());
    }
    if meta.len() > 512 * 1024 {
        return Ok(());
    }
    let Ok(text) = fs::read_to_string(path) else {
        return Ok(());
    };
    let rel = path
        .strip_prefix(workdir)
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned();
    for (i, line) in text.lines().enumerate() {
        if line.contains(pattern) {
            hits.push(format!("{rel}:{}:{line}", i + 1));
            if hits.len() >= max_hits {
                break;
            }
        }
    }
    Ok(())
}

fn tool_stat(ctx: &ToolContext, args: &Value) -> Result<String, AgenticError> {
    let rel = args
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| AgenticError::Tool("file_stat: path required".into()))?;
    let path = resolve_rel(&ctx.workdir, rel)?;
    let meta = fs::metadata(&path).map_err(|e| AgenticError::Tool(format!("file_stat: {e}")))?;
    Ok(format!(
        "path={rel} is_dir={} len={}",
        meta.is_dir(),
        meta.len()
    ))
}

fn tool_ast_summary(ctx: &ToolContext, args: &Value) -> Result<String, AgenticError> {
    let source = read_py(ctx, args)?;
    let fp = fingerprint_source(&source).map_err(|e| AgenticError::Tool(e.to_string()))?;
    Ok(summarize_fingerprint(&fp))
}

fn tool_ast_diff(ctx: &ToolContext, args: &Value) -> Result<String, AgenticError> {
    let source = read_py(ctx, args)?;
    let fp = fingerprint_source(&source).map_err(|e| AgenticError::Tool(e.to_string()))?;
    let k = usize::try_from(
        args.get("k")
            .and_then(Value::as_u64)
            .map_or(3, |n| n.clamp(1, 10)),
    )
    .unwrap_or(3);
    if ctx.corpus.is_empty() {
        return Ok(format!(
            "corpus_empty summary={}",
            summarize_fingerprint(&fp)
        ));
    }
    let nearest = top_k_nearest(&fp, &ctx.corpus, k);
    let mut lines = Vec::new();
    for n in nearest {
        lines.push(format!(
            "id={} similarity_bps={} {}",
            n.id, n.similarity_bps, n.summary
        ));
    }
    if let Some(first) = ctx.corpus.first() {
        let _ = structural_diff_summary(&fp, &first.1);
    }
    Ok(lines.join("\n"))
}

fn tool_read_metrics(ctx: &ToolContext) -> Result<String, AgenticError> {
    let Some(rel) = &ctx.metrics_relpath else {
        return Ok("(no metrics configured)".into());
    };
    let path = resolve_rel(&ctx.workdir, rel)?;
    let text =
        fs::read_to_string(&path).map_err(|e| AgenticError::Tool(format!("metrics: {e}")))?;
    Ok(truncate(&text, 24_576))
}

fn tool_read_pages(ctx: &ToolContext) -> Result<String, AgenticError> {
    let mut out = String::new();
    if let Some(rel) = &ctx.pages_relpath {
        let path = resolve_rel(&ctx.workdir, rel)?;
        if path.is_dir() {
            let mut names = Vec::new();
            for ent in fs::read_dir(&path)
                .map_err(|e| AgenticError::Tool(format!("pages: {e}")))?
                .flatten()
            {
                names.push(ent.file_name().to_string_lossy().into_owned());
            }
            names.sort();
            out.push_str("pages:\n");
            out.push_str(&names.join("\n"));
        } else {
            let _ = write!(out, "pages_path_missing={rel}");
        }
    } else {
        out.push_str("(no pages configured)");
    }
    if let Some(rel) = &ctx.sanitize_report_relpath {
        out.push_str("\n--- sanitize_report ---\n");
        match resolve_rel(&ctx.workdir, rel).and_then(|p| {
            fs::read_to_string(p).map_err(|e| AgenticError::Tool(format!("sanitize: {e}")))
        }) {
            Ok(t) => out.push_str(&truncate(&t, 16_384)),
            Err(e) => out.push_str(&e.to_string()),
        }
    }
    Ok(out)
}

/// Wall-clock cap for one `run_command` call.
const RUN_COMMAND_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(15);
/// Output cap (combined stdout+stderr).
const RUN_COMMAND_MAX_OUT: usize = 16 * 1024;

/// Sandboxed shell for the review container: fixed workdir cwd, scrubbed env
/// (no API keys), hard timeout, truncated output; refuses procfs and
/// review-secrets paths so file-mounted `OpenRouter` keys stay unread.
fn tool_run_command(ctx: &ToolContext, args: &Value) -> Result<String, AgenticError> {
    if !ctx.enable_run_command {
        return Err(AgenticError::Tool(
            "run_command disabled outside the review container".into(),
        ));
    }
    let cmd = args
        .get("command")
        .and_then(Value::as_str)
        .ok_or_else(|| AgenticError::Tool("run_command: command required".into()))?;
    if cmd.is_empty() || cmd.len() > 1_000 || cmd.contains('\0') {
        return Err(AgenticError::Tool("run_command: bad command".into()));
    }
    // File-mounted OpenRouter key + parent environ must stay unread.
    let c = cmd.to_ascii_lowercase().replace('\\', "/");
    if c.contains("/proc") || c.contains("review-secrets") || c.contains("openrouter_api_key") {
        return Err(AgenticError::Tool("run_command: forbidden path".into()));
    }
    let rel = args.get("path").and_then(Value::as_str).unwrap_or(".");
    let cwd = resolve_rel(&ctx.workdir, rel)?;
    if !cwd.is_dir() {
        return Err(AgenticError::Tool(
            "run_command: cwd not a directory".into(),
        ));
    }
    let mut child = std::process::Command::new("bash")
        .arg("-lc")
        .arg(cmd)
        .current_dir(&cwd)
        .env_clear()
        .env("PATH", "/usr/local/bin:/usr/bin:/bin")
        .env("HOME", "/tmp")
        .env("TMPDIR", "/tmp")
        .env("LANG", "C.UTF-8")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| AgenticError::Tool(format!("run_command spawn: {e}")))?;
    let deadline = std::time::Instant::now() + RUN_COMMAND_TIMEOUT;
    let mut timed_out = false;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if std::time::Instant::now() >= deadline => {
                timed_out = true;
                let _ = child.kill();
                break;
            }
            Ok(None) => std::thread::sleep(std::time::Duration::from_millis(25)),
            Err(e) => {
                let _ = child.kill();
                return Err(AgenticError::Tool(format!("run_command wait: {e}")));
            }
        }
    }
    let out = child
        .wait_with_output()
        .map_err(|e| AgenticError::Tool(format!("run_command output: {e}")))?;
    let status = out.status.code().unwrap_or(-1);
    let mut text = format!(
        "exit={status} timed_out={timed_out}\n$ {cmd}\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    if text.len() > RUN_COMMAND_MAX_OUT {
        text = truncate(&text, RUN_COMMAND_MAX_OUT);
    }
    Ok(text)
}

fn read_py(ctx: &ToolContext, args: &Value) -> Result<String, AgenticError> {
    let rel = args
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| AgenticError::Tool("path required".into()))?;
    let path = resolve_rel(&ctx.workdir, rel)?;
    fs::read_to_string(&path).map_err(|e| AgenticError::Tool(format!("read py: {e}")))
}

/// Resolve `rel` under `workdir`; reject `..` / absolute / symlink escapes.
pub(crate) fn resolve_rel(workdir: &Path, rel: &str) -> Result<PathBuf, AgenticError> {
    let rel = rel.trim();
    if rel.is_empty() {
        return Err(AgenticError::Tool("empty path".into()));
    }
    let candidate = Path::new(rel);
    if candidate.is_absolute() {
        return Err(AgenticError::Tool("absolute paths forbidden".into()));
    }
    for c in candidate.components() {
        match c {
            Component::Normal(_) | Component::CurDir => {}
            Component::ParentDir => {
                return Err(AgenticError::Tool("path escape (..) forbidden".into()));
            }
            _ => return Err(AgenticError::Tool("invalid path component".into())),
        }
    }
    let joined = workdir.join(candidate);
    let canon = joined
        .canonicalize()
        .map_err(|e| AgenticError::Tool(format!("resolve {rel}: {e}")))?;
    if !canon.starts_with(workdir) {
        return Err(AgenticError::Tool("path escapes workdir".into()));
    }
    Ok(canon)
}

pub(crate) fn parse_verdict(args: &Value) -> Result<AgenticVerdict, AgenticError> {
    let verdict = match args.get("verdict").and_then(Value::as_str).unwrap_or("") {
        "clean" => VerdictKind::Clean,
        "suspicious" => VerdictKind::Suspicious,
        "cheat" => VerdictKind::Cheat,
        other => {
            return Err(AgenticError::Parse(format!("verdict: {other:?}")));
        }
    };
    let cheat_codes = args
        .get("cheat_codes")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str())
                .map(parse_cheat_code)
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?
        .unwrap_or_default();
    let nearest_id = args.get("nearest_id").and_then(|v| {
        if v.is_null() {
            None
        } else {
            v.as_str().map(|s| s.chars().take(128).collect())
        }
    });
    let similarity_bps = args
        .get("similarity_bps")
        .and_then(Value::as_u64)
        .ok_or_else(|| AgenticError::Parse("similarity_bps missing".into()))?;
    if similarity_bps > 10_000 {
        return Err(AgenticError::Parse("similarity_bps > 10000".into()));
    }
    let rationale = args
        .get("rationale")
        .and_then(Value::as_str)
        .unwrap_or("")
        .chars()
        .take(2_000)
        .collect::<String>();
    if rationale.is_empty() {
        return Err(AgenticError::Parse("rationale empty".into()));
    }
    Ok(normalize_ast_thresholds(AgenticVerdict {
        verdict,
        cheat_codes,
        nearest_id,
        similarity_bps: u16::try_from(similarity_bps).unwrap_or(10_000),
        rationale,
    }))
}

/// Align `OpenRouter` `submit_verdict` with `SimAgent` AST bands when the only
/// signal is AST / near-identical copy (non-AST cheat codes are left alone).
fn normalize_ast_thresholds(mut v: AgenticVerdict) -> AgenticVerdict {
    let has_non_ast = v.cheat_codes.iter().any(|c| {
        !matches!(
            c,
            CheatCode::NearIdenticalHarnessCopy | CheatCode::AstArchitectureCopy
        )
    });
    if has_non_ast {
        return v;
    }
    if v.similarity_bps >= AST_CHEAT_BPS {
        v.verdict = VerdictKind::Cheat;
        if v.cheat_codes.is_empty() {
            v.cheat_codes.push(CheatCode::AstArchitectureCopy);
        }
    } else if v.similarity_bps >= AST_SUSPICIOUS_BPS {
        if matches!(v.verdict, VerdictKind::Clean | VerdictKind::Cheat) {
            v.verdict = VerdictKind::Suspicious;
        }
        if v.cheat_codes.is_empty() {
            v.cheat_codes.push(CheatCode::NearIdenticalHarnessCopy);
        }
    } else if matches!(v.verdict, VerdictKind::Suspicious | VerdictKind::Cheat) {
        // Below suspicious band: AST-only must not Score(0).
        v.verdict = VerdictKind::Clean;
        v.cheat_codes.clear();
    }
    v
}

fn parse_cheat_code(s: &str) -> Result<CheatCode, AgenticError> {
    Ok(match s {
        "near_identical_harness_copy" => CheatCode::NearIdenticalHarnessCopy,
        "trivial_republish_wrapper" => CheatCode::TrivialRepublishWrapper,
        "scraped_site_clone" => CheatCode::ScrapedSiteClone,
        "sanitize_bypass" => CheatCode::SanitizeBypass,
        "obfuscation_to_hide_copy" => CheatCode::ObfuscationToHideCopy,
        "inconsistent_metrics" => CheatCode::InconsistentMetrics,
        "eval_short_circuit" => CheatCode::EvalShortCircuit,
        "ast_architecture_copy" => CheatCode::AstArchitectureCopy,
        "missing_telemetry_hooks" => CheatCode::MissingTelemetryHooks,
        "non_causal_label_leak" => CheatCode::NonCausalLabelLeak,
        other => return Err(AgenticError::Parse(format!("cheat_code: {other:?}"))),
    })
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_owned()
    } else {
        let mut n = max;
        while n > 0 && !s.is_char_boundary(n) {
            n -= 1;
        }
        format!("{}…[truncated]", &s[..n])
    }
}

/// SHA-256 hex of source bytes (sim + optional callers).
#[must_use]
pub(crate) fn source_hash_hex(source: &str) -> String {
    use sha2::{Digest, Sha256};
    let d = Sha256::digest(source.as_bytes());
    hex::encode(d)
}

/// Load primary Python sources from the request workdir.
pub(crate) fn load_primary_sources(
    req: &ReviewRequest,
) -> Result<Vec<(String, String)>, AgenticError> {
    let workdir = req
        .workdir
        .canonicalize()
        .map_err(|e| AgenticError::Tool(format!("workdir: {e}")))?;
    let mut out = Vec::new();
    for rel in &req.primary_relpaths {
        let path = resolve_rel(&workdir, rel)?;
        let text = fs::read_to_string(&path)
            .map_err(|e| AgenticError::Tool(format!("primary {rel}: {e}")))?;
        out.push((rel.clone(), text));
    }
    Ok(out)
}

#[allow(dead_code)] // used by sim + tests via CorpusEntry
pub(crate) fn corpus_hashes(corpus: &[CorpusEntry]) -> Vec<(String, String)> {
    corpus
        .iter()
        .map(|c| (c.id.clone(), source_hash_hex(&c.source)))
        .collect()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn resolve_blocks_escape() {
        let dir = tempdir().unwrap();
        let wd = dir.path().canonicalize().unwrap();
        assert!(resolve_rel(&wd, "../etc/passwd").is_err());
        assert!(resolve_rel(&wd, "/etc/passwd").is_err());
    }

    /// `AGENTIC_ENABLE_RUN_COMMAND` is process-global; serialize the two tests
    /// that mutate it so parallel libtest cannot race enable vs disable.
    static RUN_COMMAND_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn run_command_executes_in_sandboxed_cwd() {
        let _guard = RUN_COMMAND_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("agent.py"), "x = 1\n").unwrap();
        std::env::set_var("AGENTIC_ENABLE_RUN_COMMAND", "1");
        let req = ReviewRequest {
            workdir: dir.path().to_path_buf(),
            primary_relpaths: vec!["agent.py".into()],
            corpus: vec![],
            metrics_relpath: None,
            pages_relpath: None,
            sanitize_report_relpath: None,
            domain_rules: String::new(),
        };
        let ctx = ToolContext::from_request(&req).unwrap();
        std::env::remove_var("AGENTIC_ENABLE_RUN_COMMAND");
        assert!(ctx.enable_run_command);
        let out = tool_run_command(&ctx, &json!({"command": "cat agent.py"})).unwrap();
        assert!(out.contains("x = 1"), "out={out}");
        assert!(out.contains("exit=0"), "out={out}");
        // Env is scrubbed for children.
        let out = tool_run_command(&ctx, &json!({"command": "env | wc -l"})).unwrap();
        assert!(out.contains("exit=0"), "out={out}");
        // Escape attempts fail via resolve_rel on cwd.
        assert!(tool_run_command(&ctx, &json!({"command": "ls", "path": ".."})).is_err());
        // Parent environ / secrets exfil via shell is refused (defense-in-depth).
        for bad in [
            "cat /proc/1/environ",
            "python -c 'open(\"//proc/self/environ\").read()'",
            "cat /run/review-secrets/openrouter_api_key",
        ] {
            let err = tool_run_command(&ctx, &json!({"command": bad}))
                .unwrap_err()
                .to_string();
            assert!(err.contains("forbidden"), "cmd={bad} err={err}");
        }
    }

    #[test]
    fn run_command_disabled_without_container_env() {
        let _guard = RUN_COMMAND_ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::env::remove_var("AGENTIC_ENABLE_RUN_COMMAND");
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("agent.py"), "x = 1\n").unwrap();
        let req = ReviewRequest {
            workdir: dir.path().to_path_buf(),
            primary_relpaths: vec!["agent.py".into()],
            corpus: vec![],
            metrics_relpath: None,
            pages_relpath: None,
            sanitize_report_relpath: None,
            domain_rules: String::new(),
        };
        let ctx = ToolContext::from_request(&req).unwrap();
        assert!(!ctx.enable_run_command);
        assert!(tool_run_command(&ctx, &json!({"command": "true"})).is_err());
    }

    #[test]
    fn parse_verdict_ok() {
        let v = parse_verdict(&json!({
            "verdict": "cheat",
            "cheat_codes": ["near_identical_harness_copy"],
            "nearest_id": "baseline",
            "similarity_bps": 9900,
            "rationale": "byte copy"
        }))
        .unwrap();
        assert_eq!(v.verdict, VerdictKind::Cheat);
        assert_eq!(v.similarity_bps, 9900);
    }

    #[test]
    fn parse_verdict_fail_closed() {
        assert!(parse_verdict(&json!({"verdict": "clean"})).is_err());
        assert!(parse_verdict(&json!({
            "verdict": "clean",
            "cheat_codes": [],
            "similarity_bps": 10001,
            "rationale": "x"
        }))
        .is_err());
    }

    #[test]
    fn parse_verdict_coerces_ast_below_suspicious_to_clean() {
        let v = parse_verdict(&json!({
            "verdict": "suspicious",
            "cheat_codes": [],
            "nearest_id": "harness:abc",
            "similarity_bps": 8360,
            "rationale": "overlap"
        }))
        .unwrap();
        assert_eq!(v.verdict, VerdictKind::Clean);
        assert!(v.cheat_codes.is_empty());
    }
}
