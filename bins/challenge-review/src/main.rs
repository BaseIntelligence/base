//! `challenge-review` — entrypoint of the `design-review` container image.
//!
//! Reads the staged review request (`/work/_review_request.json`), runs the
//! agentic loop (`OpenRouter` when a key file / env key is present,
//! deterministic `SimAgent` otherwise), and writes the verdict JSON to
//! `/out/verdict.json`. Exit 0 with a verdict; non-zero on infra failure
//! (caller retries).

#![forbid(unsafe_code)]

use std::path::Path;
use std::process::ExitCode;

use challenge_agentic::{
    load_api_key_file, AgentConfig, AgenticBackend, ContainerReviewRequest, OpenRouterAgent,
    SimAgent, DEFAULT_MODEL, OPENROUTER_API_BASE,
};

fn main() -> ExitCode {
    // Prefer file mount so the key never appears in the process's initial
    // environ (`/proc/<pid>/environ` is boot-fixed on Linux).
    let openrouter_key = load_openrouter_key();
    let req_path = std::env::var("REVIEW_REQUEST_PATH")
        .unwrap_or_else(|_| "/work/_review_request.json".to_owned());
    let out_path =
        std::env::var("REVIEW_OUT_PATH").unwrap_or_else(|_| "/out/verdict.json".to_owned());
    match run(&req_path, &out_path, openrouter_key) {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("challenge-review: {msg}");
            ExitCode::from(2)
        }
    }
}

fn load_openrouter_key() -> Option<String> {
    if let Ok(path) = std::env::var("OPENROUTER_API_KEY_FILE") {
        if let Ok(key) = load_api_key_file(Path::new(&path)) {
            return Some(key);
        }
    }
    // Legacy env inject: take into memory and clear the live environ map
    // (does not rewrite `/proc/*/environ`).
    let key = std::env::var("OPENROUTER_API_KEY")
        .ok()
        .map(|k| k.trim().to_owned())
        .filter(|k| !k.is_empty());
    std::env::remove_var("OPENROUTER_API_KEY");
    key
}

fn run(req_path: &str, out_path: &str, openrouter_key: Option<String>) -> Result<(), String> {
    let text = std::fs::read_to_string(req_path).map_err(|e| format!("read request: {e}"))?;
    let container_req: ContainerReviewRequest =
        serde_json::from_str(&text).map_err(|e| format!("parse request: {e}"))?;
    let req = container_req.into_request(std::path::PathBuf::from("/work"));

    let backend: Box<dyn AgenticBackend> = match openrouter_key {
        Some(key) => {
            let base = std::env::var("OPENROUTER_BASE_URL")
                .unwrap_or_else(|_| OPENROUTER_API_BASE.to_owned());
            let model = std::env::var("OPENROUTER_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into());
            let agent = OpenRouterAgent::with_config(key, base, model, AgentConfig::default())
                .map_err(|e| format!("agent init: {e}"))?;
            Box::new(agent)
        }
        None => Box::new(SimAgent::new()),
    };

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| format!("runtime: {e}"))?;
    let verdict = rt
        .block_on(backend.review(&req))
        .map_err(|e| format!("review: {e}"))?;
    let json =
        serde_json::to_string_pretty(&verdict).map_err(|e| format!("encode verdict: {e}"))?;
    if let Some(parent) = std::path::Path::new(out_path).parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("out dir: {e}"))?;
    }
    std::fs::write(out_path, &json).map_err(|e| format!("write verdict: {e}"))?;
    println!("challenge-review verdict: {json}");
    Ok(())
}
