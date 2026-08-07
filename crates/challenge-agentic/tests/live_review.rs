//! Live `OpenRouter` smoke for the agentic reviewer (E12 regression probe).
//!
//! Manual only — run with:
//!
//! ```sh
//! OPENROUTER_API_KEY_FILE=/path/to/api_key \
//!   cargo test -p challenge-agentic --test live_review -- --ignored --nocapture
//! ```
//!
//! Reviews the shipped `hybrid_delta` v3 baseline (the submission whose
//! verdict once failed to parse on `missing_telemetry_hooks`) and asserts
//! the loop completes with SOME parsed verdict inside the raised budget.

#![forbid(unsafe_code)]
#![allow(clippy::expect_used, clippy::unwrap_used)]

use challenge_agentic::{
    AgenticBackend, AgenticVerdict, OpenRouterAgent, ReviewRequest, PRISM_DOMAIN_RULES,
};

#[tokio::test]
#[ignore = "live OpenRouter probe (spends tokens); run explicitly"]
async fn live_hybrid_delta_baseline_review_completes() {
    let key_path = std::env::var("OPENROUTER_API_KEY_FILE")
        .unwrap_or_else(|_| "/run/base/openrouter/api_key".into());
    let key = challenge_agentic::load_api_key_file(std::path::Path::new(&key_path))
        .expect("openrouter key file");

    let dir = tempfile::tempdir().unwrap();
    let base = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../prism-recipe/baselines/hybrid_delta"
    );
    for f in ["architecture.py", "training.py"] {
        let src = std::fs::read_to_string(format!("{base}/{f}")).expect("baseline source");
        std::fs::write(dir.path().join(f), src).unwrap();
    }

    let agent = OpenRouterAgent::new(key).expect("agent");
    let req = ReviewRequest {
        workdir: dir.path().to_path_buf(),
        primary_relpaths: vec!["architecture.py".into(), "training.py".into()],
        corpus: vec![],
        metrics_relpath: None,
        pages_relpath: None,
        sanitize_report_relpath: None,
        domain_rules: PRISM_DOMAIN_RULES.into(),
    };
    let verdict: AgenticVerdict = agent.review(&req).await.expect("live review");
    println!("live verdict: {verdict:?}");
    // The compliant baseline must not come back flagged for the hooks it has.
    assert!(
        !verdict
            .cheat_codes
            .contains(&challenge_agentic::CheatCode::MissingTelemetryHooks),
        "baseline has the telemetry hooks: {verdict:?}"
    );
}
