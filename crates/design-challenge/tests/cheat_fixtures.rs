//! Malicious Design fixtures: harness byte-copy, scrape-style famous-site clone,
//! baseline clean eligibility, and admin 2-winner `SCORE_MAX/2` scoring.
//!
//! Uses [`SimAgent`] heuristics (CI path). Live LLM review is optional elsewhere.

#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::fs;

use challenge_agentic::{
    AgenticBackend, CheatCode, CorpusEntry, ReviewRequest, SimAgent, VerdictKind,
};
use design_challenge::score::{score_round, ScorePlan};
use design_challenge::{host_sim_allowed, require_host_sim_for_force, SCORE_MAX};
use design_harness::{validate_bundle, HarnessBundle};
use design_sandbox::{SandboxBackend, SimSandbox};
use design_sanitize::sanitize_bundle;
use design_store::FinalScore;
use tempfile::tempdir;

const BASELINE_AGENT: &str =
    include_str!("../../../docs/external-miner/examples/design-baseline/agent.py");
const BASELINE_PYPROJECT: &str =
    include_str!("../../../docs/external-miner/examples/design-baseline/pyproject.toml");

const TWITTER_CLONE_HTML: &str = r#"<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta property="og:site_name" content="Twitter">
  <title>Twitter</title>
</head>
<body>
  <div data-testid="tweet">Hello from twitter.com</div>
</body>
</html>"#;

fn review_req(
    work: &std::path::Path,
    agent_py: &str,
    corpus: Vec<CorpusEntry>,
    pages: Option<&[(&str, &str)]>,
) -> ReviewRequest {
    fs::write(work.join("agent.py"), agent_py).unwrap();
    let pages_relpath = if let Some(pages) = pages {
        let dir = work.join("pages");
        fs::create_dir_all(&dir).unwrap();
        for (name, body) in pages {
            fs::write(dir.join(name), body).unwrap();
        }
        Some("pages".into())
    } else {
        None
    };
    ReviewRequest {
        workdir: work.to_path_buf(),
        primary_relpaths: vec!["agent.py".into()],
        corpus,
        metrics_relpath: None,
        pages_relpath,
        sanitize_report_relpath: None,
        domain_rules: "design".into(),
    }
}

#[tokio::test]
async fn byte_level_harness_copy_is_cheat_score_zero() {
    let dir = tempdir().unwrap();
    let victim = BASELINE_AGENT;
    let req = review_req(
        dir.path(),
        victim,
        vec![CorpusEntry {
            id: "harness:victim".into(),
            source: victim.into(),
        }],
        Some(&[("index.html", "<html data-agent=\"design-baseline\"></html>")]),
    );
    let v = SimAgent::new().review(&req).await.unwrap();
    assert_eq!(v.verdict, VerdictKind::Cheat);
    assert!(v.cheat_codes.contains(&CheatCode::NearIdenticalHarnessCopy));
    assert_eq!(v.similarity_bps, 10_000);

    let plan = ScorePlan {
        miners_with_harness: vec!["aa".into(), "bb".into()],
        miners_clean: vec!["bb".into()],
        winner_miners: vec!["bb".into()],
        cheat_miners: vec!["aa".into()],
    };
    let scores = score_round(&plan);
    assert_eq!(scores.get("aa"), Some(&FinalScore::Score(0)));
    assert!(!matches!(
        scores.get("aa"),
        Some(FinalScore::Score(s)) if *s > 0
    ));
}

#[tokio::test]
async fn scrape_style_famous_site_clone_is_cheat() {
    let dir = tempdir().unwrap();
    let agent = r#"
import requests

def run(ctx):
    # scrape-style republish of a famous site
    html = requests.get("https://twitter.com").text
    ctx.write_page("index.html", html)
"#;
    let req = review_req(
        dir.path(),
        agent,
        vec![],
        Some(&[
            ("index.html", TWITTER_CLONE_HTML),
            ("pricing.html", TWITTER_CLONE_HTML),
            ("components.html", TWITTER_CLONE_HTML),
        ]),
    );
    let v = SimAgent::new().review(&req).await.unwrap();
    assert_eq!(v.verdict, VerdictKind::Cheat);
    assert!(v.cheat_codes.contains(&CheatCode::ScrapedSiteClone));

    let plan = ScorePlan {
        miners_with_harness: vec!["cheat".into()],
        miners_clean: vec![],
        winner_miners: vec![],
        cheat_miners: vec!["cheat".into()],
    };
    assert_eq!(score_round(&plan).get("cheat"), Some(&FinalScore::Score(0)));
}

#[tokio::test]
async fn baseline_clean_is_admin_eligible() {
    let dir = tempdir().unwrap();
    let unrelated = r"
import math
def train(xs):
    return sum(math.sin(x) for x in xs)
";
    let req = review_req(
        dir.path(),
        BASELINE_AGENT,
        vec![CorpusEntry {
            id: "other".into(),
            source: unrelated.into(),
        }],
        Some(&[(
            "index.html",
            r#"<!DOCTYPE html><html data-agent="design-baseline"><body><h1>ok</h1></body></html>"#,
        )]),
    );
    let v = SimAgent::new().review(&req).await.unwrap();
    assert_eq!(
        v.verdict,
        VerdictKind::Clean,
        "baseline must be clean/eligible, got {v:?}"
    );

    // Sim sandbox still produces the three required pages for admin review.
    let bundle = HarnessBundle {
        miner_hotkey: "ab".repeat(32),
        agent_py: BASELINE_AGENT.into(),
        pyproject_toml: BASELINE_PYPROJECT.into(),
        extra_files: BTreeMap::new(),
    };
    validate_bundle(&bundle).unwrap();
    let out = SimSandbox::new()
        .execute(&bundle, 1, "run-clean", "brief", "http://proxy")
        .unwrap();
    let sanitized = sanitize_bundle(&out.pages).unwrap();
    assert_eq!(sanitized.pages.len(), 3);

    let plan = ScorePlan {
        miners_with_harness: vec!["ab".repeat(32)],
        miners_clean: vec!["ab".repeat(32)],
        winner_miners: vec!["ab".repeat(32)],
        cheat_miners: vec![],
    };
    assert_eq!(
        score_round(&plan).get(&"ab".repeat(32)),
        Some(&FinalScore::Score(SCORE_MAX))
    );
}

#[test]
fn admin_two_winners_half_score_and_cheat_ineligible() {
    let clean_a = "aa".repeat(32);
    let clean_b = "bb".repeat(32);
    let cheat_c = "cc".repeat(32);

    // Mirror orchestrator award scoring (two admin winners + one cheat).
    let plan = ScorePlan {
        miners_with_harness: vec![clean_a.clone(), clean_b.clone(), cheat_c.clone()],
        miners_clean: vec![clean_a.clone(), clean_b.clone()],
        winner_miners: vec![clean_a.clone(), clean_b.clone()],
        cheat_miners: vec![cheat_c.clone()],
    };
    let scores = score_round(&plan);
    assert_eq!(
        scores.get(&clean_a),
        Some(&FinalScore::Score(SCORE_MAX / 2))
    );
    assert_eq!(
        scores.get(&clean_b),
        Some(&FinalScore::Score(SCORE_MAX / 2))
    );
    assert_eq!(scores.get(&cheat_c), Some(&FinalScore::Score(0)));

    // Cheat must never be admin-eligible even if wrongly nominated.
    let bad = ScorePlan {
        miners_with_harness: vec![cheat_c.clone()],
        miners_clean: vec![],
        winner_miners: vec![cheat_c.clone()],
        cheat_miners: vec![cheat_c.clone()],
    };
    assert_eq!(score_round(&bad).get(&cheat_c), Some(&FinalScore::Score(0)));
}

#[test]
fn host_sim_forbidden_without_base_allow_host_sim() {
    assert!(!host_sim_allowed(541, false, Some("staging")));
    let err = require_host_sim_for_force(true, 541, false, Some("staging")).unwrap_err();
    assert!(err.contains("BASE_ALLOW_HOST_SIM"));
    assert!(require_host_sim_for_force(true, 100, true, None).is_err());
}
