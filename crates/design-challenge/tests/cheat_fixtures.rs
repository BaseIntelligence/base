//! Malicious Design fixtures: harness byte-copy, scrape-style famous-site clone,
//! baseline clean eligibility (baseline-zeroing regression), pre-LLM copy gate
//! ordering, and rolling-window (`SCORING_VERSION = 3`) points-share scoring.
//!
//! Uses [`SimAgent`] heuristics (CI path). Live LLM review is optional elsewhere.

#![allow(clippy::unwrap_used)]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;

use challenge_agentic::{
    copy_gate, AgenticBackend, CheatCode, CorpusEntry, GateCorpusEntry, ReviewRequest, SimAgent,
    VerdictKind,
};
use design_challenge::score::{round_win_delta, score_window, ScorePlan, WindowScorePlan};
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

fn window_plan(wins: &[(&str, u32)], participants: &[&str], cheat: &[&str]) -> WindowScorePlan {
    WindowScorePlan {
        win_counts: wins.iter().map(|(hk, n)| ((*hk).to_owned(), *n)).collect(),
        miners_with_harness: participants.iter().map(|s| (*s).to_owned()).collect(),
        cheat_miners: cheat.iter().map(|s| (*s).to_owned()).collect(),
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

    // Cheat is terminal Score(0) and excluded from the window pool even with
    // wins on the board.
    let scores = score_window(&window_plan(
        &[("aa", 3), ("bb", 1)],
        &["aa", "bb"],
        &["aa"],
    ));
    assert_eq!(scores.get("aa"), Some(&FinalScore::Score(0)));
    assert_eq!(scores.get("bb"), Some(&FinalScore::Score(SCORE_MAX)));
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

    let scores = score_window(&window_plan(&[], &["cheat"], &["cheat"]));
    assert_eq!(scores.get("cheat"), Some(&FinalScore::Score(0)));
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
        env_vars: BTreeMap::new(),
    };
    validate_bundle(&bundle).unwrap();
    let out = SimSandbox::new()
        .execute(&bundle, 1, "run-clean", "brief", "http://proxy")
        .unwrap();
    let sanitized = sanitize_bundle(&out.pages).unwrap();
    assert_eq!(sanitized.pages.len(), 3);

    // scoring_version 3: a single window win already takes the full share
    // (no daily ≥2-wins threshold from v2).
    let hk = "ab".repeat(32);
    let scores = score_window(&window_plan(&[(&hk, 1)], &[&hk], &[]));
    assert_eq!(scores.get(&hk), Some(&FinalScore::Score(SCORE_MAX)));
}

#[tokio::test]
async fn baseline_zeroing_regression_reference_stays_clean() {
    // The published reference baseline must never be zeroed by anti-cheat:
    // (a) vs the public baseline corpus entry (miners start from it), and
    // (b) vs harnesses that merely share the baseline *shape*.
    let dir = tempdir().unwrap();
    let req = review_req(
        dir.path(),
        BASELINE_AGENT,
        vec![
            CorpusEntry {
                id: "baseline".into(),
                source: BASELINE_AGENT.into(),
            },
            CorpusEntry {
                id: "harness:unrelated".into(),
                source: "import math\ndef train(xs):\n    return sum(math.sin(x) for x in xs)\n"
                    .into(),
            },
        ],
        Some(&[(
            "index.html",
            r#"<!DOCTYPE html><html data-agent="design-baseline"><body><h1>ok</h1></body></html>"#,
        )]),
    );
    let v = SimAgent::new().review(&req).await.unwrap();
    assert_eq!(
        v.verdict,
        VerdictKind::Clean,
        "reference baseline must come out clean (baseline-zeroing fix), got {v:?}"
    );
}

#[tokio::test]
async fn baseline_light_variation_stays_clean() {
    // A genuine miner edit of the baseline (renames + custom copy) must not be
    // zeroed as suspicious when compared against an earlier *different* harness.
    let dir = tempdir().unwrap();
    let variant = format!("# miner customization: branded theme\n{BASELINE_AGENT}")
        .replace("task", "brief")
        .replace("llm", "client");
    let earlier = r"
import json, os

def run(task, llm, out):
    pages = {}
    for name in ['index.html', 'pricing.html', 'components.html']:
        pages[name] = '<html><body><h1>totally different</h1></body></html>'
    return pages
";
    let req = review_req(
        dir.path(),
        &variant,
        vec![CorpusEntry {
            id: "harness:earlier".into(),
            source: earlier.into(),
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
        "baseline variation must stay clean (no threshold zeroing), got {v:?}"
    );
}

#[test]
fn copy_gate_created_at_ordering() {
    let victim = BASELINE_AGENT;
    // Newer candidate byte-copying an earlier harness → gate fires.
    let corpus = vec![GateCorpusEntry {
        id: "harness:victim".into(),
        source: victim.into(),
        created_at_ms: 1_000,
    }];
    let hit = copy_gate(victim, 2_000, &corpus).unwrap();
    assert_eq!(hit.nearest_id, "harness:victim");
    assert!(hit.byte_identical);
    // The original (older) harness is never rejected by the gate.
    assert!(copy_gate(victim, 1_000, &corpus).is_none());
    // Public baseline entries are reference material, not copy victims.
    let baseline_corpus = vec![GateCorpusEntry {
        id: "baseline".into(),
        source: victim.into(),
        created_at_ms: 1_000,
    }];
    assert!(copy_gate(victim, 2_000, &baseline_corpus).is_none());
}

#[test]
fn window_share_and_cheat_ineligible() {
    let clean_a = "aa".repeat(32);
    let clean_b = "bb".repeat(32);
    let cheat_c = "cc".repeat(32);

    // 2-vs-1 window points → 2/3 and 1/3 shares; cheat zeroed + excluded.
    let scores = score_window(&window_plan(
        &[(&clean_a, 2), (&clean_b, 1), (&cheat_c, 5)],
        &[&clean_a, &clean_b, &cheat_c],
        &[&cheat_c],
    ));
    assert_eq!(
        scores.get(&clean_a),
        Some(&FinalScore::Score(SCORE_MAX / 3 * 2))
    );
    assert_eq!(
        scores.get(&clean_b),
        Some(&FinalScore::Score(SCORE_MAX / 3))
    );
    assert_eq!(scores.get(&cheat_c), Some(&FinalScore::Score(0)));

    // Cheat must never collect points even if wrongly nominated as winner.
    let bad = ScorePlan {
        miners_with_harness: vec![cheat_c.clone()],
        miners_clean: vec![],
        winner_miners: vec![cheat_c.clone()],
        cheat_miners: vec![cheat_c.clone()],
    };
    assert!(!round_win_delta(&bad).contains_key(&cheat_c));
}

#[test]
fn host_sim_forbidden_without_base_allow_host_sim() {
    assert!(!host_sim_allowed(541, false, Some("staging")));
    let err = require_host_sim_for_force(true, 541, false, Some("staging")).unwrap_err();
    assert!(err.contains("BASE_ALLOW_HOST_SIM"));
    assert!(require_host_sim_for_force(true, 100, true, None).is_err());
}

#[test]
fn window_plan_types_compile() {
    // Guard the public plan surface used by the orchestrator.
    let plan = WindowScorePlan {
        win_counts: BTreeMap::new(),
        miners_with_harness: BTreeSet::new(),
        cheat_miners: BTreeSet::new(),
    };
    assert!(score_window(&plan).is_empty());
}
