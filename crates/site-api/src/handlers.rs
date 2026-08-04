//! Axum handlers for `GET /v1/site/*`.

use std::collections::HashMap;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::frames::coding_arena;
use crate::map::{
    activity_from_lives, design_arena_from_dashboard, design_leaderboard, design_submission,
    list_arenas, prism_arena_from_live, prism_submission, prism_window,
};
use crate::paginate::page_slice;
use crate::state::SiteState;
use crate::types::{
    ArenaSlug, Governance, LandingSummary, MetricsEmission, MetricsPassRate, MetricsPopulation,
    NetworkMetrics, NetworkStats, ResultsMatrix, Validator,
};
use crate::upstream::{self, DESIGN, PRISM};

/// Mount marketing site routes under `/v1/site`.
pub fn site_router(state: SiteState) -> Router {
    Router::new()
        .route("/v1/site/network", get(get_network))
        .route("/v1/site/landing", get(get_landing))
        .route("/v1/site/arenas", get(get_arenas))
        .route("/v1/site/arenas/{slug}", get(get_arena))
        .route("/v1/site/arenas/{slug}/leaderboard", get(get_leaderboard))
        .route("/v1/site/arenas/{slug}/submissions", get(get_submissions))
        .route("/v1/site/arenas/design/duels", get(get_duels))
        .route(
            "/v1/site/arenas/coding/results-matrix",
            get(get_results_matrix),
        )
        .route("/v1/site/arenas/prism/window", get(get_prism_window))
        .route("/v1/site/validators", get(get_validators))
        .route("/v1/site/activity", get(get_activity))
        .route("/v1/site/metrics", get(get_metrics))
        .route("/v1/site/governance", get(get_governance))
        .with_state(state)
}

#[derive(Debug, Deserialize)]
struct PageQuery {
    page: Option<u32>,
    #[serde(rename = "pageSize")]
    page_size: Option<u32>,
    status: Option<String>,
}

#[derive(Debug, Deserialize)]
struct LimitQuery {
    limit: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct MetricsQuery {
    range: Option<String>,
}

fn json_err(code: StatusCode, kind: &str, msg: &str) -> Response {
    (code, Json(json!({"error": kind, "message": msg}))).into_response()
}

fn now_iso() -> String {
    let ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(0));
    crate::map::ms_to_iso(ms)
}

async fn fetch_design_dash(st: &SiteState) -> Option<Value> {
    upstream::get_json_opt(st, DESIGN, "/v1/dashboard").await
}

async fn fetch_prism_status(st: &SiteState) -> Option<Value> {
    upstream::get_json_opt(st, PRISM, "/v1/status").await
}

async fn fetch_prism_subs(st: &SiteState, limit: u32) -> Option<Value> {
    upstream::get_json_opt(st, PRISM, &format!("/v1/submissions?limit={limit}")).await
}

async fn fetch_prism_recipe(st: &SiteState) -> Option<Value> {
    upstream::get_json_opt(st, PRISM, "/v1/recipe").await
}

fn epoch_from_lives(design: Option<&Value>, prism: Option<&Value>, chain_epoch: u64) -> u64 {
    design
        .and_then(|d| d.get("epoch"))
        .and_then(Value::as_u64)
        .or_else(|| prism.and_then(|p| p.get("epoch")).and_then(Value::as_u64))
        .unwrap_or(chain_epoch)
}

fn chain_snapshot(st: &SiteState) -> (u64, u64, Vec<Validator>) {
    let Some(chain) = st.chain.as_ref() else {
        return (0, 0, Vec::new());
    };
    let block = chain.current_block().unwrap_or(0);
    let epoch = chain.subnet_epoch_index(st.netuid).unwrap_or(0);
    let validators = match chain.block_hash(block).and_then(|h| chain.metagraph_at(&h)) {
        Ok(mg) => mg
            .hotkeys
            .iter()
            .enumerate()
            .map(|(i, hk)| Validator {
                uid: u16::try_from(i).unwrap_or(u16::MAX),
                name: format!("uid-{i}"),
                hotkey: hex_hotkey(hk),
                stake: 0.0,
                trust: 0.0,
                vtrust: 0.0,
                version: "—".into(),
                updated_blocks_ago: 0,
            })
            .collect(),
        Err(_) => Vec::new(),
    };
    (block, epoch, validators)
}

fn hex_hotkey(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0xf) as usize] as char);
    }
    s
}

async fn network_stats(st: &SiteState) -> NetworkStats {
    let (block, chain_epoch, validators) = chain_snapshot(st);
    let design = fetch_design_dash(st).await;
    let prism = fetch_prism_status(st).await;
    let prism_subs = fetch_prism_subs(st, 200).await;
    let arenas = list_arenas(design.as_ref(), prism.as_ref(), prism_subs.as_ref());
    let agents: u32 = arenas.iter().map(|a| a.agents).sum();
    NetworkStats {
        epoch: epoch_from_lives(design.as_ref(), prism.as_ref(), chain_epoch),
        agents,
        validators: u32::try_from(validators.len()).unwrap_or(0),
        arenas: 3,
        emission_per_day: 0.0,
        tao_price: 0.0,
        block_height: block,
        updated_at: now_iso(),
        total_stake: None,
    }
}

async fn get_network(State(st): State<SiteState>) -> impl IntoResponse {
    Json(network_stats(&st).await)
}

async fn get_landing(State(st): State<SiteState>) -> impl IntoResponse {
    let design = fetch_design_dash(&st).await;
    let prism = fetch_prism_status(&st).await;
    let prism_subs = fetch_prism_subs(&st, 200).await;
    let stats = network_stats(&st).await;
    let arenas = list_arenas(design.as_ref(), prism.as_ref(), prism_subs.as_ref());
    let design_runs = design
        .as_ref()
        .and_then(|d| d.get("recent_runs"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let prism_rows = prism_subs
        .as_ref()
        .and_then(|d| d.get("submissions"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let activity = activity_from_lives(&design_runs, &prism_rows, 10);
    Json(LandingSummary {
        stats,
        arenas,
        activity,
    })
}

async fn get_arenas(State(st): State<SiteState>) -> impl IntoResponse {
    let design = fetch_design_dash(&st).await;
    let prism = fetch_prism_status(&st).await;
    let prism_subs = fetch_prism_subs(&st, 200).await;
    Json(list_arenas(
        design.as_ref(),
        prism.as_ref(),
        prism_subs.as_ref(),
    ))
}

async fn get_arena(State(st): State<SiteState>, Path(slug): Path<String>) -> Response {
    let Some(slug) = ArenaSlug::parse(&slug) else {
        return json_err(StatusCode::NOT_FOUND, "not_found", "unknown arena");
    };
    let arena = match slug {
        ArenaSlug::Coding => coding_arena(),
        ArenaSlug::Design => design_arena_from_dashboard(fetch_design_dash(&st).await.as_ref()),
        ArenaSlug::Prism => {
            let status = fetch_prism_status(&st).await;
            let subs = fetch_prism_subs(&st, 200).await;
            prism_arena_from_live(status.as_ref(), subs.as_ref())
        }
    };
    Json(arena).into_response()
}

async fn get_leaderboard(
    State(st): State<SiteState>,
    Path(slug): Path<String>,
    Query(q): Query<PageQuery>,
) -> Response {
    let Some(slug) = ArenaSlug::parse(&slug) else {
        return json_err(StatusCode::NOT_FOUND, "not_found", "unknown arena");
    };
    let page = q.page.unwrap_or(1);
    let page_size = q.page_size.unwrap_or(24);
    match slug {
        ArenaSlug::Coding => {
            let empty = page_slice::<crate::types::LeaderboardRow>(&[], page, page_size);
            Json(json!({
                "items": empty.items,
                "page": empty.page,
                "pageSize": empty.page_size,
                "total": empty.total,
                "pageCount": empty.page_count,
                "epoch": 0,
                "updatedAt": now_iso(),
            }))
            .into_response()
        }
        ArenaSlug::Design => {
            let dash = fetch_design_dash(&st).await;
            let round_id = dash
                .as_ref()
                .and_then(|d| d.pointer("/leaderboard/current_round"))
                .and_then(Value::as_u64)
                .or_else(|| {
                    dash.as_ref()
                        .and_then(|d| d.pointer("/round/round_id"))
                        .and_then(Value::as_u64)
                });
            let lb = if let Some(rid) = round_id {
                upstream::get_json_opt(&st, DESIGN, &format!("/v1/rounds/{rid}/leaderboard")).await
            } else {
                None
            };
            let ratings = lb
                .as_ref()
                .and_then(|v| v.get("ratings"))
                .and_then(Value::as_array)
                .cloned()
                .or_else(|| {
                    dash.as_ref()
                        .and_then(|d| d.pointer("/leaderboard/ratings"))
                        .and_then(Value::as_array)
                        .cloned()
                })
                .unwrap_or_default();
            let previous = dash
                .as_ref()
                .and_then(|d| d.pointer("/leaderboard/previous_ratings"))
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let epoch = dash
                .as_ref()
                .and_then(|d| d.get("epoch"))
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let rows = design_leaderboard(&ratings, &previous, epoch);
            let page_out = page_slice(&rows, page, page_size);
            Json(json!({
                "items": page_out.items,
                "page": page_out.page,
                "pageSize": page_out.page_size,
                "total": page_out.total,
                "pageCount": page_out.page_count,
                "epoch": epoch,
                "updatedAt": now_iso(),
            }))
            .into_response()
        }
        ArenaSlug::Prism => {
            // Prism has no Elo board — empty honest list (series live on /window).
            let empty = page_slice::<crate::types::LeaderboardRow>(&[], page, page_size);
            let epoch = fetch_prism_status(&st)
                .await
                .and_then(|s| s.get("epoch").and_then(Value::as_u64))
                .unwrap_or(0);
            Json(json!({
                "items": empty.items,
                "page": empty.page,
                "pageSize": empty.page_size,
                "total": empty.total,
                "pageCount": empty.page_count,
                "epoch": epoch,
                "updatedAt": now_iso(),
            }))
            .into_response()
        }
    }
}

async fn get_submissions(
    State(st): State<SiteState>,
    Path(slug): Path<String>,
    Query(q): Query<PageQuery>,
) -> Response {
    let Some(slug) = ArenaSlug::parse(&slug) else {
        return json_err(StatusCode::NOT_FOUND, "not_found", "unknown arena");
    };
    let page = q.page.unwrap_or(1);
    let page_size = q.page_size.unwrap_or(24);
    let status_filter = q.status.as_deref();
    match slug {
        ArenaSlug::Coding => {
            Json(page_slice::<crate::types::Submission>(&[], page, page_size)).into_response()
        }
        ArenaSlug::Design => design_submissions_page(&st, page, page_size, status_filter).await,
        ArenaSlug::Prism => {
            let raw = fetch_prism_subs(&st, 500).await;
            let rows = raw
                .as_ref()
                .and_then(|v| v.get("submissions"))
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let mut items: Vec<_> = rows.iter().filter_map(prism_submission).collect();
            if let Some(st_f) = status_filter {
                items.retain(|s| match st_f {
                    "scored" => s.status == crate::types::SubmissionStatus::Scored,
                    "pending" => s.status == crate::types::SubmissionStatus::Pending,
                    "failed" => s.status == crate::types::SubmissionStatus::Failed,
                    _ => true,
                });
            }
            Json(page_slice(&items, page, page_size)).into_response()
        }
    }
}

async fn design_submissions_page(
    st: &SiteState,
    page: u32,
    page_size: u32,
    status_filter: Option<&str>,
) -> Response {
    let dash = fetch_design_dash(st).await;
    let epoch = dash
        .as_ref()
        .and_then(|d| d.get("epoch"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let runs = dash
        .as_ref()
        .and_then(|d| d.get("recent_runs"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    // Resolve miner hotkeys via harness lookup (cached).
    let mut harness_cache: HashMap<String, String> = HashMap::new();
    let mut items = Vec::new();
    for run in &runs {
        let hid = run
            .get("harness_id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        let miner = if hid.is_empty() {
            "—".into()
        } else if let Some(m) = harness_cache.get(&hid) {
            m.clone()
        } else {
            let h = upstream::get_json_opt(st, DESIGN, &format!("/v1/harness/{hid}")).await;
            let m = h
                .and_then(|v| {
                    v.get("miner_hotkey")
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                })
                .unwrap_or_else(|| "—".into());
            harness_cache.insert(hid, m.clone());
            m
        };
        let run_id = run.get("id").and_then(Value::as_str).unwrap_or("");
        let detail = if run.get("status").and_then(Value::as_str) == Some("scored")
            || run.get("status").and_then(Value::as_str) == Some("failed")
        {
            upstream::get_json_opt(st, DESIGN, &format!("/v1/runs/{run_id}")).await
        } else {
            None
        };
        if let Some(sub) = design_submission(run, &miner, detail.as_ref(), epoch) {
            items.push(sub);
        }
    }
    if let Some(st_f) = status_filter {
        items.retain(|s| match st_f {
            "scored" => s.status == crate::types::SubmissionStatus::Scored,
            "pending" => s.status == crate::types::SubmissionStatus::Pending,
            "failed" => s.status == crate::types::SubmissionStatus::Failed,
            _ => true,
        });
    }
    Json(page_slice(&items, page, page_size)).into_response()
}

/// Design duels: always empty — product model is admin winners, not fabricated matchups.
async fn get_duels(Query(q): Query<LimitQuery>) -> impl IntoResponse {
    let _ = q.limit;
    Json(Vec::<Value>::new())
}

async fn get_results_matrix() -> impl IntoResponse {
    Json(ResultsMatrix {
        arena: ArenaSlug::Coding,
        tasks: Vec::new(),
        rows: Vec::new(),
    })
}

async fn get_prism_window(State(st): State<SiteState>) -> impl IntoResponse {
    let recipe = fetch_prism_recipe(&st).await;
    let status = fetch_prism_status(&st).await;
    let subs = fetch_prism_subs(&st, 200).await;
    let rows = subs
        .as_ref()
        .and_then(|v| v.get("submissions"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    Json(prism_window(recipe.as_ref(), status.as_ref(), &rows))
}

async fn get_validators(
    State(st): State<SiteState>,
    Query(q): Query<PageQuery>,
) -> impl IntoResponse {
    let (_, _, validators) = chain_snapshot(&st);
    let page = page_slice(&validators, q.page.unwrap_or(1), q.page_size.unwrap_or(24));
    Json(page)
}

async fn get_activity(
    State(st): State<SiteState>,
    Query(q): Query<LimitQuery>,
) -> impl IntoResponse {
    let limit = q.limit.unwrap_or(10).min(100) as usize;
    let design = fetch_design_dash(&st).await;
    let prism_subs = fetch_prism_subs(&st, 200).await;
    let design_runs = design
        .as_ref()
        .and_then(|d| d.get("recent_runs"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let prism_rows = prism_subs
        .as_ref()
        .and_then(|d| d.get("submissions"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    Json(activity_from_lives(&design_runs, &prism_rows, limit))
}

async fn get_metrics(
    State(st): State<SiteState>,
    Query(q): Query<MetricsQuery>,
) -> impl IntoResponse {
    let range = q.range.unwrap_or_else(|| "30".into());
    let stats = network_stats(&st).await;
    Json(NetworkMetrics {
        range,
        epoch: stats.epoch,
        kpis: Vec::new(),
        emission: MetricsEmission {
            points: Vec::new(),
            shares: Vec::new(),
            total_this_epoch: 0.0,
        },
        pass_rate: MetricsPassRate {
            points: Vec::new(),
            latest: Vec::new(),
        },
        population: MetricsPopulation {
            rows: Vec::new(),
            new_this_epoch: 0,
        },
        ledger: Vec::new(),
    })
}

async fn get_governance(State(st): State<SiteState>) -> impl IntoResponse {
    let stats = network_stats(&st).await;
    Json(Governance {
        epoch: stats.epoch,
        open_for_voting: 0,
        next_close_in: "—".into(),
        stages: Vec::new(),
        proposals: Vec::new(),
        rules: Vec::new(),
        decisions: Vec::new(),
        decisions_sealed: 0,
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use gateway_registry::{CreateBackend, Registry, RegistryConfig};
    use http_body_util::BodyExt;
    use serde_json::{json, Value};
    use tower::ServiceExt;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    async fn setup() -> (MockServer, MockServer, SiteState) {
        let design = MockServer::start().await;
        let prism = MockServer::start().await;
        let registry = Registry::shared(RegistryConfig::default());
        registry
            .create(&CreateBackend {
                challenge_id: "design".into(),
                base_url: design.uri(),
                weight: 1,
            })
            .unwrap();
        registry
            .create(&CreateBackend {
                challenge_id: "prism".into(),
                base_url: prism.uri(),
                weight: 1,
            })
            .unwrap();
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap();
        let st = SiteState::new(registry, client, None, 541);
        (design, prism, st)
    }

    async fn call(app: axum::Router, path: &str) -> (StatusCode, Value) {
        let res = app
            .oneshot(Request::get(path).body(Body::empty()).unwrap())
            .await
            .unwrap();
        let status = res.status();
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        let v: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        (status, v)
    }

    #[tokio::test]
    async fn arenas_and_design_submissions_from_mocks() {
        let (design, prism, st) = setup().await;
        Mock::given(method("GET"))
            .and(path("/v1/dashboard"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "epoch": 3,
                "round": {"round_id": 9},
                "leaderboard": {
                    "current_round": 9,
                    "ratings": [{"miner_hotkey":"aa","rating":1300,"wins":1,"losses":0}],
                    "previous_ratings": []
                },
                "recent_runs": [{
                    "id": "run1",
                    "status": "scored",
                    "round_id": 9,
                    "harness_id": "h1",
                    "prompt_id": "p1",
                    "error_detail": null,
                    "updated_at_ms": 1_700_000_000_000_u64
                }]
            })))
            .mount(&design)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/harness/h1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "h1",
                "miner_hotkey": "aa".repeat(32)
            })))
            .mount(&design)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/runs/run1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "run1",
                "status": "scored",
                "final_score": {"score": 1000}
            })))
            .mount(&design)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/status"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "epoch": 3,
                "backend": "sim"
            })))
            .mount(&prism)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/submissions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "submissions": [{
                    "id": "sub1",
                    "miner_hotkey": "bb".repeat(32),
                    "epoch": 3,
                    "status": "terminated",
                    "label": "base",
                    "bpb": 1.25,
                    "score": {"kind":"score","value": 900},
                    "created_at_ms": 1_700_000_000_000_u64,
                    "updated_at_ms": 1_700_000_000_000_u64
                }]
            })))
            .mount(&prism)
            .await;

        let app = site_router(st);
        let (s, v) = call(app.clone(), "/v1/site/arenas").await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v.as_array().unwrap().len(), 3);
        assert_eq!(v[0]["slug"], "coding");
        assert_eq!(v[0]["status"], "paused");
        assert_eq!(v[1]["slug"], "design");
        assert_eq!(v[1]["bestScore"], "1,300");

        let (s, v) = call(app.clone(), "/v1/site/arenas/design/submissions").await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["items"].as_array().unwrap().len(), 1);
        assert_eq!(
            v["items"][0]["url"],
            "/challenge/design/v1/view/run1/index.html"
        );
        assert_eq!(v["items"][0]["status"], "scored");
        assert_eq!(v["items"][0]["score"], 1000.0);

        let (s, v) = call(app.clone(), "/v1/site/arenas/design/duels").await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v.as_array().unwrap().len(), 0);

        let (s, v) = call(app.clone(), "/v1/site/arenas/prism/submissions").await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["items"][0]["arena"], "prism");
        assert_eq!(v["items"][0]["status"], "scored");

        let (s, v) = call(app, "/v1/site/arenas/coding/submissions").await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["total"], 0);
    }

    #[tokio::test]
    async fn prism_window_uses_recipe_and_bpb() {
        let (_design, prism, st) = setup().await;
        Mock::given(method("GET"))
            .and(path("/v1/recipe"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "version": "1.0.1",
                "dataset_ref": "ds@pin",
                "pin_hex": "abc"
            })))
            .mount(&prism)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/status"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "backend": "sim", "epoch": 1
            })))
            .mount(&prism)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/submissions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "submissions": [
                    {"id":"x1","status":"terminated","bpb":2.0,"label":"a"},
                    {"id":"x2","status":"terminated","bpb":1.0,"label":"b"}
                ]
            })))
            .mount(&prism)
            .await;
        let app = site_router(st);
        let (s, v) = call(app, "/v1/site/arenas/prism/window").await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["dataset"], "ds@pin");
        assert_eq!(v["series"].as_array().unwrap().len(), 2);
        assert_eq!(v["series"][0]["finalLoss"], 1.0);
        assert_eq!(v["series"][0]["points"].as_array().unwrap().len(), 1);
    }
}
