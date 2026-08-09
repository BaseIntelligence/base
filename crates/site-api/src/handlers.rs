//! Axum handlers for `GET /v1/site/*`.

use std::collections::HashMap;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::state::SiteState;
use crate::upstream::{self, DESIGN, PRISM};
use site_data::map::{
    activity_from_lives, design_arena_from_dashboard, design_leaderboard, design_submission,
    enrich_leaderboard_uids, enrich_leaderboard_weights, enrich_submission_uids,
    is_prism_champion_submission, leaderboard_matches_query, list_arenas, prism_arena_from_live,
    prism_bpb_leaderboard, prism_submission, prism_telemetry, prism_window,
    submission_matches_query, uid_index_from_hotkeys,
};
use site_types::coding_arena;
use site_types::page_slice;
use site_types::{
    ArenaSlug, Governance, LandingSummary, MetricsEmission, MetricsPassRate, MetricsPopulation,
    NetworkMetrics, NetworkStats, ResultsMatrix, Validator,
};

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
        .route(
            "/v1/site/arenas/prism/submissions/{id}/telemetry",
            get(get_prism_submission_telemetry),
        )
        .route("/v1/site/validators", get(get_validators))
        .route("/v1/site/weights", get(get_site_weights))
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
    /// Hotkey / handle / prompt substring filter (SS58 or hex).
    q: Option<String>,
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
    site_data::map::ms_to_iso(ms)
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
    // Challenge payloads report 0 when their own chain read never hydrated —
    // treat 0 as "unknown" so the chain-derived epoch wins.
    design
        .and_then(|d| d.get("epoch"))
        .and_then(Value::as_u64)
        .filter(|e| *e > 0)
        .or_else(|| {
            prism
                .and_then(|p| p.get("epoch"))
                .and_then(Value::as_u64)
                .filter(|e| *e > 0)
        })
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

/// Current metagraph hotkey → UID index (hex + SS58). Empty when chain is down.
fn metagraph_uid_index(st: &SiteState) -> HashMap<String, u16> {
    let Some(chain) = st.chain.as_ref() else {
        return HashMap::new();
    };
    let block = chain.current_block().unwrap_or(0);
    let Ok(hash) = chain.block_hash(block) else {
        return HashMap::new();
    };
    let Ok(mg) = chain.metagraph_at(&hash) else {
        return HashMap::new();
    };
    uid_index_from_hotkeys(&mg.hotkeys)
}

/// Sealed hotkey → normalised weight share (SS58 keys from the bundle).
fn sealed_weight_index(st: &SiteState) -> HashMap<String, f64> {
    site_data::weights::sealed_view(st.latest_sealed_bundle())
        .map(|view| view.hotkeys.into_iter().collect())
        .unwrap_or_default()
}

/// Enrich leaderboard rows with on-chain UID + sealed weight / TAO/day.
fn decorate_leaderboard(st: &SiteState, rows: &mut [crate::LeaderboardRow]) {
    let uids = metagraph_uid_index(st);
    if !uids.is_empty() {
        enrich_leaderboard_uids(rows, &uids);
    }
    let weights = sealed_weight_index(st);
    if !weights.is_empty() {
        // `emission_per_day` is not yet hydrated from chain; pass 0 so we still
        // surface `weight` and leave `taoPerDay` unset until the rate is known.
        // Clients may also compute `weight × emissionPerDay` from /v1/weights/latest
        // once network stats publish a non-zero emission.
        enrich_leaderboard_weights(rows, &weights, 0.0);
    }
}

/// Enrich submission agents with on-chain UID.
fn decorate_submissions(st: &SiteState, rows: &mut [crate::Submission]) {
    let uids = metagraph_uid_index(st);
    if !uids.is_empty() {
        enrich_submission_uids(rows, &uids);
    }
}

async fn network_stats(st: &SiteState) -> NetworkStats {
    let (block, chain_epoch, validators) = chain_snapshot(st);
    let design = fetch_design_dash(st).await;
    let prism = fetch_prism_status(st).await;
    let prism_subs = fetch_prism_subs(st, 200).await;
    let arenas = arenas_with_emission(st, design.as_ref(), prism.as_ref(), prism_subs.as_ref());
    let agents: u32 = arenas.iter().map(|a| a.agents).sum();
    let tao_price = site_data::price::tao_price_usd(&st.client, &st.tao_price).await;
    NetworkStats {
        epoch: epoch_from_lives(design.as_ref(), prism.as_ref(), chain_epoch),
        agents,
        validators: u32::try_from(validators.len()).unwrap_or(0),
        arenas: 3,
        emission_per_day: 0.0,
        tao_price,
        block_height: block,
        updated_at: now_iso(),
        total_stake: None,
    }
}

/// Apply trust-root emission share + sealed-vector weight to one arena.
fn apply_emission(st: &SiteState, arena: &mut site_types::Arena) {
    let slug = arena.slug.as_str();
    let shares = site_data::weights::configured_shares(st.trust_root());
    if let Some((_, share)) = shares.iter().find(|(s, _)| s == slug) {
        arena.emission_share = *share;
    }
    arena.weight =
        site_data::weights::arena_weight(st.trust_root(), st.latest_sealed_bundle(), slug);
}

/// Arena list with trust-root emission shares + sealed-vector weights applied.
fn arenas_with_emission(
    st: &SiteState,
    design: Option<&Value>,
    prism: Option<&Value>,
    prism_subs: Option<&Value>,
) -> Vec<site_types::Arena> {
    let mut arenas = list_arenas(design, prism, prism_subs);
    for arena in &mut arenas {
        apply_emission(st, arena);
    }
    arenas
}

async fn get_network(State(st): State<SiteState>) -> impl IntoResponse {
    Json(network_stats(&st).await)
}

async fn get_landing(State(st): State<SiteState>) -> impl IntoResponse {
    let design = fetch_design_dash(&st).await;
    let prism = fetch_prism_status(&st).await;
    let prism_subs = fetch_prism_subs(&st, 200).await;
    let stats = network_stats(&st).await;
    let arenas = arenas_with_emission(&st, design.as_ref(), prism.as_ref(), prism_subs.as_ref());
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
    let activity = activity_from_lives(design.as_ref(), &design_runs, &prism_rows, 10);
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
    Json(arenas_with_emission(
        &st,
        design.as_ref(),
        prism.as_ref(),
        prism_subs.as_ref(),
    ))
}

async fn get_arena(State(st): State<SiteState>, Path(slug): Path<String>) -> Response {
    let Some(slug) = ArenaSlug::parse(&slug) else {
        return json_err(StatusCode::NOT_FOUND, "not_found", "unknown arena");
    };
    let mut arena = match slug {
        ArenaSlug::Coding => coding_arena(),
        ArenaSlug::Design => design_arena_from_dashboard(fetch_design_dash(&st).await.as_ref()),
        ArenaSlug::Prism => {
            let status = fetch_prism_status(&st).await;
            let subs = fetch_prism_subs(&st, 200).await;
            prism_arena_from_live(status.as_ref(), subs.as_ref())
        }
    };
    apply_emission(&st, &mut arena);
    Json(arena).into_response()
}

fn empty_leaderboard_json(page: u32, page_size: u32) -> Value {
    let empty = page_slice::<crate::LeaderboardRow>(&[], page, page_size);
    json!({
        "items": empty.items,
        "page": empty.page,
        "pageSize": empty.page_size,
        "total": empty.total,
        "pageCount": empty.page_count,
        "epoch": 0,
        "updatedAt": now_iso(),
    })
}

async fn design_leaderboard_json(
    st: &SiteState,
    page: u32,
    page_size: u32,
    q: Option<&str>,
) -> Value {
    let dash = fetch_design_dash(st).await;
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
        upstream::get_json_opt(st, DESIGN, &format!("/v1/rounds/{rid}/leaderboard")).await
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
    let previous_round = dash
        .as_ref()
        .and_then(|d| d.pointer("/leaderboard/previous_round"))
        .and_then(Value::as_u64)
        .or_else(|| round_id.map(|r| r.saturating_sub(1)));
    let epoch = dash
        .as_ref()
        .and_then(|d| d.get("epoch"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    // Mid-round before admin winners: current ratings are empty. Surface the
    // previous round's standings so the marketing board is not blank.
    let (board_ratings, board_previous, board_round_id) =
        if ratings.is_empty() && !previous.is_empty() {
            (previous.as_slice(), &[][..], previous_round)
        } else {
            (ratings.as_slice(), previous.as_slice(), round_id)
        };
    let mut rows = design_leaderboard(board_ratings, board_previous, epoch);
    decorate_leaderboard(st, &mut rows);
    if let Some(needle) = q.filter(|s| !s.trim().is_empty()) {
        rows.retain(|r| leaderboard_matches_query(r, needle));
    }
    let page_out = page_slice(&rows, page, page_size);
    let seconds_remaining = dash
        .as_ref()
        .and_then(|d| d.pointer("/round/seconds_remaining"))
        .and_then(Value::as_u64);
    let round_ends_at = dash
        .as_ref()
        .and_then(|d| d.pointer("/round/closes_at_secs"))
        .and_then(Value::as_u64)
        .map(|s| site_data::map::ms_to_iso(s.saturating_mul(1000)));
    json!({
        "items": page_out.items,
        "page": page_out.page,
        "pageSize": page_out.page_size,
        "total": page_out.total,
        "pageCount": page_out.page_count,
        "epoch": epoch,
        "roundId": board_round_id,
        "roundEndsAt": round_ends_at,
        "secondsRemaining": seconds_remaining,
        "updatedAt": now_iso(),
    })
}

async fn prism_leaderboard_json(
    st: &SiteState,
    page: u32,
    page_size: u32,
    q: Option<&str>,
) -> Value {
    let status = fetch_prism_status(st).await;
    let epoch = status
        .as_ref()
        .and_then(|s| s.get("epoch").and_then(Value::as_u64))
        .unwrap_or(0);
    let raw = fetch_prism_subs(st, 500).await;
    let rows = raw
        .as_ref()
        .and_then(|v| v.get("submissions"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut board = prism_bpb_leaderboard(&rows, epoch);
    decorate_leaderboard(st, &mut board);
    if let Some(needle) = q.filter(|s| !s.trim().is_empty()) {
        board.retain(|r| leaderboard_matches_query(r, needle));
    }
    let page_out = page_slice(&board, page, page_size);
    json!({
        "items": page_out.items,
        "page": page_out.page,
        "pageSize": page_out.page_size,
        "total": page_out.total,
        "pageCount": page_out.page_count,
        "epoch": epoch,
        "metric": "bpb",
        "updatedAt": now_iso(),
    })
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
    let needle = q.q.as_deref();
    match slug {
        ArenaSlug::Coding => Json(empty_leaderboard_json(page, page_size)).into_response(),
        ArenaSlug::Design => {
            Json(design_leaderboard_json(&st, page, page_size, needle).await).into_response()
        }
        ArenaSlug::Prism => {
            Json(prism_leaderboard_json(&st, page, page_size, needle).await).into_response()
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
    let needle = q.q.as_deref();
    match slug {
        ArenaSlug::Coding => {
            Json(page_slice::<crate::Submission>(&[], page, page_size)).into_response()
        }
        ArenaSlug::Design => {
            design_submissions_page(&st, page, page_size, status_filter, needle).await
        }
        ArenaSlug::Prism => {
            let raw = fetch_prism_subs(&st, 500).await;
            let rows = raw
                .as_ref()
                .and_then(|v| v.get("submissions"))
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            // Public gallery: champions only (current top + Score>0 ex-tops).
            let mut items: Vec<_> = rows
                .iter()
                .filter(|r| is_prism_champion_submission(r))
                .filter_map(prism_submission)
                .collect();
            decorate_submissions(&st, &mut items);
            if let Some(st_f) = status_filter {
                items.retain(|s| match st_f {
                    "scored" => s.status == crate::SubmissionStatus::Scored,
                    "pending" => s.status == crate::SubmissionStatus::Pending,
                    "failed" => s.status == crate::SubmissionStatus::Failed,
                    _ => true,
                });
            }
            if let Some(n) = needle.filter(|s| !s.trim().is_empty()) {
                items.retain(|s| submission_matches_query(s, n));
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
    q: Option<&str>,
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
        // Refresh stage from run detail for in-flight + terminal rows.
        let detail = if run_id.is_empty() {
            None
        } else {
            upstream::get_json_opt(st, DESIGN, &format!("/v1/runs/{run_id}")).await
        };
        if let Some(sub) = design_submission(run, &miner, detail.as_ref(), epoch) {
            items.push(sub);
        }
    }
    decorate_submissions(st, &mut items);
    if let Some(st_f) = status_filter {
        items.retain(|s| match st_f {
            "scored" => s.status == crate::SubmissionStatus::Scored,
            "pending" => s.status == crate::SubmissionStatus::Pending,
            "failed" => s.status == crate::SubmissionStatus::Failed,
            _ => true,
        });
    }
    if let Some(n) = q.filter(|s| !s.trim().is_empty()) {
        items.retain(|s| submission_matches_query(s, n));
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

/// Max per-submission detail fetches the window fans out for loss curves.
const PRISM_WINDOW_TELEMETRY_FANOUT: usize = 8;

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
    // Fan out to submission details for the best terminal runs so the window
    // renders real loss curves; bounded, and every miss falls back to the
    // single-point bpb curve in `prism_window`.
    let mut ids: Vec<(f64, String)> = rows
        .iter()
        .filter(|r| r.get("status").and_then(Value::as_str) == Some("terminated"))
        .filter_map(|r| {
            Some((
                r.get("bpb").and_then(Value::as_f64)?,
                r.get("id")?.as_str()?.to_owned(),
            ))
        })
        .collect();
    ids.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    ids.truncate(PRISM_WINDOW_TELEMETRY_FANOUT);
    let mut telemetry = HashMap::new();
    for (_, id) in ids {
        if let Some(detail) =
            upstream::get_json_opt(&st, PRISM, &format!("/v1/submissions/{id}")).await
        {
            if let Some(t) = prism_telemetry(&detail) {
                telemetry.insert(id, t);
            }
        }
    }
    Json(prism_window(
        recipe.as_ref(),
        status.as_ref(),
        &rows,
        &telemetry,
    ))
}

/// Full telemetry for one prism submission (loss series + gradient stats).
async fn get_prism_submission_telemetry(
    State(st): State<SiteState>,
    Path(id): Path<String>,
) -> Response {
    let detail = upstream::get_json_opt(&st, PRISM, &format!("/v1/submissions/{id}")).await;
    match detail.as_ref().and_then(prism_telemetry) {
        Some(t) => Json(t).into_response(),
        None => json_err(
            StatusCode::NOT_FOUND,
            "not_found",
            "unknown prism submission",
        ),
    }
}

async fn get_validators(
    State(st): State<SiteState>,
    Query(q): Query<PageQuery>,
) -> impl IntoResponse {
    let (_, _, validators) = chain_snapshot(&st);
    let page = page_slice(&validators, q.page.unwrap_or(1), q.page_size.unwrap_or(24));
    Json(page)
}

/// Sealed weight vector + configured emission split (trust root + bundle).
async fn get_site_weights(State(st): State<SiteState>) -> impl IntoResponse {
    Json(site_data::weights::site_weights(
        st.netuid,
        st.trust_root(),
        st.latest_sealed_bundle(),
    ))
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
    Json(activity_from_lives(
        design.as_ref(),
        &design_runs,
        &prism_rows,
        limit,
    ))
}

async fn get_metrics(
    State(st): State<SiteState>,
    Query(q): Query<MetricsQuery>,
) -> impl IntoResponse {
    let range = q.range.unwrap_or_else(|| "30".into());
    let (block, chain_epoch, validators) = chain_snapshot(&st);
    let design = fetch_design_dash(&st).await;
    let prism = fetch_prism_status(&st).await;
    let prism_subs = fetch_prism_subs(&st, 200).await;
    let arenas = arenas_with_emission(&st, design.as_ref(), prism.as_ref(), prism_subs.as_ref());
    let epoch = epoch_from_lives(design.as_ref(), prism.as_ref(), chain_epoch);
    let agents: u32 = arenas.iter().map(|a| a.agents).sum();
    let tao_price = site_data::price::tao_price_usd(&st.client, &st.tao_price).await;
    let weights =
        site_data::weights::site_weights(st.netuid, st.trust_root(), st.latest_sealed_bundle());

    Json(NetworkMetrics {
        range,
        epoch,
        kpis: site_data::metrics::kpis(agents, validators.len(), tao_price, block, &weights),
        emission: MetricsEmission {
            // No epoch-close history store yet — points stay empty rather than
            // invented; the configured split above is the honest current frame.
            points: Vec::new(),
            shares: site_data::metrics::emission_shares(&arenas),
            total_this_epoch: 0.0,
        },
        pass_rate: MetricsPassRate {
            points: Vec::new(),
            latest: Vec::new(),
        },
        population: MetricsPopulation {
            rows: site_data::metrics::population_rows(&arenas),
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

    async fn mount_site_mocks(design: &MockServer, prism: &MockServer) {
        Mock::given(method("GET"))
            .and(path("/v1/dashboard"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "epoch": 3,
                "leaderboard": {
                    "current_round": 9,
                    "ratings": [{"miner_hotkey":"aa","rating":1300,"wins":1,"losses":0}],
                    "previous_ratings": []
                },
                "round": {
                    "round_id": 9,
                    "closes_at_secs": 1_700_000_100_u64,
                    "seconds_remaining": 120
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
            .mount(design)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/harness/h1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "h1",
                "miner_hotkey": "aa".repeat(32)
            })))
            .mount(design)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/runs/run1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "run1",
                "status": "scored",
                "final_score": {"score": 1000},
                "prompt_title": "SaaS PR review",
                "prompt": "Build a three-page marketing site for a SaaS PR review tool.",
                "pages": [
                    {"path": "index.html", "bytes": 128, "raw_sha256": "aa"},
                    {"path": "index.png", "bytes": 4096, "raw_sha256": "bb"}
                ],
                "screenshot_url": "/challenge/design/v1/view/run1/index.png"
            })))
            .mount(design)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/status"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "epoch": 3,
                "backend": "sim"
            })))
            .mount(prism)
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
                    "n_params": 12_000_000_u64,
                    "score": {"kind":"score","value": 900},
                    "created_at_ms": 1_700_000_000_000_u64,
                    "updated_at_ms": 1_700_000_000_000_u64
                }]
            })))
            .mount(prism)
            .await;
        mount_prism_detail_mock(prism).await;
    }

    async fn mount_prism_detail_mock(prism: &MockServer) {
        Mock::given(method("GET"))
            .and(path("/v1/submissions/sub1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "submission": {
                    "id": "sub1",
                    "miner_hotkey": "bb".repeat(32),
                    "epoch": 3,
                    "status": "terminated",
                    "label": "base",
                    "bpb": 1.25,
                    "n_params": 12_000_000_u64,
                    "metrics": {
                        "bpb": 1.25,
                        "tokens_seen": 2048,
                        "wall_clock_seconds": 12.0,
                        "gpu_type": "SIM",
                        "n_params": 12_000_000_u64,
                        "val_rows": 256,
                        "telemetry": {
                            "finish_reason": "finish_evaluation",
                            "report_count": 5,
                            "loss_series": [
                                {"step": 1, "loss": 3.9, "grad_norm": 1.0, "at_secs": 2.0},
                                {"step": 2, "loss": 3.1, "grad_norm": 0.5, "at_secs": 4.0},
                                {"step": 3, "loss": 2.5, "grad_norm": 0.33, "at_secs": 6.0}
                            ]
                        }
                    }
                },
                "events": []
            })))
            .mount(prism)
            .await;
    }

    #[tokio::test]
    async fn arenas_and_design_submissions_from_mocks() {
        let (design, prism, st) = setup().await;
        mount_site_mocks(&design, &prism).await;
        let app = site_router(st);

        let (s, v) = call(app.clone(), "/v1/site/arenas").await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v.as_array().unwrap().len(), 3);
        assert_eq!(v[0]["slug"], "coding");
        assert_eq!(v[1]["bestScore"], "1,300");
        assert_eq!(v[1]["roundId"], 9);
        assert_eq!(v[1]["secondsRemaining"], 120);

        let (s, v) = call(app.clone(), "/v1/site/arenas/design/submissions").await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["items"][0]["stage"], "scored");
        assert_eq!(v["items"][0]["score"], 1000.0);
        // The full prompt brief is the public run label, not "design run <id>".
        assert_eq!(
            v["items"][0]["title"],
            "Build a three-page marketing site for a SaaS PR review tool."
        );
        assert_eq!(v["items"][0]["promptTitle"], "SaaS PR review");
        // Screenshots-only viewer: no html `url` key, only `screenshotUrl`.
        assert!(v["items"][0].get("url").is_none());
        assert_eq!(
            v["items"][0]["screenshotUrl"],
            "/challenge/design/v1/view/run1/index.png"
        );

        let (s, v) = call(app.clone(), "/v1/site/arenas/design/duels").await;
        assert_eq!(s, StatusCode::OK);
        assert!(v.as_array().unwrap().is_empty());

        let (s, v) = call(app.clone(), "/v1/site/arenas/prism/submissions").await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["items"][0]["stage"], "terminated");
        assert_eq!(v["items"][0]["bpb"], 1.25);
        assert_eq!(v["items"][0]["paramsM"], 12.0);
        let expected_hk = keystore::ss58_encode(&[0xbb; 32], keystore::BITTENSOR_SS58_PREFIX);
        assert_eq!(v["items"][0]["agent"]["hotkey"], expected_hk);
        assert_eq!(
            v["items"][0]["agent"]["operator"],
            format!(
                "{}…{}",
                &expected_hk[..8],
                &expected_hk[expected_hk.len() - 4..]
            )
        );

        let (s, v) = call(app.clone(), "/v1/site/arenas/prism/leaderboard").await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["metric"], "bpb");
        assert_eq!(v["items"][0]["elo"], 1.25);
        assert_eq!(v["items"][0]["bpb"], 1.25);
        assert_eq!(v["items"][0]["paramsM"], 12.0);

        let (s, v) = call(
            app.clone(),
            "/v1/site/arenas/prism/submissions/sub1/telemetry",
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["submissionId"], "sub1");
        assert_eq!(v["finishReason"], "finish_evaluation");
        assert_eq!(v["reportCount"], 5);
        assert_eq!(v["nParams"], 12_000_000_u64);
        let pts = v["points"].as_array().unwrap();
        assert_eq!(pts.len(), 3);
        assert_eq!(pts[0]["step"], 1);
        assert_eq!(pts[1]["gradNorm"], 0.5);

        let (s, v) = call(
            app.clone(),
            "/v1/site/arenas/prism/submissions/nope/telemetry",
        )
        .await;
        assert_eq!(s, StatusCode::NOT_FOUND, "{v}");

        let (s, v) = call(app.clone(), "/v1/site/arenas/coding/submissions").await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(v["total"], 0);

        // Hotkey / prompt search (`?q=`) filters design submissions.
        let expected_hk = keystore::ss58_encode(&[0xaa; 32], keystore::BITTENSOR_SS58_PREFIX);
        let (s, v) = call(
            app.clone(),
            &format!("/v1/site/arenas/design/submissions?q={}", &expected_hk[..8]),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["total"], 1, "{v}");
        let (s, v) = call(app.clone(), "/v1/site/arenas/design/submissions?q=nope-xyz").await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["total"], 0, "{v}");

        let (s, v) = call(app, "/v1/site/activity?limit=8").await;
        assert_eq!(s, StatusCode::OK, "{v}");
        let msgs: Vec<&str> = v
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|e| e.get("message").and_then(Value::as_str))
            .collect();
        assert!(
            msgs.iter()
                .any(|m| m.contains("Design evaluated") || m.contains("winner")),
            "{msgs:?}"
        );
    }

    #[tokio::test]
    async fn design_submissions_exclude_runs_without_screenshots() {
        let (design, _prism, st) = setup().await;
        // One scored run with a captured screenshot, one failed run that never
        // produced pages (dead view link before the screenshots-only viewer).
        Mock::given(method("GET"))
            .and(path("/v1/dashboard"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "epoch": 3,
                "leaderboard": {"current_round": 9, "ratings": [], "previous_ratings": []},
                "round": {"round_id": 9, "closes_at_secs": 1_700_000_100_u64, "seconds_remaining": 120},
                "recent_runs": [
                    {"id": "ok1", "status": "scored", "round_id": 9, "harness_id": "h1", "prompt_id": "p1", "error_detail": null, "updated_at_ms": 1_700_000_000_000_u64},
                    {"id": "bad1", "status": "failed", "round_id": 9, "harness_id": "h1", "prompt_id": "p1", "error_detail": "agent crashed", "updated_at_ms": 1_700_000_001_000_u64}
                ]
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
            .and(path("/v1/runs/ok1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "ok1",
                "status": "scored",
                "final_score": {"score": 1000},
                "pages": [{"path": "index.png", "bytes": 4096, "raw_sha256": "bb"}],
                "screenshot_url": "/challenge/design/v1/view/ok1/index.png"
            })))
            .mount(&design)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/runs/bad1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "bad1",
                "status": "failed",
                "error_detail": "agent crashed",
                "pages": []
            })))
            .mount(&design)
            .await;
        let app = site_router(st);

        let (s, v) = call(app, "/v1/site/arenas/design/submissions").await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["total"], 1, "{v}");
        let items = v["items"].as_array().unwrap();
        assert_eq!(items.len(), 1, "{v}");
        assert_eq!(items[0]["id"], "ok1");
        assert!(items[0].get("url").is_none());
        assert_eq!(
            items[0]["screenshotUrl"],
            "/challenge/design/v1/view/ok1/index.png"
        );
    }

    #[tokio::test]
    async fn design_leaderboard_falls_back_to_previous_round() {
        let (design, _prism, st) = setup().await;
        Mock::given(method("GET"))
            .and(path("/v1/dashboard"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "epoch": 3,
                "leaderboard": {
                    "current_round": 9,
                    "previous_round": 8,
                    "ratings": [],
                    "previous_ratings": [
                        {"miner_hotkey": "aa".repeat(32), "rating": 250, "wins": 1, "losses": 0}
                    ]
                },
                "round": {
                    "round_id": 9,
                    "closes_at_secs": 1_700_000_100_u64,
                    "seconds_remaining": 120
                },
                "recent_runs": []
            })))
            .mount(&design)
            .await;
        let app = site_router(st);

        let (s, v) = call(app, "/v1/site/arenas/design/leaderboard").await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["total"], 1, "{v}");
        assert_eq!(v["roundId"], 8, "{v}");
        assert_eq!(v["items"][0]["elo"], 250.0);
    }

    #[tokio::test]
    async fn arenas_carry_trust_root_shares_and_weights_endpoint() {
        use std::sync::Arc;
        use trustroot::{ChallengeEntry, ChallengesBody, ParticipantPolicy};
        let (design, prism, st) = setup().await;
        mount_site_mocks(&design, &prism).await;
        let entry = |id: &str, bps: u16| ChallengeEntry {
            id: id.as_bytes().to_vec(),
            public_key: [7u8; 32],
            emission_share_bps: bps,
            policy: ParticipantPolicy::AllMetagraphHotkeys,
        };
        let st = st.with_weights(
            Arc::new(ChallengesBody {
                challenges: vec![entry("design", 5_000), entry("prism", 5_000)],
            }),
            Arc::new(|| None),
        );
        let app = site_router(st);

        let (s, v) = call(app.clone(), "/v1/site/arenas").await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v[1]["emissionShare"], 0.5);
        assert_eq!(v[2]["emissionShare"], 0.5);
        // Unsealed: effective weights stay 0.
        assert_eq!(v[1]["weight"], 0.0);
        assert_eq!(v[2]["weight"], 0.0);

        let (s, v) = call(app.clone(), "/v1/site/weights").await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["sealed"], false);
        assert_eq!(v["burnShare"], 1.0);
        assert_eq!(v["emissionShares"][0]["arena"], "design");
        assert_eq!(v["emissionShares"][0]["share"], 0.5);
        assert_eq!(v["emissionShares"][1]["arena"], "prism");
        assert!(v["hotkeyWeights"].as_array().unwrap().is_empty());
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
        Mock::given(method("GET"))
            .and(path("/v1/submissions/x2"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "submission": {
                    "id": "x2",
                    "bpb": 1.0,
                    "metrics": {
                        "bpb": 1.0,
                        "n_params": 12_000_000_u64,
                        "telemetry": {
                            "finish_reason": "finish_evaluation",
                            "report_count": 2,
                            "loss_series": [
                                {"step": 1, "loss": 4.0},
                                {"step": 2, "loss": 2.0}
                            ]
                        }
                    }
                },
                "events": []
            })))
            .mount(&prism)
            .await;
        let app = site_router(st);
        let (s, v) = call(app, "/v1/site/arenas/prism/window").await;
        assert_eq!(s, StatusCode::OK, "{v}");
        assert_eq!(v["dataset"], "ds@pin");
        assert_eq!(v["series"].as_array().unwrap().len(), 2);
        assert_eq!(v["series"][0]["finalLoss"], 1.0);
        // x2 has a detail payload → real curve + params; x1 (no detail mock)
        // falls back to the single-point bpb curve at the axis end.
        assert_eq!(v["series"][0]["points"].as_array().unwrap().len(), 2);
        assert_eq!(v["series"][0]["params"], 12.0);
        assert_eq!(v["series"][0]["submissionId"], "x2");
        assert_eq!(v["tokenBudget"], 0);
        assert_eq!(v["series"][1]["points"].as_array().unwrap().len(), 1);
        assert_eq!(v["series"][1]["points"][0]["step"], 2);
    }
}
