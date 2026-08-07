//! Honest mappers from design/prism JSON → site contract (no invented scores).

use std::collections::{BTreeSet, HashMap};

use serde_json::Value;

use site_types::{coding_arena, design_frame, prism_frame};
use site_types::{
    ActivityEvent, ActivitySeverity, Agent, Arena, ArenaSlug, LeaderboardRow, LossPoint,
    LossSeries, PrismTelemetry, PrismTelemetryPoint, PrismWindow, RulesGate, SealedPaths,
    Submission, SubmissionStatus,
};

/// Truncated hotkey → agent shell (no invented miner numbers / models).
#[must_use]
pub fn agent_from_hotkey(hotkey: &str, joined_epoch: u64) -> Agent {
    let hk = hotkey.trim();
    let short = if hk.len() >= 8 { &hk[..8] } else { hk };
    let slug = short.to_ascii_lowercase();
    Agent {
        slug: slug.clone(),
        handle: format!("@{slug}"),
        miner_number: "—".into(),
        model: "—".into(),
        operator: if hk.len() > 16 {
            format!("{}…{}", &hk[..8], &hk[hk.len().saturating_sub(4)..])
        } else {
            hk.to_owned()
        },
        joined_epoch,
    }
}

/// ISO-8601 from unix millis (UTC, second precision).
#[must_use]
pub fn ms_to_iso(ms: u64) -> String {
    let secs = ms / 1000;
    let (year, month, day, hour, minute, second) = civil_parts(secs);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.000Z")
}

/// `HH:MM:SS` UTC from millis.
#[must_use]
pub fn ms_to_clock(ms: u64) -> String {
    let secs = ms / 1000;
    let (_, _, _, hour, minute, second) = civil_parts(secs);
    format!("{hour:02}:{minute:02}:{second:02}")
}

#[allow(clippy::many_single_char_names)]
fn civil_parts(secs: u64) -> (i64, u32, u32, u32, u32, u32) {
    let days = i64::try_from(secs / 86_400).unwrap_or(0);
    let rem = secs % 86_400;
    let hour = u32::try_from(rem / 3600).unwrap_or(0);
    let minute = u32::try_from((rem % 3600) / 60).unwrap_or(0);
    let second = u32::try_from(rem % 60).unwrap_or(0);
    // Howard Hinnant civil-from-days.
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097).cast_unsigned();
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let mut year = yoe.cast_signed() + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = u32::try_from(doy - (153 * mp + 2) / 5 + 1).unwrap_or(1);
    let month_raw = if mp < 10 { mp + 3 } else { mp - 9 };
    if month_raw <= 2 {
        year += 1;
    }
    let month = u32::try_from(month_raw).unwrap_or(1);
    (year, month, day, hour, minute, second)
}

/// Map design run status → site submission status.
#[must_use]
pub fn map_design_status(status: &str) -> SubmissionStatus {
    match status {
        "scored" => SubmissionStatus::Scored,
        "failed" => SubmissionStatus::Failed,
        _ => SubmissionStatus::Pending,
    }
}

/// Map prism stage → site submission status.
#[must_use]
pub fn map_prism_status(status: &str) -> SubmissionStatus {
    match status {
        "terminated" => SubmissionStatus::Scored,
        "failed" => SubmissionStatus::Failed,
        _ => SubmissionStatus::Pending,
    }
}

/// Design view URL on the public gateway proxy.
#[must_use]
pub fn design_view_url(run_id: &str) -> String {
    format!("/challenge/design/v1/view/{run_id}/index.html")
}

/// Design full-page screenshot URL on the public gateway proxy.
#[must_use]
pub fn design_screenshot_url(run_id: &str) -> String {
    format!("/challenge/design/v1/view/{run_id}/index.png")
}

/// Enrich design arena frame from dashboard JSON.
#[must_use]
pub fn design_arena_from_dashboard(dash: Option<&Value>) -> Arena {
    let mut a = design_frame();
    let Some(dash) = dash else {
        return a;
    };
    let ratings = dash
        .pointer("/leaderboard/ratings")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut miners = BTreeSet::new();
    let mut best: Option<f64> = None;
    for r in &ratings {
        if let Some(hk) = r.get("miner_hotkey").and_then(Value::as_str) {
            miners.insert(hk.to_owned());
        }
        if let Some(rating) = num_f64_val(r.get("rating").unwrap_or(&Value::Null)) {
            best = Some(best.map_or(rating, |b| b.max(rating)));
        }
    }
    a.agents = u32::try_from(miners.len()).unwrap_or(u32::MAX);
    if let Some(b) = best {
        a.best_score = format_elo(b);
    }
    // Round clock from dashboard `/round` (same shape as `/v1/stats`).
    if let Some(rid) = dash
        .pointer("/round/round_id")
        .and_then(Value::as_u64)
        .or_else(|| {
            dash.pointer("/leaderboard/current_round")
                .and_then(Value::as_u64)
        })
    {
        a.round_id = Some(rid);
    }
    if let Some(closes) = dash
        .pointer("/round/closes_at_secs")
        .and_then(Value::as_u64)
    {
        a.round_ends_at = Some(ms_to_iso(closes.saturating_mul(1000)));
    }
    if let Some(rem) = dash
        .pointer("/round/seconds_remaining")
        .and_then(Value::as_u64)
    {
        a.seconds_remaining = Some(rem);
    }
    a
}

/// Prism arena from status + submissions list.
#[must_use]
pub fn prism_arena_from_live(status: Option<&Value>, subs: Option<&Value>) -> Arena {
    let mut a = prism_frame();
    let mut miners = BTreeSet::new();
    let mut best_bpb: Option<f64> = None;
    if let Some(arr) = subs
        .and_then(|v| v.get("submissions"))
        .and_then(Value::as_array)
    {
        for s in arr {
            if let Some(hk) = s.get("miner_hotkey").and_then(Value::as_str) {
                miners.insert(hk.to_owned());
            }
            if s.get("status").and_then(Value::as_str) == Some("terminated") {
                if let Some(bpb) = s.get("bpb").and_then(Value::as_f64) {
                    best_bpb = Some(match best_bpb {
                        Some(b) => b.min(bpb),
                        None => bpb,
                    });
                }
            }
        }
    }
    a.agents = u32::try_from(miners.len()).unwrap_or(u32::MAX);
    if let Some(b) = best_bpb {
        a.best_score = format!("{b:.4}");
    } else if status.is_some() {
        a.best_score = "—".into();
    }
    a
}

/// All three arenas (coding always paused).
#[must_use]
pub fn list_arenas(
    design_dash: Option<&Value>,
    prism_status: Option<&Value>,
    prism_subs: Option<&Value>,
) -> Vec<Arena> {
    vec![
        coding_arena(),
        design_arena_from_dashboard(design_dash),
        prism_arena_from_live(prism_status, prism_subs),
    ]
}

fn format_elo(v: f64) -> String {
    let rounded = v.round();
    #[allow(clippy::cast_possible_truncation)]
    let as_i = rounded as i64;
    let s = as_i.unsigned_abs().to_string();
    let mut out = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    let body: String = out.chars().rev().collect();
    if as_i < 0 {
        format!("-{body}")
    } else {
        body
    }
}

/// Design leaderboard from ratings (+ optional previous for real delta).
#[must_use]
pub fn design_leaderboard(
    ratings: &[Value],
    previous: &[Value],
    epoch: u64,
) -> Vec<LeaderboardRow> {
    let prev_map: HashMap<String, f64> = previous
        .iter()
        .filter_map(|r| {
            let hk = r.get("miner_hotkey")?.as_str()?.to_owned();
            let rating = num_f64(r.get("rating")?)?;
            Some((hk, rating))
        })
        .collect();
    let mut rows: Vec<_> = ratings
        .iter()
        .filter_map(|r| {
            let hk = r.get("miner_hotkey")?.as_str()?;
            let elo = num_f64(r.get("rating")?)?;
            let wins = num_u32(r.get("wins")).unwrap_or(0);
            let losses = num_u32(r.get("losses")).unwrap_or(0);
            let games = wins.saturating_add(losses);
            let win_rate = if games == 0 {
                0.0
            } else {
                f64::from(wins) / f64::from(games)
            };
            let delta7d = prev_map.get(hk).map_or(0.0, |p| elo - p);
            Some((elo, wins, losses, win_rate, delta7d, hk.to_owned()))
        })
        .collect();
    rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    rows.into_iter()
        .enumerate()
        .map(
            |(i, (elo, wins, losses, win_rate, delta7d, hk))| LeaderboardRow {
                rank: u32::try_from(i + 1).unwrap_or(u32::MAX),
                agent: agent_from_hotkey(&hk, epoch),
                elo,
                wins,
                losses,
                win_rate,
                submissions: wins.saturating_add(losses),
                delta7d,
                bpb: None,
                params_m: None,
            },
        )
        .collect()
}

/// Map one design `recent_run` (+ optional harness/run detail) to a submission.
#[must_use]
pub fn design_submission(
    run: &Value,
    miner_hotkey: &str,
    run_detail: Option<&Value>,
    epoch: u64,
) -> Option<Submission> {
    let id = run.get("id")?.as_str()?;
    let status_s = run_detail
        .and_then(|d| d.get("status"))
        .and_then(Value::as_str)
        .or_else(|| run.get("status").and_then(Value::as_str))
        .unwrap_or("queued");
    let stage = status_s.to_owned();
    let status = map_design_status(status_s);
    let prompt_id = run
        .get("prompt_id")
        .and_then(Value::as_str)
        .unwrap_or("—")
        .to_owned();
    let ms = run
        .get("updated_at_ms")
        .and_then(Value::as_u64)
        .or_else(|| run.get("created_at_ms").and_then(Value::as_u64))
        .unwrap_or(0);
    let mut score = None;
    let mut failure_reason = run
        .get("error_detail")
        .and_then(Value::as_str)
        .map(str::to_owned);
    if let Some(detail) = run_detail {
        if let Some(fs) = detail.get("final_score") {
            if let Some(v) = fs.get("score").and_then(num_f64_val) {
                score = Some(v);
            } else if let Some(c) = fs.get("no_score") {
                failure_reason = Some(format!("no_score:{c}"));
            }
        }
        if failure_reason.is_none() {
            failure_reason = detail
                .get("error_detail")
                .and_then(Value::as_str)
                .map(str::to_owned);
        }
    }
    let status_detail = failure_reason.clone().or_else(|| match status_s {
        "awaiting_admin" => Some("awaiting admin winners".into()),
        "awaiting_annotation" => Some("awaiting annotation".into()),
        "agentic_review" => Some("agentic anti-cheat review".into()),
        "installing" => Some("installing harness".into()),
        "running" => Some("running agent".into()),
        "sanitizing" => Some("sanitizing pages".into()),
        _ => None,
    });
    let screenshot_url = run_detail
        .and_then(|d| d.get("screenshot_url"))
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or_else(|| {
            // Prefer explicit pages list from run detail when present.
            let pages = run_detail.and_then(|d| d.get("pages")).and_then(Value::as_array);
            pages.and_then(|arr| {
                arr.iter().any(|p| p.get("path").and_then(Value::as_str) == Some("index.png"))
                    .then(|| design_screenshot_url(id))
            })
        });
    Some(Submission {
        id: id.to_owned(),
        arena: ArenaSlug::Design,
        agent: agent_from_hotkey(miner_hotkey, epoch),
        prompt_id: format!("#{prompt_id}"),
        title: format!("design run {id}"),
        url: design_view_url(id),
        screenshot_url,
        status,
        stage,
        status_detail,
        score: if status == SubmissionStatus::Scored {
            score
        } else {
            None
        },
        bpb: None,
        params_m: None,
        failure_reason: if status == SubmissionStatus::Failed {
            failure_reason
        } else {
            None
        },
        submitted_at: ms_to_iso(ms),
    })
}

/// Map prism list row → submission.
#[must_use]
pub fn prism_submission(row: &Value) -> Option<Submission> {
    let id = row.get("id")?.as_str()?;
    let hk = row
        .get("miner_hotkey")
        .and_then(Value::as_str)
        .unwrap_or("—");
    let status_s = row
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("queued");
    let stage = status_s.to_owned();
    let status = map_prism_status(status_s);
    let epoch = row.get("epoch").and_then(Value::as_u64).unwrap_or(0);
    let ms = row
        .get("created_at_ms")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let label = row
        .get("label")
        .and_then(Value::as_str)
        .unwrap_or("prism submission");
    let mut score = None;
    if let Some(s) = row.get("score") {
        if s.get("kind").and_then(Value::as_str) == Some("score") {
            score = s.get("value").and_then(num_f64_val);
        }
    }
    let bpb = row.get("bpb").and_then(Value::as_f64);
    let params_m = row
        .get("n_params")
        .and_then(Value::as_u64)
        .map(|p| p as f64 / 1e6);
    let failure_reason = if status == SubmissionStatus::Failed {
        row.get("error_detail")
            .and_then(Value::as_str)
            .map(str::to_owned)
    } else {
        None
    };
    let status_detail = failure_reason.clone().or_else(|| match status_s {
        "provisioning" => Some("provisioning eval pod".into()),
        "running" => Some("training on recipe".into()),
        "llm_review" => Some("LLM review".into()),
        "similarity" => Some("similarity check".into()),
        "scoring" => Some("scoring".into()),
        "terminated" => bpb.map(|b| format!("bpb={b:.4}")),
        _ => None,
    });
    Some(Submission {
        id: id.to_owned(),
        arena: ArenaSlug::Prism,
        agent: agent_from_hotkey(hk, epoch),
        prompt_id: format!("#epoch-{epoch}"),
        title: label.to_owned(),
        url: format!("/challenge/prism/v1/submissions/{id}"),
        screenshot_url: None,
        status,
        stage,
        status_detail,
        score: if status == SubmissionStatus::Scored {
            score
        } else {
            None
        },
        bpb,
        params_m,
        failure_reason,
        submitted_at: ms_to_iso(ms),
    })
}

/// Prism BPB leaderboard from terminal submissions (lower BPB ranks better).
///
/// `elo` carries the BPB value so the existing leaderboard row contract can
/// surface rankings without inventing Elo/duels; `bpb` / `paramsM` mirror the
/// measured values explicitly for telemetry-aware clients.
#[must_use]
pub fn prism_bpb_leaderboard(subs: &[Value], epoch: u64) -> Vec<LeaderboardRow> {
    let mut best: HashMap<String, (f64, Option<u64>)> = HashMap::new();
    let mut counts: HashMap<String, u32> = HashMap::new();
    for row in subs {
        if row.get("status").and_then(Value::as_str) != Some("terminated") {
            continue;
        }
        let Some(bpb) = row.get("bpb").and_then(Value::as_f64) else {
            continue;
        };
        let n_params = row.get("n_params").and_then(Value::as_u64);
        let hk = row
            .get("miner_hotkey")
            .and_then(Value::as_str)
            .unwrap_or("—")
            .to_owned();
        *counts.entry(hk.clone()).or_insert(0) += 1;
        best.entry(hk)
            .and_modify(|(b, p)| {
                if bpb < *b {
                    *b = bpb;
                    *p = n_params;
                }
            })
            .or_insert((bpb, n_params));
    }
    let mut rows: Vec<(f64, String, u32, Option<u64>)> = best
        .into_iter()
        .map(|(hk, (bpb, n_params))| {
            let n = counts.get(&hk).copied().unwrap_or(1);
            (bpb, hk, n, n_params)
        })
        .collect();
    rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    rows.into_iter()
        .enumerate()
        .map(|(i, (bpb, hk, submissions, n_params))| LeaderboardRow {
            rank: u32::try_from(i + 1).unwrap_or(u32::MAX),
            agent: agent_from_hotkey(&hk, epoch),
            elo: bpb,
            wins: 0,
            losses: 0,
            win_rate: 0.0,
            submissions,
            delta7d: 0.0,
            bpb: Some(bpb),
            params_m: n_params.map(|p| p as f64 / 1e6),
        })
        .collect()
}

/// Map one prism submission detail payload (`GET /v1/submissions/{id}`) to the
/// site telemetry contract. `None` when the payload has no `submission` object;
/// the series is empty until the harness publishes `metrics.telemetry`.
#[must_use]
pub fn prism_telemetry(detail: &Value) -> Option<PrismTelemetry> {
    let sub = detail.get("submission")?;
    let id = sub.get("id")?.as_str()?;
    let metrics = sub.get("metrics").filter(|m| !m.is_null());
    let tele = metrics.and_then(|m| m.get("telemetry"));
    let points = tele
        .and_then(|t| t.get("loss_series"))
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|p| {
                    Some(PrismTelemetryPoint {
                        step: p.get("step")?.as_u64()?,
                        loss: p.get("loss")?.as_f64()?,
                        grad_norm: p.get("grad_norm").and_then(Value::as_f64),
                        at_secs: p.get("at_secs").and_then(Value::as_f64),
                        layer_stats: p.get("layer_stats").cloned(),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    Some(PrismTelemetry {
        submission_id: id.to_owned(),
        bpb: sub
            .get("bpb")
            .and_then(Value::as_f64)
            .or_else(|| metrics.and_then(|m| m.get("bpb")).and_then(Value::as_f64)),
        n_params: metrics
            .and_then(|m| m.get("n_params"))
            .and_then(Value::as_u64),
        val_rows: metrics
            .and_then(|m| m.get("val_rows"))
            .and_then(Value::as_u64),
        gpu_type: metrics
            .and_then(|m| m.get("gpu_type"))
            .and_then(Value::as_str)
            .map(str::to_owned),
        wall_clock_seconds: metrics
            .and_then(|m| m.get("wall_clock_seconds"))
            .and_then(Value::as_f64),
        finish_reason: tele
            .and_then(|t| t.get("finish_reason"))
            .and_then(Value::as_str)
            .map(str::to_owned),
        report_count: tele
            .and_then(|t| t.get("report_count"))
            .and_then(Value::as_u64)
            .unwrap_or(0),
        points,
    })
}

/// Chart x-value for one telemetry point: prefer harness `layer_stats.tokens`
/// (tokens seen) so the window plots against the egalitarian token axis;
/// fall back to the optimizer step when tokens were not reported.
fn telemetry_x(point: &PrismTelemetryPoint) -> u32 {
    let tokens = point
        .layer_stats
        .as_ref()
        .and_then(|ls| ls.get("tokens"))
        .and_then(Value::as_f64)
        .filter(|t| t.is_finite() && *t >= 0.0)
        .map(|t| t as u64);
    let x = tokens.unwrap_or(point.step);
    u32::try_from(x).unwrap_or(u32::MAX)
}

/// Prism window from recipe + terminal scored submissions.
///
/// Series carry the real miner-reported loss curves when `telemetry` has a
/// payload for the submission id; otherwise they fall back to the minimal
/// single-point `[final_bpb]` curve (pre-telemetry recipes, upstream miss).
/// Params prefer telemetry `n_params`, then the list-row `n_params` so
/// historical runs still surface a measured size when the detail blob is thin.
#[must_use]
pub fn prism_window(
    recipe: Option<&Value>,
    status: Option<&Value>,
    subs: &[Value],
    telemetry: &HashMap<String, PrismTelemetry>,
) -> PrismWindow {
    let dataset = recipe
        .and_then(|r| r.get("dataset_ref"))
        .and_then(Value::as_str)
        .unwrap_or("—")
        .to_owned();
    let revision = recipe
        .and_then(|r| r.get("version"))
        .and_then(Value::as_str)
        .or_else(|| {
            recipe
                .and_then(|r| r.get("pin_hex"))
                .and_then(Value::as_str)
                .map(|p| if p.len() >= 12 { &p[..12] } else { p })
        })
        .unwrap_or("—")
        .to_owned();
    let provider = status
        .and_then(|s| s.get("backend"))
        .and_then(Value::as_str)
        .unwrap_or("—")
        .to_owned();
    let pin = recipe
        .and_then(|r| r.get("pin_hex"))
        .and_then(Value::as_str)
        .unwrap_or("—");
    // Recipe publishes max_params in absolute count; site contract uses millions.
    let param_ceiling = recipe
        .and_then(|r| r.get("max_params"))
        .and_then(Value::as_u64)
        .map(|p| p / 1_000_000)
        .unwrap_or(0);
    let mut series: Vec<(f64, LossSeries)> = Vec::new();
    for row in subs {
        if row.get("status").and_then(Value::as_str) != Some("terminated") {
            continue;
        }
        let Some(bpb) = row.get("bpb").and_then(Value::as_f64) else {
            continue;
        };
        let id = row.get("id").and_then(Value::as_str).unwrap_or("?");
        let label = row
            .get("label")
            .and_then(Value::as_str)
            .map_or_else(|| format!("run:{}", &id[..id.len().min(8)]), str::to_owned);
        let tele = telemetry.get(id);
        let points = tele
            .map(|t| {
                t.points
                    .iter()
                    .map(|p| LossPoint {
                        step: telemetry_x(p),
                        loss: p.loss,
                    })
                    .collect::<Vec<_>>()
            })
            .filter(|pts| !pts.is_empty())
            .unwrap_or_else(|| vec![LossPoint { step: 0, loss: bpb }]);
        let params = tele
            .and_then(|t| t.n_params)
            .or_else(|| row.get("n_params").and_then(Value::as_u64))
            .map_or(0.0, |p| p as f64 / 1e6);
        series.push((
            bpb,
            LossSeries {
                architecture: label,
                submission_id: Some(id.to_owned()),
                params,
                final_loss: bpb,
                rank: 0,
                points,
            },
        ));
    }
    series.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut series: Vec<LossSeries> = series
        .into_iter()
        .enumerate()
        .map(|(i, (_, mut s))| {
            s.rank = u32::try_from(i + 1).unwrap_or(u32::MAX);
            s
        })
        .collect();
    // Axis span = max tokens (or steps) observed across curves so lossPath can
    // draw; single-point historical fallbacks (step 0) move to the right edge.
    let token_budget = series
        .iter()
        .flat_map(|s| s.points.iter().map(|p| u64::from(p.step)))
        .max()
        .unwrap_or(0);
    if token_budget > 0 {
        let end = u32::try_from(token_budget).unwrap_or(u32::MAX);
        for s in &mut series {
            if s.points.len() == 1 && s.points[0].step == 0 {
                s.points[0].step = end;
            }
        }
    }
    PrismWindow {
        dataset,
        revision,
        token_budget,
        offset: "pinned".into(),
        rules_gate: RulesGate {
            provider,
            passed: recipe.is_some(),
        },
        image_digest: if pin == "—" {
            "—".into()
        } else {
            format!("recipe:{pin}")
        },
        mid_run_mutation: false,
        sealed_paths: SealedPaths {
            verified: 0,
            total: 0,
        },
        param_ceiling,
        series,
    }
}

/// Activity lines from design `recent_runs` + prism submissions (bounded).
#[must_use]
pub fn activity_from_lives(
    design_runs: &[Value],
    prism_subs: &[Value],
    limit: usize,
) -> Vec<ActivityEvent> {
    let mut events: Vec<(u64, ActivityEvent)> = Vec::new();
    for r in design_runs {
        let id = r.get("id").and_then(Value::as_str).unwrap_or("?");
        let status = r.get("status").and_then(Value::as_str).unwrap_or("");
        let ms = r.get("updated_at_ms").and_then(Value::as_u64).unwrap_or(0);
        let (sev, msg) = match status {
            "scored" => (ActivitySeverity::Score, format!("design run {id} scored")),
            "failed" => (ActivitySeverity::Fail, format!("design run {id} failed")),
            "awaiting_admin" => (
                ActivitySeverity::Settle,
                format!("design run {id} awaiting admin"),
            ),
            _ => continue,
        };
        events.push((
            ms,
            ActivityEvent {
                id: format!("design:{id}:{status}"),
                at: ms_to_clock(ms),
                severity: sev,
                message: msg,
            },
        ));
    }
    for r in prism_subs {
        let id = r.get("id").and_then(Value::as_str).unwrap_or("?");
        let status = r.get("status").and_then(Value::as_str).unwrap_or("");
        let ms = r
            .get("updated_at_ms")
            .and_then(Value::as_u64)
            .or_else(|| r.get("created_at_ms").and_then(Value::as_u64))
            .unwrap_or(0);
        let (sev, msg) = match status {
            "terminated" => (
                ActivitySeverity::Score,
                format!("prism submission {id} terminated"),
            ),
            "failed" => (
                ActivitySeverity::Fail,
                format!("prism submission {id} failed"),
            ),
            _ => continue,
        };
        events.push((
            ms,
            ActivityEvent {
                id: format!("prism:{id}:{status}"),
                at: ms_to_clock(ms),
                severity: sev,
                message: msg,
            },
        ));
    }
    events.sort_by_key(|b| std::cmp::Reverse(b.0));
    events.into_iter().take(limit).map(|(_, e)| e).collect()
}

fn num_f64(v: &Value) -> Option<f64> {
    v.as_f64()
        .or_else(|| v.as_u64().map(|u| u as f64))
        .or_else(|| v.as_i64().map(|i| i as f64))
}

fn num_f64_val(v: &Value) -> Option<f64> {
    num_f64(v)
}

fn num_u32(v: Option<&Value>) -> Option<u32> {
    let v = v?;
    v.as_u64()
        .and_then(|u| u32::try_from(u).ok())
        .or_else(|| v.as_i64().and_then(|i| u32::try_from(i).ok()))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;
    use serde_json::json;

    #[test]
    fn design_view_url_shape() {
        assert_eq!(
            design_view_url("abc"),
            "/challenge/design/v1/view/abc/index.html"
        );
        assert_eq!(
            design_screenshot_url("abc"),
            "/challenge/design/v1/view/abc/index.png"
        );
    }

    #[test]
    fn leaderboard_maps_rating_to_elo_with_real_delta() {
        let cur = vec![json!({"miner_hotkey":"aa","rating":1200,"wins":2,"losses":1})];
        let prev = vec![json!({"miner_hotkey":"aa","rating":1100,"wins":1,"losses":1})];
        let rows = design_leaderboard(&cur, &prev, 9);
        assert_eq!(rows.len(), 1);
        assert!((rows[0].elo - 1200.0).abs() < f64::EPSILON);
        assert!((rows[0].delta7d - 100.0).abs() < f64::EPSILON);
        assert_eq!(rows[0].rank, 1);
    }

    #[test]
    fn prism_window_series_from_bpb_only() {
        let recipe = json!({
            "dataset_ref": "HuggingFaceFW/fineweb-edu@sample/10BT",
            "version": "1.0.1",
            "pin_hex": "abcd1234ffff"
        });
        let subs = vec![
            json!({"id":"deadbeef01","status":"terminated","bpb":1.5,"label":"a"}),
            json!({"id":"cafebabe02","status":"queued","bpb":0.1,"label":"b"}),
            json!({"id":"facefeed03","status":"terminated","bpb":1.1,"label":"c"}),
        ];
        let w = prism_window(Some(&recipe), None, &subs, &HashMap::new());
        assert_eq!(w.series.len(), 2);
        assert_eq!(w.series[0].rank, 1);
        assert!((w.series[0].final_loss - 1.1).abs() < f64::EPSILON);
        assert_eq!(w.series[0].points.len(), 1);
        assert_eq!(w.offset, "pinned");
        // No telemetry → single-point fallbacks stay at step 0; budget stays 0.
        assert_eq!(w.token_budget, 0);
        assert_eq!(w.series[0].submission_id.as_deref(), Some("facefeed03"));
    }

    #[test]
    fn prism_window_uses_real_telemetry_series_when_present() {
        let subs = vec![
            json!({"id":"deadbeef01","status":"terminated","bpb":1.5,"label":"a"}),
            json!({"id":"facefeed03","status":"terminated","bpb":1.1,"label":"c"}),
        ];
        let mut telemetry = HashMap::new();
        telemetry.insert(
            "facefeed03".to_owned(),
            PrismTelemetry {
                submission_id: "facefeed03".into(),
                bpb: Some(1.1),
                n_params: Some(12_000_000),
                val_rows: Some(256),
                gpu_type: Some("SIM".into()),
                wall_clock_seconds: Some(12.0),
                finish_reason: Some("finish_evaluation".into()),
                report_count: 3,
                points: vec![
                    PrismTelemetryPoint {
                        step: 1,
                        loss: 4.0,
                        grad_norm: Some(1.0),
                        at_secs: Some(0.5),
                        layer_stats: None,
                    },
                    PrismTelemetryPoint {
                        step: 2,
                        loss: 2.0,
                        grad_norm: None,
                        at_secs: None,
                        layer_stats: None,
                    },
                ],
            },
        );
        let w = prism_window(None, None, &subs, &telemetry);
        assert_eq!(w.series.len(), 2);
        // Rank 1 (bpb 1.1) carries the real curve + params; rank 2 falls back.
        assert_eq!(w.series[0].points.len(), 2);
        assert!((w.series[0].points[0].loss - 4.0).abs() < f64::EPSILON);
        assert_eq!(w.series[0].points[0].step, 1);
        assert!((w.series[0].params - 12.0).abs() < f64::EPSILON);
        assert_eq!(w.series[1].points.len(), 1);
        assert!((w.series[1].params - 0.0).abs() < f64::EPSILON);
        // Budget follows the max x across curves; single-point fallback sits at end.
        assert_eq!(w.token_budget, 2);
        assert_eq!(w.series[1].points[0].step, 2);
    }

    #[test]
    fn prism_window_uses_tokens_and_list_n_params() {
        let recipe = json!({"max_params": 350_000_000_u64, "dataset_ref": "ds"});
        let subs = vec![json!({
            "id":"deadbeef01",
            "status":"terminated",
            "bpb":1.5,
            "label":"a",
            "n_params": 24_000_000_u64
        })];
        let mut telemetry = HashMap::new();
        telemetry.insert(
            "deadbeef01".to_owned(),
            PrismTelemetry {
                submission_id: "deadbeef01".into(),
                bpb: Some(1.5),
                n_params: None, // force list-row fallback
                val_rows: None,
                gpu_type: None,
                wall_clock_seconds: None,
                finish_reason: Some("finish_evaluation".into()),
                report_count: 2,
                points: vec![
                    PrismTelemetryPoint {
                        step: 1,
                        loss: 4.0,
                        grad_norm: None,
                        at_secs: None,
                        layer_stats: Some(json!({"tokens": 100_000.0})),
                    },
                    PrismTelemetryPoint {
                        step: 2,
                        loss: 2.0,
                        grad_norm: None,
                        at_secs: None,
                        layer_stats: Some(json!({"tokens": 2_000_000.0})),
                    },
                ],
            },
        );
        let w = prism_window(Some(&recipe), None, &subs, &telemetry);
        assert_eq!(w.param_ceiling, 350);
        assert_eq!(w.token_budget, 2_000_000);
        assert_eq!(w.series[0].points[0].step, 100_000);
        assert_eq!(w.series[0].points[1].step, 2_000_000);
        assert!((w.series[0].params - 24.0).abs() < f64::EPSILON);
    }

    #[test]
    fn prism_telemetry_parses_detail_payload() {
        let detail = json!({
            "submission": {
                "id": "sub1",
                "bpb": 1.25,
                "metrics": {
                    "bpb": 1.25,
                    "wall_clock_seconds": 12.0,
                    "gpu_type": "NVIDIA RTX 5090",
                    "n_params": 12_000_000_u64,
                    "val_rows": 256,
                    "telemetry": {
                        "finish_reason": "finish_evaluation",
                        "report_count": 2,
                        "loss_series": [
                            {"step": 1, "loss": 4.0, "grad_norm": 0.9, "at_secs": 0.5,
                             "layer_stats": {"head": 0.1}},
                            {"step": 2, "loss": 3.5}
                        ]
                    }
                }
            },
            "events": []
        });
        let t = prism_telemetry(&detail).unwrap();
        assert_eq!(t.submission_id, "sub1");
        assert!((t.bpb.unwrap() - 1.25).abs() < f64::EPSILON);
        assert_eq!(t.n_params, Some(12_000_000));
        assert_eq!(t.finish_reason.as_deref(), Some("finish_evaluation"));
        assert_eq!(t.report_count, 2);
        assert_eq!(t.points.len(), 2);
        assert_eq!(t.points[0].step, 1);
        assert!(t.points[0].layer_stats.is_some());
        assert!(t.points[1].grad_norm.is_none());
    }

    #[test]
    fn prism_telemetry_handles_missing_metrics() {
        let detail = json!({"submission": {"id": "sub1", "bpb": null, "metrics": null}});
        let t = prism_telemetry(&detail).unwrap();
        assert!(t.points.is_empty());
        assert!(t.bpb.is_none());
        assert!(t.finish_reason.is_none());
        assert!(prism_telemetry(&json!({"events": []})).is_none());
    }

    #[test]
    fn prism_leaderboard_exposes_bpb_and_params_fields() {
        let subs = vec![
            json!({"id":"a","status":"terminated","bpb":2.0,"miner_hotkey":"aa","n_params":12_000_000_u64}),
            json!({"id":"b","status":"terminated","bpb":1.0,"miner_hotkey":"bb"}),
        ];
        let rows = prism_bpb_leaderboard(&subs, 3);
        assert_eq!(rows.len(), 2);
        assert!((rows[0].bpb.unwrap() - 1.0).abs() < f64::EPSILON);
        assert!(rows[0].params_m.is_none());
        assert!((rows[1].params_m.unwrap() - 12.0).abs() < f64::EPSILON);
        // Backwards compat: old payloads without the new fields still decode.
        let old: LeaderboardRow = serde_json::from_value(json!({
            "rank": 1,
            "agent": {"slug":"a","handle":"@a","minerNumber":"—","model":"—","operator":"a","joinedEpoch":0},
            "elo": 1.0, "wins": 0, "losses": 0, "winRate": 0.0, "submissions": 1, "delta7d": 0.0
        }))
        .unwrap();
        assert!(old.bpb.is_none());
    }

    #[test]
    fn design_submission_exposes_fine_stage() {
        let run = json!({
            "id": "r1",
            "status": "agentic_review",
            "prompt_id": "p1",
            "updated_at_ms": 1_700_000_000_000_u64
        });
        let sub = design_submission(&run, "aabbccdd", None, 1).unwrap();
        assert_eq!(sub.status, SubmissionStatus::Pending);
        assert_eq!(sub.stage, "agentic_review");
        assert!(sub.status_detail.as_deref().unwrap().contains("agentic"));
        assert!(sub.bpb.is_none());
    }

    #[test]
    fn prism_submission_surfaces_bpb_and_stage() {
        let row = json!({
            "id": "s1",
            "miner_hotkey": "bb".repeat(32),
            "epoch": 2,
            "status": "terminated",
            "label": "base",
            "bpb": 1.25,
            "score": {"kind":"score","value": 900},
            "created_at_ms": 1_700_000_000_000_u64
        });
        let sub = prism_submission(&row).unwrap();
        assert_eq!(sub.status, SubmissionStatus::Scored);
        assert_eq!(sub.stage, "terminated");
        assert!((sub.bpb.unwrap() - 1.25).abs() < f64::EPSILON);
        assert!((sub.score.unwrap() - 900.0).abs() < f64::EPSILON);
    }

    #[test]
    fn prism_bpb_leaderboard_ranks_lower_first() {
        let subs = vec![
            json!({"id":"a","status":"terminated","bpb":2.0,"miner_hotkey":"aa"}),
            json!({"id":"b","status":"terminated","bpb":1.0,"miner_hotkey":"bb"}),
            json!({"id":"c","status":"queued","bpb":0.1,"miner_hotkey":"cc"}),
            json!({"id":"d","status":"terminated","bpb":0.5,"miner_hotkey":"aa"}),
        ];
        let rows = prism_bpb_leaderboard(&subs, 3);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].rank, 1);
        assert!((rows[0].elo - 0.5).abs() < f64::EPSILON);
        assert_eq!(rows[0].submissions, 2);
        assert!((rows[1].elo - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn design_arena_round_clock_from_dashboard() {
        let dash = json!({
            "round": {
                "round_id": 42,
                "closes_at_secs": 1_700_000_100_u64,
                "seconds_remaining": 99
            },
            "leaderboard": {"ratings": []}
        });
        let a = design_arena_from_dashboard(Some(&dash));
        assert_eq!(a.round_id, Some(42));
        assert_eq!(a.seconds_remaining, Some(99));
        assert!(a.round_ends_at.as_ref().unwrap().starts_with("20"));
    }
}
