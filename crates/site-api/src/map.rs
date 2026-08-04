//! Honest mappers from design/prism JSON → site contract (no invented scores).

use std::collections::{BTreeSet, HashMap};

use serde_json::Value;

use crate::frames::{coding_arena, design_frame, prism_frame};
use crate::types::{
    ActivityEvent, ActivitySeverity, Agent, Arena, ArenaSlug, LeaderboardRow, LossPoint,
    LossSeries, PrismWindow, RulesGate, SealedPaths, Submission, SubmissionStatus,
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
    let status_s = run
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("queued");
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
    Some(Submission {
        id: id.to_owned(),
        arena: ArenaSlug::Design,
        agent: agent_from_hotkey(miner_hotkey, epoch),
        prompt_id: format!("#{prompt_id}"),
        title: format!("design run {id}"),
        url: design_view_url(id),
        status,
        score: if status == SubmissionStatus::Scored {
            score
        } else {
            None
        },
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
    let failure_reason = if status == SubmissionStatus::Failed {
        row.get("error_detail")
            .and_then(Value::as_str)
            .map(str::to_owned)
    } else {
        None
    };
    Some(Submission {
        id: id.to_owned(),
        arena: ArenaSlug::Prism,
        agent: agent_from_hotkey(hk, epoch),
        prompt_id: format!("#epoch-{epoch}"),
        title: label.to_owned(),
        url: format!("/challenge/prism/v1/submissions/{id}"),
        status,
        score: if status == SubmissionStatus::Scored {
            score
        } else {
            None
        },
        failure_reason,
        submitted_at: ms_to_iso(ms),
    })
}

/// Prism window from recipe + terminal scored submissions (minimal series).
#[must_use]
pub fn prism_window(recipe: Option<&Value>, status: Option<&Value>, subs: &[Value]) -> PrismWindow {
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
    let mut series: Vec<(f64, LossSeries)> = Vec::new();
    for (i, row) in subs.iter().enumerate() {
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
        series.push((
            bpb,
            LossSeries {
                architecture: label,
                params: 0.0,
                final_loss: bpb,
                rank: 0,
                points: vec![LossPoint { step: 0, loss: bpb }],
            },
        ));
        let _ = i;
    }
    series.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let series: Vec<LossSeries> = series
        .into_iter()
        .enumerate()
        .map(|(i, (_, mut s))| {
            s.rank = u32::try_from(i + 1).unwrap_or(u32::MAX);
            s
        })
        .collect();
    PrismWindow {
        dataset,
        revision,
        token_budget: 0,
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
        param_ceiling: 0,
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
        let w = prism_window(Some(&recipe), None, &subs);
        assert_eq!(w.series.len(), 2);
        assert_eq!(w.series[0].rank, 1);
        assert!((w.series[0].final_loss - 1.1).abs() < f64::EPSILON);
        assert_eq!(w.series[0].points.len(), 1);
        assert_eq!(w.offset, "pinned");
        assert_eq!(w.token_budget, 0);
    }
}
