//! In-memory harness log tails keyed by submission id (no secrets).

use std::collections::HashMap;
use std::sync::Mutex;

use serde_json::{json, Value};

/// One log line with a monotonic cursor.
#[derive(Debug, Clone)]
pub struct LogLine {
    /// Monotonic cursor (1-based sequence).
    pub seq: u64,
    /// Unix ms when appended.
    pub at_ms: u64,
    /// Sanitized harness text (never contains API keys).
    pub line: String,
}

/// Ring of recent harness lines per submission.
#[derive(Debug, Default)]
pub struct LogBuffer {
    inner: Mutex<HashMap<String, SubLogs>>,
}

#[derive(Debug, Default)]
struct SubLogs {
    next_seq: u64,
    lines: Vec<LogLine>,
}

const MAX_LINES: usize = 400;

impl LogBuffer {
    /// Empty buffer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn now_ms() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_millis() as u64)
    }

    /// Append one line.
    pub fn append(&self, submission_id: &str, line: &str) {
        let line = line.trim_end();
        if line.is_empty() {
            return;
        }
        if let Ok(mut g) = self.inner.lock() {
            let e = g.entry(submission_id.to_owned()).or_default();
            e.next_seq = e.next_seq.saturating_add(1);
            e.lines.push(LogLine {
                seq: e.next_seq,
                at_ms: Self::now_ms(),
                line: line.to_owned(),
            });
            if e.lines.len() > MAX_LINES {
                let drop_n = e.lines.len() - MAX_LINES;
                e.lines.drain(0..drop_n);
            }
        }
    }

    /// Replace buffer with the non-empty lines from a harvested tail.
    pub fn replace_tail(&self, submission_id: &str, tail: &str) {
        let lines: Vec<&str> = tail.lines().filter(|l| !l.trim().is_empty()).collect();
        if lines.is_empty() {
            return;
        }
        if let Ok(mut g) = self.inner.lock() {
            let e = g.entry(submission_id.to_owned()).or_default();
            e.lines.clear();
            for line in lines {
                e.next_seq = e.next_seq.saturating_add(1);
                e.lines.push(LogLine {
                    seq: e.next_seq,
                    at_ms: Self::now_ms(),
                    line: line.to_owned(),
                });
            }
            if e.lines.len() > MAX_LINES {
                let drop_n = e.lines.len() - MAX_LINES;
                e.lines.drain(0..drop_n);
            }
        }
    }

    /// Lines with `seq > since`, plus next cursor.
    #[must_use]
    pub fn since(&self, submission_id: &str, since: u64) -> (Vec<LogLine>, u64) {
        let Ok(g) = self.inner.lock() else {
            return (vec![], since);
        };
        let Some(e) = g.get(submission_id) else {
            return (vec![], since);
        };
        let out: Vec<LogLine> = e.lines.iter().filter(|l| l.seq > since).cloned().collect();
        let next = if out.is_empty() {
            e.next_seq.max(since)
        } else {
            out.last().map_or(since, |l| l.seq)
        };
        (out, next)
    }

    /// Drop buffer for a finished submission.
    pub fn clear(&self, submission_id: &str) {
        if let Ok(mut g) = self.inner.lock() {
            g.remove(submission_id);
        }
    }
}

/// JSON body for `GET /v1/submissions/{id}/logs?since=`.
pub async fn logs_json(
    buf: Option<&LogBuffer>,
    store: &dyn prism_store::PrismStore,
    id: &str,
    since: u64,
) -> Value {
    let (mut lines, mut next) = buf.map_or((vec![], since), |b| b.since(id, since));
    // Cold start / post-restart: surface stage-event heartbeats + error tails.
    if lines.is_empty() {
        if let Ok(evs) = store.events(id).await {
            for (i, ev) in evs.iter().enumerate() {
                let seq = (i as u64).saturating_add(1);
                if seq <= since {
                    continue;
                }
                let msg = ev.detail.as_ref().map_or_else(
                    || format!("stage={}", ev.stage.as_str()),
                    |d| {
                        if d.get("heartbeat").and_then(serde_json::Value::as_bool) == Some(true) {
                            format!(
                                "heartbeat pod={}",
                                d.get("pod_id").and_then(|p| p.as_str()).unwrap_or("?")
                            )
                        } else {
                            format!("stage={} detail={d}", ev.stage.as_str())
                        }
                    },
                );
                lines.push(LogLine {
                    seq,
                    at_ms: ev.at_ms,
                    line: msg,
                });
                next = seq;
            }
        }
        if let Ok(Some(row)) = store.get(id).await {
            if let Some(err) = row.error_detail.as_deref() {
                if let Some(idx) = err.find("harvested:") {
                    let tail = err[idx..].trim();
                    for (j, line) in tail.lines().enumerate() {
                        let seq = next.saturating_add(1 + j as u64);
                        if seq <= since {
                            continue;
                        }
                        lines.push(LogLine {
                            seq,
                            at_ms: row.updated_at_ms,
                            line: line.to_owned(),
                        });
                        next = seq;
                    }
                }
            }
        }
    }
    let live = store
        .get(id)
        .await
        .ok()
        .flatten()
        .is_some_and(|r| !r.status.is_terminal());
    json!({
        "submission_id": id,
        "logs": lines.iter().map(|l| json!({
            "seq": l.seq,
            "at_ms": l.at_ms,
            "line": l.line,
        })).collect::<Vec<_>>(),
        "next_since": next,
        "poll_hint_ms": if live { 2000 } else { 0 },
        "live": live,
    })
}
