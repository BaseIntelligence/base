//! Screenshot backfill: runs with `index.html` but no `index.png` get a
//! capture stored; runs already screenshotted (or without pages) are
//! skipped, so repeat passes are no-ops. Uses a stub "chromium" shell script
//! so no real browser is needed.

#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use base64::Engine;
use design_challenge::backfill::backfill_screenshots;
use design_challenge::screenshot::png_artifact_tuple;
use design_store::{DesignStore, HarnessRow, MemoryDesignStore, RunStage, RunState};

const FAKE_PNG: &[u8] = b"\x89PNG\r\n\x1a\nbackfill-pixels";

fn write_stub(dir: &Path) -> PathBuf {
    let fake = dir.join("fake.png");
    std::fs::write(&fake, FAKE_PNG).unwrap();
    let stub = dir.join("chrome-stub.sh");
    let mut f = std::fs::File::create(&stub).unwrap();
    write!(
        f,
        "#!/bin/sh\nout=\"\"\nfor a in \"$@\"; do case \"$a\" in --screenshot=*) out=\"${{a#--screenshot=}}\";; esac; done\ncase \"$*\" in *--dump-dom*) echo '<title>SHOTH=1800</title>'; exit 0;; esac\nif [ -n \"$out\" ]; then cp \"{}\" \"$out\"; exit 0; fi\nexit 1\n",
        fake.display()
    )
    .unwrap();
    drop(f);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&stub, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    stub
}

async fn seed_run(store: &MemoryDesignStore, id: &str, created_at_ms: u64) {
    store
        .insert_harness(&HarnessRow {
            id: format!("h-{id}"),
            miner_hotkey: "cd".repeat(32),
            miner_coldkey: None,
            agent_py: "def run(task, llm, out):\n    pass\n".into(),
            pyproject_toml: "[project]\nname='x'\nversion='0'\n".into(),
            extra_files: BTreeMap::new(),
            active: true,
            eliminated_until_round: 0,
            created_at_ms,
        })
        .await
        .unwrap();
    store
        .insert_run(&RunState {
            id: id.into(),
            round_id: 1,
            harness_id: format!("h-{id}"),
            prompt_id: "p01".into(),
            status: RunStage::Scored,
            artifact_digest: None,
            sanitize_report: None,
            agentic_verdict: None,
            error_detail: None,
            final_score: None,
            retry_count: 0,
            created_at_ms,
            updated_at_ms: created_at_ms,
        })
        .await
        .unwrap();
}

fn html_pages() -> Vec<(String, String, String, String, u32)> {
    ["index.html", "pricing.html", "components.html"]
        .into_iter()
        .map(|p| {
            (
                p.to_owned(),
                format!("<main>{p}</main>"),
                String::new(),
                "ab".repeat(32),
                42,
            )
        })
        .collect()
}

// Single test per binary: it exports DESIGN_CHROME_BIN for the stub browser.
#[tokio::test]
async fn backfill_captures_missing_and_is_idempotent() {
    let dir = std::env::temp_dir().join(format!("backfill-it-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let stub = write_stub(&dir);
    // Single test in this binary, so no parallel env mutation race.
    std::env::set_var("DESIGN_CHROME_BIN", &stub);

    let store = MemoryDesignStore::new();
    // Missing screenshot (3 HTML pages only).
    seed_run(&store, "run-a", 3).await;
    store.put_artifacts("run-a", &html_pages()).await.unwrap();
    // No pages at all (never sanitized) — not a candidate.
    seed_run(&store, "run-b", 2).await;
    // Already has a screenshot — skipped.
    seed_run(&store, "run-c", 1).await;
    let mut with_png = html_pages();
    with_png.push(png_artifact_tuple(FAKE_PNG));
    store.put_artifacts("run-c", &with_png).await.unwrap();

    let first = backfill_screenshots(&store, &dir.join("staging"), 100)
        .await
        .unwrap();
    assert_eq!(first.scanned, 3);
    assert_eq!(first.missing, 1);
    assert_eq!(first.captured, 1);
    assert_eq!(first.failed, 0);

    // HTML pages survive the screenshot-only put; PNG round-trips.
    let pages = store.list_pages("run-a").await.unwrap();
    assert_eq!(pages.len(), 4, "html pages must survive: {pages:?}");
    let b64 = store.get_page("run-a", "index.png").await.unwrap().unwrap();
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .unwrap();
    assert_eq!(decoded, FAKE_PNG);

    // Second pass is a no-op.
    let second = backfill_screenshots(&store, &dir.join("staging"), 100)
        .await
        .unwrap();
    assert_eq!(second.missing, 0);
    assert_eq!(second.captured, 0);
    assert_eq!(second.failed, 0);

    let _ = std::fs::remove_dir_all(&dir);
}
