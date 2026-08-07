//! Re-sanitize backfill: style-stripped sanitized HTML is rebuilt from raw,
//! then `index.png` is force-recaptured. Uses a stub Chromium.

#![allow(clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use base64::Engine;
use design_challenge::screenshot::png_artifact_tuple;
use design_challenge_bin::resanitize::backfill_resanitize;
use design_store::{DesignStore, HarnessRow, MemoryDesignStore, RunStage, RunState};

const FAKE_PNG_OLD: &[u8] = b"\x89PNG\r\n\x1a\nold-unstyled";
const FAKE_PNG_NEW: &[u8] = b"\x89PNG\r\n\x1a\nnew-styled-pixels";

fn write_stub(dir: &Path) -> PathBuf {
    let fake = dir.join("fake.png");
    std::fs::write(&fake, FAKE_PNG_NEW).unwrap();
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

fn stripped_pages() -> Vec<(String, String, String, String, u32)> {
    let raw = r#"<!DOCTYPE html><html><head><style>
html{scroll-behavior:smooth} .hero{color:#c00}
</style></head><body><main class="hero">Hi</main></body></html>"#;
    // Simulate the historical bug: style block wiped.
    let sanitized = "<main class=\"hero\">Hi</main>";
    ["index.html", "pricing.html", "components.html"]
        .into_iter()
        .map(|p| {
            (
                p.to_owned(),
                sanitized.to_owned(),
                raw.to_owned(),
                "ab".repeat(32),
                u32::try_from(raw.len()).unwrap(),
            )
        })
        .collect()
}

#[tokio::test]
async fn resanitize_restores_style_and_force_screenshots() {
    let dir = std::env::temp_dir().join(format!("resanitize-it-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let stub = write_stub(&dir);
    std::env::set_var("DESIGN_CHROME_BIN", &stub);

    let store = MemoryDesignStore::new();
    seed_run(&store, "run-broken", 2).await;
    let mut pages = stripped_pages();
    pages.push(png_artifact_tuple(FAKE_PNG_OLD));
    store.put_artifacts("run-broken", &pages).await.unwrap();

    // Healthy run: already has style — must be skipped.
    seed_run(&store, "run-ok", 1).await;
    let ok_raw = r"<html><head><style>.x{color:red}</style></head><body>ok</body></html>";
    let ok_pages: Vec<_> = ["index.html", "pricing.html", "components.html"]
        .into_iter()
        .map(|p| {
            (
                p.to_owned(),
                ok_raw.to_owned(),
                ok_raw.to_owned(),
                "cd".repeat(32),
                10,
            )
        })
        .collect();
    store.put_artifacts("run-ok", &ok_pages).await.unwrap();

    let dry = backfill_resanitize(&store, &dir.join("staging"), 100, &[], 0, true)
        .await
        .unwrap();
    assert_eq!(dry.candidates, 1);
    assert_eq!(dry.resanitized, 0);
    assert_eq!(dry.skipped, 1);

    let live = backfill_resanitize(&store, &dir.join("staging"), 100, &[], 0, false)
        .await
        .unwrap();
    assert_eq!(live.candidates, 1);
    assert_eq!(live.resanitized, 1);
    assert_eq!(live.screenshots, 1);
    assert_eq!(live.failed, 0);

    let index = store
        .get_page("run-broken", "index.html")
        .await
        .unwrap()
        .unwrap();
    assert!(
        index.to_ascii_lowercase().contains("<style"),
        "style restored: {index}"
    );
    assert!(
        index.contains("scroll-behavior") || index.contains("color"),
        "{index}"
    );

    let png_b64 = store
        .get_page("run-broken", "index.png")
        .await
        .unwrap()
        .unwrap();
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(png_b64)
        .unwrap();
    assert_eq!(decoded, FAKE_PNG_NEW);

    let run = store.get_run("run-broken").await.unwrap().unwrap();
    assert!(run.artifact_digest.is_some());

    let _ = std::fs::remove_dir_all(&dir);
}
