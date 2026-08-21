//! Live Verda BYOK 1h dense-1b proof. Requires `VERDA_*` and
//! `PRISM_AUTOMODEL_PIN_DIR`. Never logs secrets.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use base64::Engine;
use prism_lium::{EvalJobBackend, InstanceSpec};
use prism_pipeline::{expand_zip_fields, tree_blob_for, SubmissionRequest};
use prism_verda::{VerdaClient, VerdaCreds};

fn creds() -> Option<VerdaCreds> {
    let client_id = std::env::var("VERDA_CLIENT_ID").ok()?;
    let client_secret = std::env::var("VERDA_CLIENT_SECRET").ok()?;
    let inference_key = std::env::var("VERDA_INFERENCE_KEY").ok()?;
    if client_id.is_empty() || client_secret.is_empty() || inference_key.is_empty() {
        return None;
    }
    Some(VerdaCreds {
        client_id,
        client_secret,
        inference_key,
    })
}

fn dense_zip() -> Vec<u8> {
    use std::io::Write;
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/external-miner/examples/dense-1b");
    let mut buf = std::io::Cursor::new(Vec::new());
    {
        let mut w = zip::ZipWriter::new(&mut buf);
        let opts = zip::write::SimpleFileOptions::default();
        for name in ["automodel.base", "automodel.patch", "prism.toml"] {
            w.start_file(name, opts).unwrap();
            w.write_all(&std::fs::read(root.join(name)).unwrap())
                .unwrap();
        }
        w.finish().unwrap();
    }
    buf.into_inner()
}

#[tokio::test]
#[ignore = "live Verda GPU 1h; set VERDA_*"]
async fn verda_live_dense_eval_1h() {
    let creds = creds().expect("VERDA_CLIENT_ID / SECRET / INFERENCE_KEY");
    std::env::set_var(
        "PRISM_VERDA_IMAGE_REF",
        // Public digest — nvcr.io stays Queue forever on miner Verda.
        "docker.io/pytorch/pytorch@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755",
    );
    if std::env::var("PRISM_TEST_TRAIN_MINUTES").is_err() {
        std::env::set_var("PRISM_TEST_TRAIN_MINUTES", "60");
    }
    let client = VerdaClient::new(creds).unwrap();
    let zip = dense_zip();
    let mut req = SubmissionRequest {
        miner_hotkey: "dd".repeat(32),
        zip_base64: Some(base64::engine::general_purpose::STANDARD.encode(zip)),
        ..Default::default()
    };
    expand_zip_fields(&mut req).expect("expand dense-1b");
    let tree = tree_blob_for(&req).expect("tree");
    let name = format!(
        "prism-1h-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    );
    eprintln!("verda_live deployment={name}");
    let spec = InstanceSpec {
        name: name.clone(),
        max_lifetime_hours: 2.0,
        ..InstanceSpec::default()
    };
    let inst = client.provision(&spec).await.expect("provision");
    assert_eq!(inst.provider, "verda");
    let eval = client
        .exec_eval(
            &inst.id,
            &req.architecture_py,
            &req.training_py,
            tree.as_deref(),
        )
        .await;
    let _ = client.terminate(&inst.id).await;
    let gone = client.verify_terminated(&inst.id).await.unwrap_or(false);
    let eval = eval.expect("exec_eval");
    eprintln!(
        "EVAL_METRICS bpb={} tokens_seen={} wall={} notes={:?} gpu={:?}",
        eval.bpb, eval.tokens_seen, eval.wall_clock_seconds, eval.notes, eval.gpu_type
    );
    assert!(gone, "deployment must be deleted");
    assert!(eval.tokens_seen > 0 || eval.bpb.is_finite());
}
