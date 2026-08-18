//! Live Lium e2e (billable). Gated by `BASE_LIVE_PROVIDER_TESTS=1`.
//!
//! Flow: inventory → ensure SSH key → rent cheapest under cap → wait RUNNING →
//! SSH nvidia-smi + sealed eval → terminate + verify → write evidence JSON.
//!
//! Run:
//! ```text
//! BASE_LIVE_PROVIDER_TESTS=1 cargo test -p prism-lium --test live_e2e -- --ignored --nocapture
//! ```

#![forbid(unsafe_code)]
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::too_many_lines)]

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use prism_lium::{
    EvalJobBackend, GpuPreference, InstanceSpec, LiumClient, LiumSshConfig, MIN_LIFETIME_HOURS,
};

fn load_api_key() -> String {
    if let Ok(k) = std::env::var("LIUM_API_KEY") {
        if !k.trim().is_empty() {
            return k.trim().to_owned();
        }
    }
    let cred = PathBuf::from("/root/.config/prism-mission/credentials.env");
    let text = std::fs::read_to_string(&cred).expect("credentials.env");
    for line in text.lines() {
        let line = line.trim();
        let line = line.strip_prefix("export ").unwrap_or(line);
        if let Some(rest) = line.strip_prefix("LIUM_API_KEY=") {
            let v = rest.trim().trim_matches('"').trim_matches('\'');
            if !v.is_empty() {
                return v.to_owned();
            }
        }
    }
    panic!("LIUM_API_KEY not found");
}

fn load_ssh_pub() -> String {
    let p = PathBuf::from("/root/.config/prism-mission/lium_ssh_ed25519.pub");
    std::fs::read_to_string(p)
        .expect("ssh pub")
        .trim()
        .to_owned()
}

fn evidence_dir() -> PathBuf {
    let d = PathBuf::from("/root/.omo/evidence/prism-lium");
    let _ = std::fs::create_dir_all(&d);
    d
}

#[tokio::test]
#[ignore = "billable live Lium; set BASE_LIVE_PROVIDER_TESTS=1"]
async fn live_rent_ssh_eval_terminate() {
    if std::env::var("BASE_LIVE_PROVIDER_TESTS").ok().as_deref() != Some("1") {
        eprintln!("BASE_LIVE_PROVIDER_TESTS != 1; skip");
        return;
    }

    let api_key = load_api_key();
    // Never log api_key
    let mut ssh = LiumSshConfig::default_live();
    ssh.private_key_path = Some(PathBuf::from(
        "/root/.config/prism-mission/lium_ssh_ed25519",
    ));
    ssh.running_timeout_secs = 900;
    ssh.ssh_attempts = 10;
    ssh.ssh_retry_secs = 5;

    let client =
        LiumClient::with_config(api_key, prism_lium::LIUM_API_BASE_URL, ssh).expect("client");

    let mut trace = serde_json::json!({
        "started_at": now_rfc3339(),
        "challenge": "prism",
        "provider": "lium",
    });

    let balance_before = client.balance().await.unwrap_or(-1.0);
    trace["balance_before"] = serde_json::json!(balance_before);

    let max_price = std::env::var("PRISM_LIVE_MAX_PRICE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.5_f64);
    let gpu_count = prism_lium::pod_gpu_count_from_env();
    trace["requested_gpu_count"] = serde_json::json!(gpu_count);

    let offers = client
        .list_offers(Some(max_price))
        .await
        .expect("list_offers");
    trace["offers_under_cap"] = serde_json::json!(offers.len());
    assert!(
        !offers.is_empty(),
        "no offers under ${max_price}/gpu/hr — widen PRISM_LIVE_MAX_PRICE"
    );

    // Fail closed to the ranked SKU and requested width before any billable call.
    let mut sorted = offers;
    GpuPreference::default_prism().filter_sort_offers(&mut sorted, gpu_count);
    assert!(
        !sorted.is_empty(),
        "no {gpu_count}-GPU RTX 5090 offer under ${max_price}/gpu/hr"
    );
    let top: Vec<_> = sorted
        .iter()
        .take(5)
        .map(|o| {
            serde_json::json!({
                "id": o.id,
                "gpu_type": o.gpu_type,
                "gpu_count": o.gpu_count,
                "price_per_hour": o.price_per_hour,
            })
        })
        .collect();
    trace["candidate_offers"] = serde_json::json!(top);

    let pub_key = load_ssh_pub();
    let name = format!(
        "prism-live-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            % 1_000_000
    );

    let mut last_err = String::new();
    let mut rented = None;
    let mut attempts = Vec::new();

    for offer in sorted.into_iter().take(8) {
        let attempt_name = format!("{name}-{}", &offer.id[..8.min(offer.id.len())]);
        let spec = InstanceSpec {
            name: attempt_name,
            max_lifetime_hours: MIN_LIFETIME_HOURS,
            max_price_per_hour: max_price,
            gpu_count,
            image_digest: None,
            ssh_public_keys: vec![pub_key.clone()],
            ssh_key_name: Some("prism-mission-worker".into()),
            preferred_offer_id: Some(offer.id.clone()),
            template_id: None,
            // Exercise digest-only `PRISM_POD_IMAGE_REF` resolution, not a
            // historical provider-side template.
            template_name: None,
        };
        let mut att = serde_json::json!({
            "offer_id": offer.id,
            "gpu_type": offer.gpu_type,
            "gpu_count": offer.gpu_count,
            "price": offer.price_per_hour,
        });
        match client.provision(&spec).await {
            Ok(inst) => {
                att["result"] = serde_json::json!("rented");
                att["pod_id"] = serde_json::json!(inst.id);
                attempts.push(att);
                rented = Some((inst, offer));
                break;
            }
            Err(e) => {
                let msg = e.to_string();
                // ensure key never appears
                assert!(
                    !msg.contains(&load_api_key_prefix()),
                    "api key leaked in error"
                );
                att["result"] = serde_json::json!(format!("failed: {msg}"));
                attempts.push(att);
                last_err = msg;
            }
        }
    }
    trace["rent_attempts"] = serde_json::json!(attempts);

    let Some((inst, offer)) = rented else {
        write_trace(&trace, "failure_no_rent");
        panic!("could not rent any offer: {last_err}");
    };

    let pod_id = inst.id.clone();
    trace["pod_id"] = serde_json::json!(pod_id);
    trace["selected_gpu"] = serde_json::json!(offer.gpu_type);
    trace["selected_price"] = serde_json::json!(offer.price_per_hour);

    let arch = r#"
import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import NVFP4BlockScaling

class TinyLM(torch.nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, 16)
        self.head = torch.nn.Linear(16, vocab, bias=False)

    def forward(self, input_ids):
        return self.head(self.embed(input_ids))

def build_model(ctx):
    assert torch.cuda.device_count() >= 4, torch.cuda.device_count()
    assert te is not None and NVFP4BlockScaling is not None
    return TinyLM(int(ctx["vocab_size"]))
"#;
    let train = r#"
import torch

def train(model, ctx):
    width = min(4, int(ctx["gpu_count"]))
    assert width == 4, width
    inputs, labels = next(iter(ctx["train_stream"]))
    parallel = torch.nn.DataParallel(model, device_ids=list(range(width)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    logits = parallel(inputs)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return {"loss": float(loss.detach()), "data_parallel_width": width}
"#;

    let eval_result = {
        let r = client.exec_eval(&pod_id, arch, train, None).await;
        // Always terminate in finally
        let term = client.terminate(&pod_id).await;
        let mut verified = client.verify_terminated(&pod_id).await.unwrap_or(false);
        if !verified {
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            verified = client.verify_terminated(&pod_id).await.unwrap_or(false);
        }
        trace["terminate_ok"] = serde_json::json!(term.is_ok());
        if let Err(e) = &term {
            trace["terminate_err"] = serde_json::json!(e.to_string());
        }
        trace["termination_verified"] = serde_json::json!(verified);
        (r, verified)
    };

    let (eval, verified) = eval_result;
    match eval {
        Ok(metrics) => {
            trace["bpb"] = serde_json::json!(metrics.bpb);
            trace["gpu_type"] = serde_json::json!(metrics.gpu_type);
            trace["notes"] = serde_json::json!(metrics.notes);
            trace["tokens_seen"] = serde_json::json!(metrics.tokens_seen);
            trace["pod_manifest"] = serde_json::json!(metrics.pod_manifest);
            trace["battery"] = metrics.extra.get("battery").cloned().unwrap_or_default();
            assert!(metrics.bpb.is_finite() && metrics.bpb > 0.0);
            assert!(metrics.tokens_seen > 0, "train stream was not consumed");
            assert_eq!(metrics.netns, Some(true), "train/eval netns was not active");
            assert!(
                metrics.notes.contains("live") || metrics.gpu_type.is_some(),
                "expected live-attested metrics"
            );
        }
        Err(e) => {
            trace["eval_error"] = serde_json::json!(e.to_string());
            write_trace(&trace, "failure_eval");
            panic!("exec_eval failed: {e}");
        }
    }

    assert!(verified, "pod not verified terminated — billing leak risk");

    let balance_after = client.balance().await.unwrap_or(-1.0);
    trace["balance_after"] = serde_json::json!(balance_after);
    if balance_before >= 0.0 && balance_after >= 0.0 {
        let delta = balance_before - balance_after;
        trace["balance_delta"] = serde_json::json!(delta);
        assert!(delta < 2.0, "balance delta ${delta} exceeds $2 guardrail");
    }

    let pods_left = client.list_offers(None).await; // just keep client alive
    let _ = pods_left;

    trace["result"] = serde_json::json!("success");
    trace["finished_at"] = serde_json::json!(now_rfc3339());
    write_trace(&trace, "success");
}

fn load_api_key_prefix() -> String {
    let k = load_api_key();
    k.chars().take(8).collect()
}

fn now_rfc3339() -> String {
    // simple UTC-ish stamp without chrono dep
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("unix:{secs}")
}

fn write_trace(trace: &serde_json::Value, tag: &str) {
    let path = evidence_dir().join(format!("live_e2e_{tag}.json"));
    let mut t = trace.clone();
    if let Some(obj) = t.as_object_mut() {
        obj.insert("tag".into(), serde_json::json!(tag));
        obj.insert("finished_at".into(), serde_json::json!(now_rfc3339()));
    }
    std::fs::write(&path, serde_json::to_string_pretty(&t).unwrap_or_default())
        .expect("write evidence");
    eprintln!("evidence written to {}", path.display());
}
