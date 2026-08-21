//! Live proof: create a 1× B200 job-deployment and wait until Running.
//! Does not run the 1h harness. Requires `VERDA_*`. Never logs secrets.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use prism_lium::{EvalJobBackend, InstanceSpec};
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

#[tokio::test]
#[ignore = "live Verda B200 boot; set VERDA_*"]
async fn verda_b200_deployment_reaches_running() {
    let creds = creds().expect("VERDA_CLIENT_ID / SECRET / INFERENCE_KEY");
    std::env::set_var("PRISM_VERDA_COMPUTE", "B200");
    std::env::set_var(
        "PRISM_VERDA_IMAGE_REF",
        "docker.io/pytorch/pytorch@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755",
    );
    let client = VerdaClient::new(creds).unwrap();
    let offers = client.list_offers(None).await.expect("list compute");
    assert!(
        offers
            .iter()
            .any(|o| o.gpu_type.to_ascii_lowercase().contains("b200")),
        "no B200 SKU in serverless catalog: {:?}",
        offers.iter().map(|o| &o.gpu_type).collect::<Vec<_>>()
    );
    let name = format!(
        "prism-b200-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    );
    eprintln!("verda_b200_boot deployment={name}");
    let spec = InstanceSpec {
        name: name.clone(),
        max_lifetime_hours: 1.0,
        ..InstanceSpec::default()
    };
    let inst = match client.provision(&spec).await {
        Ok(i) => i,
        Err(e) => {
            let _ = client.terminate(&name).await;
            panic!("provision B200: {e}");
        }
    };
    assert_eq!(inst.provider, "verda");
    let gpu = inst.gpu_type.unwrap_or_default();
    assert!(
        gpu.to_ascii_lowercase().contains("b200"),
        "expected B200 compute, got {gpu}"
    );
    let running = client.instance_running(&name).await.unwrap_or(false);
    if !running {
        // Pull + schedule can take several minutes.
        for i in 0..40 {
            tokio::time::sleep(std::time::Duration::from_secs(15)).await;
            if client.instance_running(&name).await.unwrap_or(false) {
                eprintln!("verda_b200_boot running after {}s", (i + 1) * 15);
                break;
            }
        }
    }
    let ok = client.instance_running(&name).await.unwrap_or(false);
    let _ = client.terminate(&name).await;
    assert!(ok, "B200 job-deployment {name} never reached running");
}
