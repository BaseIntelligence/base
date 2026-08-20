//! Lium JSON field helpers (kept here for the `prism-lium` LOC cap).

use serde_json::Value;

use crate::{Instance, Offer};

/// First string value found at any of `keys` (top-level object lookups).
#[must_use]
pub fn get_str<'a>(v: &'a Value, keys: &[&str]) -> Option<&'a str> {
    keys.iter().find_map(|k| v.get(k).and_then(|x| x.as_str()))
}

/// First array found at any of `keys` (bare arrays pass through at call sites).
#[must_use]
pub fn get_array(v: &Value, keys: &[&str]) -> Vec<Value> {
    keys.iter()
        .find_map(|k| v.get(k).and_then(|x| x.as_array()))
        .cloned()
        .unwrap_or_default()
}

/// Parse one `/executors` row.
#[must_use]
pub fn parse_one_offer(item: &Value) -> Option<Offer> {
    let id = get_str(item, &["id", "executor_id"])?.to_owned();
    let gpu_type = get_str(item, &["gpu_type", "gpu_name", "machine_name"])
        .or_else(|| item.pointer("/machine/gpu_type").and_then(|x| x.as_str()))
        .unwrap_or("UNKNOWN")
        .to_owned();
    let raw_count = item
        .get("gpu_count")
        .and_then(|x| x.as_u64())
        .or_else(|| item.get("available_gpu_count").and_then(|x| x.as_u64()))
        .or_else(|| item.get("gpus").and_then(|x| x.as_u64()))
        .unwrap_or(1) as u32;
    let gpu_count = crate::effective_gpu_count(raw_count, &gpu_type);
    let price = item
        .get("price_per_hour")
        .or_else(|| item.get("price_per_gpu"))
        .or_else(|| item.get("price"))
        .or_else(|| item.pointer("/price/per_gpu_hour"))
        .and_then(|x| x.as_f64())
        .or_else(|| {
            get_str(item, &["price_per_hour", "price_per_gpu"]).and_then(|s| s.parse().ok())
        })
        .unwrap_or(f64::MAX);
    Some(Offer {
        id,
        gpu_type,
        gpu_count,
        price_per_hour: price,
        provider: "lium".into(),
    })
}

/// Parse a `/pods/{id}` object.
#[must_use]
pub fn parse_instance(v: &Value, fallback_id: &str) -> Instance {
    Instance {
        id: get_str(v, &["id", "pod_id"])
            .unwrap_or(fallback_id)
            .to_owned(),
        status: get_str(v, &["status", "state"])
            .unwrap_or("UNKNOWN")
            .to_owned(),
        provider: "lium".into(),
        gpu_type: get_str(v, &["gpu_type"])
            .or_else(|| v.pointer("/executor/gpu_type").and_then(|x| x.as_str()))
            .map(str::to_owned),
        ssh_connect_cmd: get_str(v, &["ssh_connect_cmd"]).map(str::to_owned),
    }
}

/// Pod id from a rent response.
#[must_use]
pub fn extract_pod_id(v: &Value) -> Option<String> {
    v.get("id")
        .or_else(|| v.get("pod_id"))
        .or_else(|| v.pointer("/pod/id"))
        .and_then(|x| x.as_str())
        .map(str::to_owned)
}
