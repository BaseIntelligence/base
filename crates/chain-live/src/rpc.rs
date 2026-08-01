//! Blocking JSON-RPC client for Substrate-based chains.

use chain::ChainError;
use serde_json::{json, Value};

/// Runtime version fields checked against the lockfile before signing.
#[derive(Debug, Clone)]
pub struct RuntimeVersion {
    /// `specVersion` from `state_getRuntimeVersion`.
    pub spec_version: u32,
    /// `transactionVersion`.
    pub transaction_version: u32,
    /// `specName`.
    pub spec_name: String,
}

/// Blocking HTTPS JSON-RPC client.
#[derive(Debug)]
pub struct LiveChainRpc {
    http: reqwest::blocking::Client,
    endpoint: String,
}

impl LiveChainRpc {
    /// Create a client. `wss://` is rewritten to `https://`.
    ///
    /// # Errors
    /// HTTP client build failure.
    pub fn connect(endpoint: &str) -> Result<Self, ChainError> {
        let http = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| ChainError::Other(format!("http client: {e}")))?;
        Ok(Self {
            http,
            endpoint: http_endpoint(endpoint),
        })
    }

    /// Raw JSON-RPC call.
    fn rpc(&self, method: &str, params: &Value) -> Result<Value, ChainError> {
        let body = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        });
        let resp = self
            .http
            .post(&self.endpoint)
            .json(&body)
            .send()
            .map_err(|e| ChainError::Other(format!("rpc {method}: {e}")))?;
        let v: Value = resp
            .json()
            .map_err(|e| ChainError::Other(format!("rpc {method} json: {e}")))?;
        if let Some(err) = v.get("error") {
            return Err(ChainError::Other(format!("rpc {method} error: {err}")));
        }
        v.get("result")
            .cloned()
            .ok_or_else(|| ChainError::Other(format!("rpc {method}: missing result")))
    }

    /// `state_getStorage` — returns `None` if the key has no value.
    ///
    /// # Errors
    /// Transport or hex decode failure.
    pub fn state_get_storage(&self, key: &[u8]) -> Result<Option<Vec<u8>>, ChainError> {
        let hex_key = format!("0x{}", hex::encode(key));
        let result = self.rpc("state_getStorage", &json!([hex_key]))?;
        if result.is_null() {
            return Ok(None);
        }
        Ok(Some(decode_hex_result(&result, "state_getStorage")?))
    }

    /// `state_call` — invoke a runtime API.
    ///
    /// # Errors
    /// Transport or decode failure.
    pub fn state_call(&self, method: &str, params: &[u8]) -> Result<Vec<u8>, ChainError> {
        let hex_params = format!("0x{}", hex::encode(params));
        let result = self.rpc("state_call", &json!([method, hex_params]))?;
        decode_hex_result(&result, "state_call")
    }

    /// `system_accountNextIndex`.
    ///
    /// # Errors
    /// Transport or parse failure.
    pub fn system_account_next_index(&self, account_id: [u8; 32]) -> Result<u64, ChainError> {
        let hex_addr = format!("0x{}", hex::encode(account_id));
        let result = self.rpc("system_accountNextIndex", &json!([hex_addr]))?;
        result
            .as_u64()
            .ok_or_else(|| ChainError::Other("accountNextIndex not u64".into()))
    }

    /// `chain_getBlockHash`.
    ///
    /// # Errors
    /// Transport or hex decode failure.
    pub fn chain_get_block_hash(&self, block: u64) -> Result<[u8; 32], ChainError> {
        let hex_n = format!("0x{block:x}");
        let result = self.rpc("chain_getBlockHash", &json!([hex_n]))?;
        let s = result
            .as_str()
            .ok_or_else(|| ChainError::Other("block hash not string".into()))?;
        let s = s.strip_prefix("0x").unwrap_or(s);
        let bytes =
            hex::decode(s).map_err(|e| ChainError::Other(format!("block hash hex: {e}")))?;
        if bytes.len() != 32 {
            return Err(ChainError::Other(format!(
                "expected 32-byte hash, got {} bytes",
                bytes.len()
            )));
        }
        let mut out = [0_u8; 32];
        out.copy_from_slice(&bytes);
        Ok(out)
    }

    /// `chain_getHeader`.
    ///
    /// # Errors
    /// Transport failure.
    pub fn chain_get_header(&self) -> Result<Value, ChainError> {
        self.rpc("chain_getHeader", &json!([]))
    }

    /// `author_submitAndWatchExtrinsic` — returns extrinsic hash or subscription ID.
    ///
    /// # Errors
    /// Transport failure.
    pub fn author_submit_and_watch_extrinsic(
        &self,
        extrinsic: &[u8],
    ) -> Result<String, ChainError> {
        let hex_ext = format!("0x{}", hex::encode(extrinsic));
        let result = self.rpc("author_submitAndWatchExtrinsic", &json!([hex_ext]))?;
        match result {
            Value::String(s) => Ok(s),
            other => Ok(other.to_string()),
        }
    }

    /// `state_getRuntimeVersion`.
    ///
    /// # Errors
    /// Transport or parse failure.
    pub fn state_get_runtime_version(&self) -> Result<RuntimeVersion, ChainError> {
        let v = self.rpc("state_getRuntimeVersion", &json!([]))?;
        let spec_version = json_u32(&v, "specVersion")?;
        let transaction_version = json_u32(&v, "transactionVersion")?;
        let spec_name = v
            .get("specName")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        Ok(RuntimeVersion {
            spec_version,
            transaction_version,
            spec_name,
        })
    }
}

fn http_endpoint(endpoint: &str) -> String {
    if let Some(rest) = endpoint.strip_prefix("wss://") {
        format!("https://{rest}")
    } else if let Some(rest) = endpoint.strip_prefix("ws://") {
        format!("http://{rest}")
    } else {
        endpoint.to_owned()
    }
}

fn json_u32(value: &Value, field: &str) -> Result<u32, ChainError> {
    let n = value
        .get(field)
        .and_then(|v| {
            v.as_u64()
                .or_else(|| v.as_i64().and_then(|i| u64::try_from(i).ok()))
        })
        .ok_or_else(|| ChainError::Other(format!("{field} missing or not integer")))?;
    u32::try_from(n).map_err(|_| ChainError::Other(format!("{field} overflow u32")))
}

fn decode_hex_result(value: &Value, ctx: &str) -> Result<Vec<u8>, ChainError> {
    let s = value
        .as_str()
        .ok_or_else(|| ChainError::Other(format!("{ctx}: expected hex string")))?;
    let s = s.strip_prefix("0x").unwrap_or(s);
    hex::decode(s).map_err(|e| ChainError::Other(format!("{ctx}: hex decode: {e}")))
}
