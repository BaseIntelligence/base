//! Intake header → payer creds (Lium XOR Verda unless `X-Compute-Provider`).

use serde_json::Value;

use super::ProviderCreds;

/// `X-Compute-Provider` when both Lium and Verda headers are complete.
pub const COMPUTE_PROVIDER_HEADER: &str = "x-compute-provider";
/// Verda OAuth client id.
pub const VERDA_CLIENT_ID_HEADER: &str = "x-verda-client-id";
/// Verda OAuth client secret.
pub const VERDA_CLIENT_SECRET_HEADER: &str = "x-verda-client-secret";
/// Verda inference / tasks token.
pub const VERDA_INFERENCE_KEY_HEADER: &str = "x-verda-inference-key";
/// Alias for the inference token (`X-Verda-Api-Key`).
pub const VERDA_API_KEY_HEADER: &str = "x-verda-api-key";

/// Intake payer failure (HTTP 400).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntakePayerError {
    /// Both providers complete, no `X-Compute-Provider`.
    Ambiguous,
    /// Partial Verda triplet.
    MissingVerda,
}

impl IntakePayerError {
    /// Stable API error code.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::Ambiguous => "ambiguous_compute_provider",
            Self::MissingVerda => "missing_verda_credentials",
        }
    }

    /// Miner-facing message.
    #[must_use]
    pub const fn message(&self) -> &'static str {
        match self {
            Self::Ambiguous => {
                "both Lium and Verda credentials present — set X-Compute-Provider: lium or verda"
            }
            Self::MissingVerda => {
                "Verda BYOK needs X-Verda-Client-Id, X-Verda-Client-Secret, and X-Verda-Inference-Key"
            }
        }
    }
}

fn nz(s: Option<&str>) -> Option<&str> {
    s.map(str::trim).filter(|t| !t.is_empty())
}

fn header_str<'a>(headers: &'a http::HeaderMap, name: &str) -> Option<&'a str> {
    headers.get(name).and_then(|v| v.to_str().ok())
}

/// Resolve miner payer from HTTP headers (case-insensitive names).
///
/// # Errors
/// Ambiguous dual complete creds, or incomplete Verda triplet.
pub fn creds_from_headers(
    headers: &http::HeaderMap,
) -> Result<Option<ProviderCreds>, IntakePayerError> {
    let inference = header_str(headers, VERDA_INFERENCE_KEY_HEADER)
        .or_else(|| header_str(headers, VERDA_API_KEY_HEADER))
        .or_else(|| header_str(headers, "X-Verda-Inference-Key"))
        .or_else(|| header_str(headers, "X-Verda-Api-Key"));
    creds_from_parts(
        header_str(headers, COMPUTE_PROVIDER_HEADER)
            .or_else(|| header_str(headers, "X-Compute-Provider")),
        header_str(headers, super::LIUM_API_KEY_HEADER)
            .or_else(|| header_str(headers, "X-Lium-Api-Key")),
        header_str(headers, VERDA_CLIENT_ID_HEADER)
            .or_else(|| header_str(headers, "X-Verda-Client-Id")),
        header_str(headers, VERDA_CLIENT_SECRET_HEADER)
            .or_else(|| header_str(headers, "X-Verda-Client-Secret")),
        inference,
    )
}

/// Resolve miner payer from already-extracted header values.
///
/// # Errors
/// Ambiguous dual complete creds, or incomplete Verda triplet.
pub fn creds_from_parts(
    provider: Option<&str>,
    lium: Option<&str>,
    verda_id: Option<&str>,
    verda_sec: Option<&str>,
    verda_inf: Option<&str>,
) -> Result<Option<ProviderCreds>, IntakePayerError> {
    let lium = nz(lium).and_then(super::normalize_lium_api_key);
    let id = nz(verda_id);
    let sec = nz(verda_sec);
    let inf = nz(verda_inf);
    let verda_any = id.is_some() || sec.is_some() || inf.is_some();
    let verda = match (id, sec, inf) {
        (Some(id), Some(sec), Some(inf)) => Some(ProviderCreds::Verda {
            id: id.to_owned(),
            sec: sec.to_owned(),
            inf: inf.to_owned(),
        }),
        _ if verda_any => return Err(IntakePayerError::MissingVerda),
        _ => None,
    };
    let want = nz(provider).map(str::to_ascii_lowercase);
    if want.as_deref() == Some("lium") {
        if let Some(k) = lium {
            return Ok(Some(ProviderCreds::Lium(k)));
        }
    }
    if want.as_deref() == Some("verda") {
        return verda.ok_or(IntakePayerError::MissingVerda).map(Some);
    }
    match (lium, verda) {
        (Some(k), None) => Ok(Some(ProviderCreds::Lium(k))),
        (None, Some(v)) => Ok(Some(v)),
        (None, None) => Ok(None),
        (Some(_), Some(_)) => Err(IntakePayerError::Ambiguous),
    }
}

/// JSON keys miners must not use to override the operator image/cmd.
#[must_use]
pub fn miner_image_override_error(v: &Value) -> Option<&'static str> {
    const BAD: &[&str] = &[
        "image",
        "docker_image",
        "image_digest",
        "cmd",
        "command",
        "entrypoint",
        "template",
        "template_id",
    ];
    let obj = v.as_object()?;
    BAD.iter()
        .find(|k| obj.contains_key(**k))
        .map(|_| "miners cannot set image, template, cmd, or entrypoint — operator pin only")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lium_or_verda_or_ambiguous() {
        let l = creds_from_parts(None, Some("sk"), None, None, None).unwrap();
        assert_eq!(l.unwrap().provider(), "lium");
        let v = creds_from_parts(None, None, Some("id"), Some("sec"), Some("inf")).unwrap();
        assert_eq!(v.unwrap().provider(), "verda");
        assert_eq!(
            creds_from_parts(None, Some("sk"), Some("id"), Some("sec"), Some("inf")),
            Err(IntakePayerError::Ambiguous)
        );
        let forced = creds_from_parts(
            Some("verda"),
            Some("sk"),
            Some("id"),
            Some("sec"),
            Some("inf"),
        )
        .unwrap();
        assert_eq!(forced.unwrap().provider(), "verda");
        assert_eq!(
            creds_from_parts(None, None, Some("id"), None, Some("inf")),
            Err(IntakePayerError::MissingVerda)
        );
    }

    #[test]
    fn reject_miner_image_fields() {
        let v = serde_json::json!({"image": "evil:latest", "miner_hotkey": "aa"});
        assert!(miner_image_override_error(&v).is_some());
        let ok = serde_json::json!({"miner_hotkey": "aa", "zip_base64": "e30="});
        assert!(miner_image_override_error(&ok).is_none());
    }
}
