//! Structural parser for Intel TDX ECDSA quote **v4**.
//!
//! Layout matches BASE `phala_quote.py` and the Intel TDREPORT10 body that
//! follows the 48-byte quote header. Cryptographic verification is out of
//! scope (see `gbase-attest-policy`).
//!
//! Offsets (absolute, bytes from start of quote):
//! - header: `0..48`
//! - `mr_td`: `184..232`
//! - `mr_config_id`: `232..280`
//! - `rtmr0..3`: `376`, `424`, `472`, `520` (each 48 bytes)
//! - `report_data`: `568..632`
//!
//! Minimum length to contain a full TD report body through `report_data`: **632**.

#![forbid(unsafe_code)]

use thiserror::Error;

/// Quote header length preceding the TD report body.
pub const QUOTE_HEADER_LEN: usize = 48;
/// SHA-384 measurement register width.
pub const REGISTER_LEN: usize = 48;
/// TDX `report_data` field width.
pub const REPORT_DATA_LEN: usize = 64;

/// Absolute offset of `MRTD` within a v4 quote.
pub const MR_TD_OFFSET: usize = QUOTE_HEADER_LEN + 136;
/// Absolute offset of `MRCONFIGID` within a v4 quote.
pub const MR_CONFIG_ID_OFFSET: usize = QUOTE_HEADER_LEN + 184;
/// Absolute offset of `RTMR0`.
pub const RTMR0_OFFSET: usize = QUOTE_HEADER_LEN + 328;
/// Absolute offset of `RTMR1`.
pub const RTMR1_OFFSET: usize = QUOTE_HEADER_LEN + 376;
/// Absolute offset of `RTMR2`.
pub const RTMR2_OFFSET: usize = QUOTE_HEADER_LEN + 424;
/// Absolute offset of `RTMR3`.
pub const RTMR3_OFFSET: usize = QUOTE_HEADER_LEN + 472;
/// Absolute offset of `REPORTDATA`.
pub const REPORT_DATA_OFFSET: usize = QUOTE_HEADER_LEN + 520;
/// Minimum quote length covering header + TD report through `report_data`.
pub const MIN_QUOTE_LEN: usize = REPORT_DATA_OFFSET + REPORT_DATA_LEN;

/// Expected quote version for this parser (Intel TDX quote v4).
pub const QUOTE_VERSION_V4: u16 = 4;
/// TEE type value for Intel TDX in the quote header (`tee_type` LE u32).
pub const TEE_TYPE_TDX: u32 = 0x0000_0081;

/// Typed parse failures. Library code never panics on malformed input.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// Buffer shorter than the bytes needed for the requested field.
    #[error("quote too short: have {have} bytes, need at least {need}")]
    Truncated {
        /// Observed length.
        have: usize,
        /// Required length.
        need: usize,
    },
    /// Header `version` is not v4.
    #[error("unsupported quote version {got}, expected {QUOTE_VERSION_V4}")]
    UnsupportedVersion {
        /// Version field from the header.
        got: u16,
    },
    /// Header `tee_type` is not TDX.
    #[error("unsupported tee_type {got:#010x}, expected TDX ({TEE_TYPE_TDX:#010x})")]
    UnsupportedTeeType {
        /// TEE type field from the header.
        got: u32,
    },
}

/// 48-byte Intel TDX/SGX quote header (v3/v4 layout).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuoteHeader {
    /// Quote format version (must be 4 for this crate).
    pub version: u16,
    /// Attestation key type.
    pub att_key_type: u16,
    /// TEE type (`TEE_TYPE_TDX` for TDX).
    pub tee_type: u32,
    /// Reserved / QE SVN packing (opaque to structural parse).
    pub reserved: u32,
    /// Vendor ID (16 bytes).
    pub vendor_id: [u8; 16],
    /// User data (20 bytes).
    pub user_data: [u8; 20],
}

/// Measurement registers + `report_data` from the TD report body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TdReport {
    /// MRTD (SHA-384).
    pub mr_td: [u8; REGISTER_LEN],
    /// MRCONFIGID (SHA-384 / compose binding prefix region).
    pub mr_config_id: [u8; REGISTER_LEN],
    /// RTMR0.
    pub rtmr0: [u8; REGISTER_LEN],
    /// RTMR1.
    pub rtmr1: [u8; REGISTER_LEN],
    /// RTMR2.
    pub rtmr2: [u8; REGISTER_LEN],
    /// RTMR3.
    pub rtmr3: [u8; REGISTER_LEN],
    /// 64-byte REPORTDATA.
    pub report_data: [u8; REPORT_DATA_LEN],
}

/// Fully parsed v4 TDX quote (header + TD report fields of interest).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TdxQuoteV4 {
    /// Quote header.
    pub header: QuoteHeader,
    /// TD report measurements.
    pub td_report: TdReport,
}

/// Parse a TDX quote v4 buffer: header + TD report measurement fields.
///
/// Never panics on attacker-controlled input.
///
/// # Errors
///
/// Returns [`ParseError`] on truncation or unsupported header values.
pub fn parse_tdx_quote_v4(quote: &[u8]) -> Result<TdxQuoteV4, ParseError> {
    let header = parse_quote_header(quote)?;
    let td_report = parse_td_report(quote)?;
    Ok(TdxQuoteV4 { header, td_report })
}

/// Parse only the 48-byte quote header (validates version + `tee_type`).
///
/// # Errors
///
/// Returns [`ParseError::Truncated`] if fewer than [`QUOTE_HEADER_LEN`] bytes
/// are present, [`ParseError::UnsupportedVersion`] if version is not v4, or
/// [`ParseError::UnsupportedTeeType`] if `tee_type` is not TDX.
pub fn parse_quote_header(quote: &[u8]) -> Result<QuoteHeader, ParseError> {
    if quote.len() < QUOTE_HEADER_LEN {
        return Err(ParseError::Truncated {
            have: quote.len(),
            need: QUOTE_HEADER_LEN,
        });
    }
    let version = u16::from_le_bytes([quote[0], quote[1]]);
    if version != QUOTE_VERSION_V4 {
        return Err(ParseError::UnsupportedVersion { got: version });
    }
    let att_key_type = u16::from_le_bytes([quote[2], quote[3]]);
    let tee_type = u32::from_le_bytes([quote[4], quote[5], quote[6], quote[7]]);
    if tee_type != TEE_TYPE_TDX {
        return Err(ParseError::UnsupportedTeeType { got: tee_type });
    }
    let reserved = u32::from_le_bytes([quote[8], quote[9], quote[10], quote[11]]);
    let mut vendor_id = [0u8; 16];
    vendor_id.copy_from_slice(&quote[12..28]);
    let mut user_data = [0u8; 20];
    user_data.copy_from_slice(&quote[28..48]);
    Ok(QuoteHeader {
        version,
        att_key_type,
        tee_type,
        reserved,
        vendor_id,
        user_data,
    })
}

/// Parse TD report measurement fields from a full quote buffer.
///
/// Does **not** re-check header version (use [`parse_tdx_quote_v4`] for that).
/// Fails closed if the buffer is shorter than [`MIN_QUOTE_LEN`].
///
/// # Errors
///
/// Returns [`ParseError::Truncated`] when `quote.len() < MIN_QUOTE_LEN`.
pub fn parse_td_report(quote: &[u8]) -> Result<TdReport, ParseError> {
    if quote.len() < MIN_QUOTE_LEN {
        return Err(ParseError::Truncated {
            have: quote.len(),
            need: MIN_QUOTE_LEN,
        });
    }
    Ok(TdReport {
        mr_td: copy_reg(quote, MR_TD_OFFSET),
        mr_config_id: copy_reg(quote, MR_CONFIG_ID_OFFSET),
        rtmr0: copy_reg(quote, RTMR0_OFFSET),
        rtmr1: copy_reg(quote, RTMR1_OFFSET),
        rtmr2: copy_reg(quote, RTMR2_OFFSET),
        rtmr3: copy_reg(quote, RTMR3_OFFSET),
        report_data: copy_report_data(quote),
    })
}

fn copy_reg(quote: &[u8], offset: usize) -> [u8; REGISTER_LEN] {
    let mut out = [0u8; REGISTER_LEN];
    out.copy_from_slice(&quote[offset..offset + REGISTER_LEN]);
    out
}

fn copy_report_data(quote: &[u8]) -> [u8; REPORT_DATA_LEN] {
    let mut out = [0u8; REPORT_DATA_LEN];
    out.copy_from_slice(&quote[REPORT_DATA_OFFSET..REPORT_DATA_OFFSET + REPORT_DATA_LEN]);
    out
}

/// Overwrite `report_data` in a mutable quote buffer (fixture / D10 bind path).
///
/// # Errors
///
/// Returns [`ParseError::Truncated`] when the buffer is shorter than [`MIN_QUOTE_LEN`].
pub fn patch_report_data(
    quote: &mut [u8],
    report_data: &[u8; REPORT_DATA_LEN],
) -> Result<(), ParseError> {
    if quote.len() < MIN_QUOTE_LEN {
        return Err(ParseError::Truncated {
            have: quote.len(),
            need: MIN_QUOTE_LEN,
        });
    }
    quote[REPORT_DATA_OFFSET..REPORT_DATA_OFFSET + REPORT_DATA_LEN].copy_from_slice(report_data);
    Ok(())
}

/// Build a minimal synthetic v4 TDX quote for tests and fixtures.
///
/// Places each register and `report_data` at the fixed Intel/BASE offsets.
/// Header defaults to version=4, `tee_type` = TDX, zeros elsewhere unless overridden.
#[must_use]
pub fn build_synthetic_quote_v4(fields: SyntheticQuoteFields<'_>) -> Vec<u8> {
    let mut buf = vec![0u8; MIN_QUOTE_LEN];
    // version = 4 LE
    buf[0] = 4;
    buf[1] = 0;
    // att_key_type
    let akt = fields.att_key_type.to_le_bytes();
    buf[2] = akt[0];
    buf[3] = akt[1];
    // tee_type = TDX
    let tt = TEE_TYPE_TDX.to_le_bytes();
    buf[4..8].copy_from_slice(&tt);
    if let Some(vid) = fields.vendor_id {
        buf[12..28].copy_from_slice(vid);
    }
    if let Some(ud) = fields.user_data {
        buf[28..48].copy_from_slice(ud);
    }
    write_reg(&mut buf, MR_TD_OFFSET, fields.mr_td);
    write_reg(&mut buf, MR_CONFIG_ID_OFFSET, fields.mr_config_id);
    write_reg(&mut buf, RTMR0_OFFSET, fields.rtmr0);
    write_reg(&mut buf, RTMR1_OFFSET, fields.rtmr1);
    write_reg(&mut buf, RTMR2_OFFSET, fields.rtmr2);
    write_reg(&mut buf, RTMR3_OFFSET, fields.rtmr3);
    let rd = fields.report_data;
    let n = rd.len().min(REPORT_DATA_LEN);
    buf[REPORT_DATA_OFFSET..REPORT_DATA_OFFSET + n].copy_from_slice(&rd[..n]);
    if let Some(tail) = fields.tail {
        buf.extend_from_slice(tail);
    }
    buf
}

/// Field bundle for [`build_synthetic_quote_v4`].
#[derive(Debug, Clone, Copy)]
pub struct SyntheticQuoteFields<'a> {
    /// Optional `att_key_type` (default 2 = ECDSA P-256, commonly used).
    pub att_key_type: u16,
    /// Optional 16-byte vendor id.
    pub vendor_id: Option<&'a [u8; 16]>,
    /// Optional 20-byte user data.
    pub user_data: Option<&'a [u8; 20]>,
    /// MRTD (48 bytes).
    pub mr_td: &'a [u8; REGISTER_LEN],
    /// MRCONFIGID (48 bytes).
    pub mr_config_id: &'a [u8; REGISTER_LEN],
    /// RTMR0.
    pub rtmr0: &'a [u8; REGISTER_LEN],
    /// RTMR1.
    pub rtmr1: &'a [u8; REGISTER_LEN],
    /// RTMR2.
    pub rtmr2: &'a [u8; REGISTER_LEN],
    /// RTMR3.
    pub rtmr3: &'a [u8; REGISTER_LEN],
    /// REPORTDATA (up to 64 bytes; shorter values are zero-padded).
    pub report_data: &'a [u8],
    /// Optional trailing signature/cert bytes (opaque).
    pub tail: Option<&'a [u8]>,
}

fn write_reg(buf: &mut [u8], offset: usize, reg: &[u8; REGISTER_LEN]) {
    buf[offset..offset + REGISTER_LEN].copy_from_slice(reg);
}

#[cfg(test)]
mod tests {
    use super::*;

    const MR_TD: [u8; REGISTER_LEN] = [0xa1; REGISTER_LEN];
    const MR_CONFIG: [u8; REGISTER_LEN] = [0xc1; REGISTER_LEN];
    const RTMR0: [u8; REGISTER_LEN] = [0xb0; REGISTER_LEN];
    const RTMR1: [u8; REGISTER_LEN] = [0xb1; REGISTER_LEN];
    const RTMR2: [u8; REGISTER_LEN] = [0xb2; REGISTER_LEN];
    const RTMR3: [u8; REGISTER_LEN] = [0xb3; REGISTER_LEN];
    const REPORT_DATA_HALF: [u8; 32] = [0xff; 32];

    fn default_fields() -> SyntheticQuoteFields<'static> {
        SyntheticQuoteFields {
            att_key_type: 2,
            vendor_id: None,
            user_data: None,
            mr_td: &MR_TD,
            mr_config_id: &MR_CONFIG,
            rtmr0: &RTMR0,
            rtmr1: &RTMR1,
            rtmr2: &RTMR2,
            rtmr3: &RTMR3,
            report_data: &REPORT_DATA_HALF,
            tail: None,
        }
    }

    #[test]
    fn parse_synthetic_round_trips_registers() {
        let quote = build_synthetic_quote_v4(default_fields());
        assert_eq!(quote.len(), MIN_QUOTE_LEN);
        let parsed = parse_tdx_quote_v4(&quote).expect("parse");
        assert_eq!(parsed.header.version, QUOTE_VERSION_V4);
        assert_eq!(parsed.header.tee_type, TEE_TYPE_TDX);
        assert_eq!(parsed.header.att_key_type, 2);
        assert_eq!(parsed.td_report.mr_td, MR_TD);
        assert_eq!(parsed.td_report.mr_config_id, MR_CONFIG);
        assert_eq!(parsed.td_report.rtmr0, RTMR0);
        assert_eq!(parsed.td_report.rtmr1, RTMR1);
        assert_eq!(parsed.td_report.rtmr2, RTMR2);
        assert_eq!(parsed.td_report.rtmr3, RTMR3);
        let mut expected_rd = [0u8; REPORT_DATA_LEN];
        expected_rd[..32].fill(0xff);
        assert_eq!(parsed.td_report.report_data, expected_rd);
    }

    #[test]
    fn parse_preserves_header_vendor_and_user_data() {
        let vendor = [0x11u8; 16];
        let user = [0x22u8; 20];
        let mut fields = default_fields();
        fields.vendor_id = Some(&vendor);
        fields.user_data = Some(&user);
        let quote = build_synthetic_quote_v4(fields);
        let h = parse_quote_header(&quote).expect("header");
        assert_eq!(h.vendor_id, vendor);
        assert_eq!(h.user_data, user);
    }

    #[test]
    fn parse_rejects_empty() {
        let err = parse_tdx_quote_v4(&[]).expect_err("empty");
        assert!(matches!(err, ParseError::Truncated { have: 0, .. }));
    }

    #[test]
    fn parse_rejects_wrong_version() {
        let mut quote = build_synthetic_quote_v4(default_fields());
        quote[0] = 3;
        let err = parse_tdx_quote_v4(&quote).expect_err("v3");
        assert_eq!(err, ParseError::UnsupportedVersion { got: 3 });
    }

    #[test]
    fn parse_rejects_wrong_tee_type() {
        let mut quote = build_synthetic_quote_v4(default_fields());
        // SGX tee_type = 0
        quote[4..8].copy_from_slice(&0u32.to_le_bytes());
        let err = parse_tdx_quote_v4(&quote).expect_err("sgx");
        assert_eq!(err, ParseError::UnsupportedTeeType { got: 0 });
    }

    #[test]
    fn truncation_every_16_byte_boundary_returns_err_without_panic() {
        let full = build_synthetic_quote_v4(default_fields());
        assert_eq!(full.len(), MIN_QUOTE_LEN);
        let mut len = 0usize;
        while len < MIN_QUOTE_LEN {
            let slice = &full[..len];
            let result = std::panic::catch_unwind(|| parse_tdx_quote_v4(slice));
            let outcome = result.expect("parser must not panic on truncated input");
            assert!(
                outcome.is_err(),
                "expected Err at truncated len={len}, got Ok"
            );
            len = len.saturating_add(16);
        }
        // Also hit the exact min-1 boundary.
        let almost = &full[..MIN_QUOTE_LEN - 1];
        assert!(parse_tdx_quote_v4(almost).is_err());
        assert!(parse_tdx_quote_v4(&full).is_ok());
    }

    #[test]
    fn offsets_match_base_phala_quote_layout() {
        assert_eq!(QUOTE_HEADER_LEN, 48);
        assert_eq!(REGISTER_LEN, 48);
        assert_eq!(REPORT_DATA_LEN, 64);
        assert_eq!(MR_TD_OFFSET, 48 + 136);
        assert_eq!(MR_CONFIG_ID_OFFSET, 48 + 184);
        assert_eq!(RTMR0_OFFSET, 48 + 328);
        assert_eq!(RTMR1_OFFSET, 48 + 376);
        assert_eq!(RTMR2_OFFSET, 48 + 424);
        assert_eq!(RTMR3_OFFSET, 48 + 472);
        assert_eq!(REPORT_DATA_OFFSET, 48 + 520);
        assert_eq!(MIN_QUOTE_LEN, 632);
    }

    #[test]
    fn short_fuzz_random_prefixes_never_panic() {
        // Lightweight stand-in for cargo-fuzz: many random lengths + patterns.
        let full = build_synthetic_quote_v4(default_fields());
        let patterns: &[&[u8]] = &[
            &[],
            &[0xff; 1],
            &[0x00; 47],
            &[0x00; 48],
            &[0x04, 0x00],
            full.as_slice(),
        ];
        for p in patterns {
            std::panic::catch_unwind(|| {
                let _ = parse_tdx_quote_v4(p);
            })
            .expect("no panic");
        }
        // Walk every length 0..=MIN_QUOTE_LEN+32
        for len in 0..=(MIN_QUOTE_LEN + 32) {
            let mut buf = vec![0u8; len];
            if len >= 2 {
                buf[0] = 4;
            }
            if len >= 8 {
                buf[4..8].copy_from_slice(&TEE_TYPE_TDX.to_le_bytes());
            }
            if len >= MIN_QUOTE_LEN {
                buf[..MIN_QUOTE_LEN].copy_from_slice(&full);
                if len > MIN_QUOTE_LEN {
                    buf[MIN_QUOTE_LEN..].fill(0xaa);
                }
            }
            let r = std::panic::catch_unwind(|| parse_tdx_quote_v4(&buf));
            assert!(r.is_ok(), "panic at len={len}");
        }
    }

    #[test]
    fn tail_bytes_ignored_after_report_data() {
        let mut fields = default_fields();
        let tail = [0x09u8, 0x09, 0x09];
        fields.tail = Some(&tail);
        let quote = build_synthetic_quote_v4(fields);
        assert!(quote.ends_with(&tail));
        let parsed = parse_tdx_quote_v4(&quote).expect("parse with tail");
        assert_eq!(parsed.td_report.mr_td, MR_TD);
    }
}
