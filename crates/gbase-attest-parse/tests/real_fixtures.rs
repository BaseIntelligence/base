//! Real Phala TDX fixtures from task 34 (task 35 re-verify).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use gbase_attest_parse::{
    parse_tdx_quote_v4, MIN_QUOTE_LEN, QUOTE_VERSION_V4, REGISTER_LEN, TEE_TYPE_TDX,
};

const QUOTE: &[u8] = include_bytes!("fixtures/real/quote.bin");
const QUOTE_HEX: &str = include_str!("fixtures/real/quote.hex");

/// BASE `phala_quote.parse_td_report` / task-34 captured registers.
const EXPECT_MR_TD: &str =
    "f06dfda6dce1cf904d4e2bab1dc370634cf95cefa2ceb2de2eee127c9382698090d7a4a13e14c536ec6c9c3c8fa87077";
const EXPECT_MR_CONFIG: &str =
    "011b8a63efb0f7afda1e52c823f39cc0a79d6d75c2e7a086b58e0e6a2db548524b000000000000000000000000000000";
const EXPECT_RTMR0: &str =
    "68102e7b524af310f7b7d426ce75481e36c40f5d513a9009c046e9d37e31551f0134d954b496a3357fd61d03f07ffe96";
const EXPECT_RTMR1: &str =
    "07e6f51aa763abfe75c3ddfbf4f425fe3f0ceff66d807a75e049303dce9addf68e7218729bd419638af63a370f65878c";
const EXPECT_RTMR2: &str =
    "df67e467e60edc1737bcf8e682d48131bfb427f523226aa7f197a7608e9b3784783fa759ef5b28191fa12f9ddb36b858";
const EXPECT_RTMR3: &str =
    "9c343d652b721b398b2b1233251831c67162b8e7c1f09956d5162715c5be48f8ae8c5001facd341759c4abc7aff61a0f";

fn hex_decode(hex: &str) -> Vec<u8> {
    let hex = hex.trim();
    assert!(hex.len().is_multiple_of(2));
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("hex"))
        .collect()
}

fn hex_reg(hex: &str) -> [u8; REGISTER_LEN] {
    let v = hex_decode(hex);
    assert_eq!(v.len(), REGISTER_LEN);
    let mut out = [0_u8; REGISTER_LEN];
    out.copy_from_slice(&v);
    out
}

#[test]
fn real_quote_bin_parses_like_phala_quote_py() {
    assert!(QUOTE.len() >= MIN_QUOTE_LEN);
    let parsed = parse_tdx_quote_v4(QUOTE).expect("parse real quote");
    assert_eq!(parsed.header.version, QUOTE_VERSION_V4);
    assert_eq!(parsed.header.tee_type, TEE_TYPE_TDX);
    assert_eq!(parsed.td_report.mr_td, hex_reg(EXPECT_MR_TD));
    assert_eq!(parsed.td_report.mr_config_id, hex_reg(EXPECT_MR_CONFIG));
    assert_eq!(parsed.td_report.rtmr0, hex_reg(EXPECT_RTMR0));
    assert_eq!(parsed.td_report.rtmr1, hex_reg(EXPECT_RTMR1));
    assert_eq!(parsed.td_report.rtmr2, hex_reg(EXPECT_RTMR2));
    assert_eq!(parsed.td_report.rtmr3, hex_reg(EXPECT_RTMR3));
}

#[test]
fn real_quote_hex_matches_bin() {
    let from_hex = hex_decode(QUOTE_HEX);
    assert_eq!(from_hex.as_slice(), QUOTE);
    let a = parse_tdx_quote_v4(QUOTE).expect("bin");
    let b = parse_tdx_quote_v4(&from_hex).expect("hex");
    assert_eq!(a, b);
}
