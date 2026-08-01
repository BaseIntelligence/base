//! Test-only support: an order-preserving JSON reader plus the vector case runner.
//!
//! `serde_json`'s default `Map` is a `BTreeMap`, which would destroy the Python `dict`
//! insertion order these vectors encode, and turning on its `preserve_order` feature
//! would leak into every other crate in the workspace build. A ~150-line reader here
//! (test-only, excluded from the LOC cap) keeps order fidelity with zero new deps.

#![allow(dead_code, clippy::unwrap_used, clippy::expect_used)]

use aggregate::python::{ChallengeWeightsResult, FinalWeights};

// --- minimal order-preserving JSON ------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn obj(&self) -> &[(String, Json)] {
        match self {
            Json::Obj(entries) => entries,
            _ => panic!("expected object, got {self:?}"),
        }
    }

    pub fn arr(&self) -> &[Json] {
        match self {
            Json::Arr(items) => items,
            _ => panic!("expected array, got {self:?}"),
        }
    }

    pub fn num(&self) -> f64 {
        match self {
            Json::Num(n) => *n,
            _ => panic!("expected number, got {self:?}"),
        }
    }

    pub fn str(&self) -> &str {
        match self {
            Json::Str(s) => s,
            _ => panic!("expected string, got {self:?}"),
        }
    }

    pub fn boolean(&self) -> bool {
        match self {
            Json::Bool(b) => *b,
            _ => panic!("expected bool, got {self:?}"),
        }
    }
}

pub fn parse_json(text: &str) -> Json {
    let bytes: Vec<char> = text.chars().collect();
    let mut pos = 0usize;
    let value = parse_value(&bytes, &mut pos);
    skip_ws(&bytes, &mut pos);
    assert!(pos >= bytes.len(), "trailing JSON input at {pos}");
    value
}

fn skip_ws(b: &[char], pos: &mut usize) {
    while let Some(c) = b.get(*pos) {
        if c.is_ascii_whitespace() {
            *pos += 1;
        } else {
            break;
        }
    }
}

fn literal(b: &[char], pos: &mut usize, word: &str) -> bool {
    let chars: Vec<char> = word.chars().collect();
    if b.len() >= *pos + chars.len() && b[*pos..*pos + chars.len()] == chars[..] {
        *pos += chars.len();
        return true;
    }
    false
}

fn parse_value(b: &[char], pos: &mut usize) -> Json {
    skip_ws(b, pos);
    match b.get(*pos) {
        Some('{') => parse_object(b, pos),
        Some('[') => parse_array(b, pos),
        Some('"') => Json::Str(parse_string(b, pos)),
        Some('t') if literal(b, pos, "true") => Json::Bool(true),
        Some('f') if literal(b, pos, "false") => Json::Bool(false),
        Some('n') if literal(b, pos, "null") => Json::Null,
        // Python's json module emits these bare tokens for non-finite floats.
        Some('N') if literal(b, pos, "NaN") => Json::Num(f64::NAN),
        Some('I') if literal(b, pos, "Infinity") => Json::Num(f64::INFINITY),
        Some('-') if literal(b, pos, "-Infinity") => Json::Num(f64::NEG_INFINITY),
        Some(_) => parse_number(b, pos),
        None => panic!("unexpected end of JSON"),
    }
}

fn parse_object(b: &[char], pos: &mut usize) -> Json {
    *pos += 1; // '{'
    let mut entries = Vec::new();
    skip_ws(b, pos);
    if b.get(*pos) == Some(&'}') {
        *pos += 1;
        return Json::Obj(entries);
    }
    loop {
        skip_ws(b, pos);
        let key = parse_string(b, pos);
        skip_ws(b, pos);
        assert_eq!(b.get(*pos), Some(&':'), "expected ':' at {pos}");
        *pos += 1;
        let value = parse_value(b, pos);
        entries.push((key, value));
        skip_ws(b, pos);
        match b.get(*pos) {
            Some(',') => *pos += 1,
            Some('}') => {
                *pos += 1;
                return Json::Obj(entries);
            }
            other => panic!("expected ',' or '}}' at {pos}, got {other:?}"),
        }
    }
}

fn parse_array(b: &[char], pos: &mut usize) -> Json {
    *pos += 1; // '['
    let mut items = Vec::new();
    skip_ws(b, pos);
    if b.get(*pos) == Some(&']') {
        *pos += 1;
        return Json::Arr(items);
    }
    loop {
        items.push(parse_value(b, pos));
        skip_ws(b, pos);
        match b.get(*pos) {
            Some(',') => *pos += 1,
            Some(']') => {
                *pos += 1;
                return Json::Arr(items);
            }
            other => panic!("expected ',' or ']' at {pos}, got {other:?}"),
        }
    }
}

fn parse_string(b: &[char], pos: &mut usize) -> String {
    assert_eq!(b.get(*pos), Some(&'"'), "expected string at {pos}");
    *pos += 1;
    let mut out = String::new();
    loop {
        let c = *b.get(*pos).expect("unterminated string");
        *pos += 1;
        match c {
            '"' => return out,
            '\\' => {
                let esc = *b.get(*pos).expect("unterminated escape");
                *pos += 1;
                match esc {
                    'n' => out.push('\n'),
                    't' => out.push('\t'),
                    'r' => out.push('\r'),
                    'b' => out.push('\u{8}'),
                    'f' => out.push('\u{c}'),
                    'u' => {
                        let hex: String = b[*pos..*pos + 4].iter().collect();
                        *pos += 4;
                        let code = u32::from_str_radix(&hex, 16).expect("bad \\u escape");
                        out.push(char::from_u32(code).unwrap_or('\u{fffd}'));
                    }
                    other => out.push(other),
                }
            }
            other => out.push(other),
        }
    }
}

fn parse_number(b: &[char], pos: &mut usize) -> Json {
    let start = *pos;
    while let Some(c) = b.get(*pos) {
        if c.is_ascii_digit() || matches!(c, '-' | '+' | '.' | 'e' | 'E') {
            *pos += 1;
        } else {
            break;
        }
    }
    let text: String = b[start..*pos].iter().collect();
    // Rust's f64 parser is correctly rounded, so Python's repr() round-trips exactly.
    Json::Num(
        text.parse::<f64>()
            .unwrap_or_else(|_| panic!("bad number {text:?}")),
    )
}

// --- vector case decoding ---------------------------------------------------------

pub struct Case {
    pub name: String,
    pub results: Vec<ChallengeWeightsResult>,
    pub hotkey_to_uid: Vec<(String, u16)>,
    pub min_allowed_weights: u32,
    pub max_weight_limit: u16,
    /// Present unless the case expects a `ZeroMinerWeightError`.
    pub expected: Option<FinalWeights>,
    /// Expected `[[uid, weight_u16], ...]`.
    pub expected_vector: Option<Vec<(u16, u16)>>,
    /// Expected `ZeroMinerWeightError` message, verbatim from Python.
    pub expected_error: Option<String>,
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn as_u16(v: f64) -> u16 {
    v as u16
}

pub fn decode_case(name: &str, root: &Json) -> Case {
    let inputs = root.get("inputs").expect("inputs");

    let results = inputs
        .get("challenge_results")
        .expect("challenge_results")
        .arr()
        .iter()
        .map(|r| ChallengeWeightsResult {
            slug: r.get("slug").expect("slug").str().to_owned(),
            emission_percent: r.get("emission_percent").expect("emission_percent").num(),
            weights: r
                .get("weights")
                .map(|w| {
                    w.obj()
                        .iter()
                        .map(|(k, v)| (k.clone(), v.num()))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default(),
            ok: r.get("ok").is_none_or(Json::boolean),
            error: r.get("error").and_then(|e| match e {
                Json::Str(s) => Some(s.clone()),
                _ => None,
            }),
        })
        .collect();

    let hotkey_to_uid = inputs
        .get("hotkey_to_uid")
        .expect("hotkey_to_uid")
        .obj()
        .iter()
        .map(|(k, v)| (k.clone(), as_u16(v.num())))
        .collect();

    let kwargs = inputs
        .get("kwargs")
        .cloned()
        .unwrap_or(Json::Obj(Vec::new()));
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let min_allowed_weights = kwargs
        .get("min_allowed_weights")
        .map_or(1_u32, |v| v.num() as u32);
    let max_weight_limit = kwargs
        .get("max_weight_limit")
        .map_or(65_535_u16, |v| as_u16(v.num()));

    let expected = root.get("python_float_output").map(|out| FinalWeights {
        uids: out
            .get("uids")
            .expect("uids")
            .arr()
            .iter()
            .map(|v| as_u16(v.num()))
            .collect(),
        weights: out
            .get("weights")
            .expect("weights")
            .arr()
            .iter()
            .map(Json::num)
            .collect(),
        hotkey_weights: out
            .get("hotkey_weights")
            .expect("hotkey_weights")
            .obj()
            .iter()
            .map(|(k, v)| (k.clone(), v.num()))
            .collect(),
    });

    let expected_vector = root.get("expected_vector").map(|v| {
        v.arr()
            .iter()
            .map(|pair| {
                let p = pair.arr();
                (as_u16(p[0].num()), as_u16(p[1].num()))
            })
            .collect()
    });

    let expected_error = root
        .get("python_error")
        .and_then(|e| match e {
            Json::Str(s) => Some(s.clone()),
            _ => None,
        })
        .or_else(|| {
            root.get("header")
                .and_then(|h| h.get("python_error"))
                .and_then(|e| match e {
                    Json::Str(s) => Some(s.clone()),
                    _ => None,
                })
        });

    Case {
        name: name.to_owned(),
        results,
        hotkey_to_uid,
        min_allowed_weights,
        max_weight_limit,
        expected,
        expected_vector,
        expected_error,
    }
}

/// Assert bit-exact `f64` equality, reporting the raw bit patterns on failure.
pub fn assert_bits_eq(context: &str, got: &[f64], want: &[f64]) {
    assert_eq!(
        got.len(),
        want.len(),
        "{context}: length {got:?} vs {want:?}"
    );
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{context}[{i}]: rust {g:?} (0x{:016x}) != python {w:?} (0x{:016x})",
            g.to_bits(),
            w.to_bits()
        );
    }
}
