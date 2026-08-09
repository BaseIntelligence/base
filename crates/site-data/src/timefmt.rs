//! UTC timestamp formatting for the site contract (no chrono dependency).

/// ISO-8601 from unix millis (UTC, second precision).
#[must_use]
pub fn ms_to_iso(ms: u64) -> String {
    let secs = ms / 1000;
    let (year, month, day, hour, minute, second) = civil_parts(secs);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.000Z")
}

/// `HH:MM:SS` UTC from millis.
#[must_use]
pub fn ms_to_clock(ms: u64) -> String {
    let secs = ms / 1000;
    let (_, _, _, hour, minute, second) = civil_parts(secs);
    format!("{hour:02}:{minute:02}:{second:02}")
}

#[allow(clippy::many_single_char_names)]
fn civil_parts(secs: u64) -> (i64, u32, u32, u32, u32, u32) {
    let days = i64::try_from(secs / 86_400).unwrap_or(0);
    let rem = secs % 86_400;
    let hour = u32::try_from(rem / 3600).unwrap_or(0);
    let minute = u32::try_from((rem % 3600) / 60).unwrap_or(0);
    let second = u32::try_from(rem % 60).unwrap_or(0);
    // Howard Hinnant civil-from-days.
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097).cast_unsigned();
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let mut year = yoe.cast_signed() + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = u32::try_from(doy - (153 * mp + 2) / 5 + 1).unwrap_or(1);
    let month_raw = if mp < 10 { mp + 3 } else { mp - 9 };
    if month_raw <= 2 {
        year += 1;
    }
    let month = u32::try_from(month_raw).unwrap_or(1);
    (year, month, day, hour, minute, second)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn epoch_zero_and_known_instant() {
        assert_eq!(ms_to_iso(0), "1970-01-01T00:00:00.000Z");
        assert_eq!(ms_to_clock(0), "00:00:00");
        // 2026-08-01T08:22:45Z
        assert_eq!(ms_to_iso(1_785_572_565_000), "2026-08-01T08:22:45.000Z");
        assert_eq!(ms_to_clock(1_785_572_565_000), "08:22:45");
    }
}
