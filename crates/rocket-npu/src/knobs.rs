//! Environment knobs, read with the C library's semantics.
//!
//! The upstream library resolves most knobs lazily from the environment (so a test
//! can set one between calls). These helpers keep that behaviour; callers that need
//! a once-per-process value cache it themselves (as the C code does for the hardware
//! profile and capability probe).
//!
//! Burn-rocket-only knobs (`ROCKET_CTX_CANONICAL`, `ROCKET_BF16_STREAM`,
//! `ROCKET_MATMUL_CHUNK_M`) stay in the Burn crate: they are policy, not driver.

/// The raw value of `name`, if set and non-empty.
pub fn string(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

/// Whether `name` is present at all (the C library's `getenv(...) != NULL` checks).
pub fn present(name: &str) -> bool {
    std::env::var_os(name).is_some()
}

/// A boolean knob: absent -> `None`; `0`, `false`, `off`, `no` -> `Some(false)`;
/// anything else -> `Some(true)`. Matches the C code's `!strcmp(v, "0")` tests.
pub fn boolean(name: &str) -> Option<bool> {
    let v = std::env::var(name).ok()?;
    Some(!matches!(v.as_str(), "0" | "false" | "off" | "no"))
}

/// Whether a boolean knob is explicitly enabled (`set` and not a false spelling).
pub fn enabled(name: &str) -> bool {
    boolean(name) == Some(true)
}

/// Parse an integer knob, accepting the leading-numeric form C's `strtol` accepts.
pub fn int(name: &str) -> Option<i64> {
    let v = std::env::var(name).ok()?;
    let v = v.trim();
    let end = v
        .char_indices()
        .find(|(i, c)| !(c.is_ascii_digit() || (*i == 0 && matches!(c, '+' | '-'))))
        .map(|(i, _)| i)
        .unwrap_or(v.len());
    v.get(..end)?.parse().ok()
}

/// An integer knob with a default.
pub fn int_or(name: &str, default: i64) -> i64 {
    int(name).unwrap_or(default)
}

/// A floating-point knob with C `atof` semantics: `None` when unset or empty,
/// otherwise the longest numeric prefix (`0.0` when nothing parses).
pub fn float(name: &str) -> Option<f64> {
    let v = std::env::var(name).ok()?;
    if v.is_empty() {
        return None;
    }
    Some(atof(&v))
}

/// C `atof`: leading whitespace, then the longest prefix `strtod` accepts (`0.0`
/// when there is none). C also accepts hex floats (`0x1p3`); the plain decimal form
/// covers the knobs this crate reads.
pub fn atof(s: &str) -> f64 {
    let t = s.trim_start();
    let mut end = t.len();
    while end > 0 {
        if let Some(prefix) = t.get(..end)
            && let Ok(v) = prefix.parse::<f64>()
        {
            return v;
        }
        end -= 1;
        while end > 0 && !t.is_char_boundary(end) {
            end -= 1;
        }
    }
    0.0
}

/// An unsigned knob in C `strtoul(..., 0)` form: decimal, or `0x`/`0o`/`0b` prefixed.
/// A leading `-` is accepted (C's strtoul wraps it) but rarely useful.
pub fn u32_auto(name: &str) -> Option<u32> {
    let v = std::env::var(name).ok()?;
    let v = v.trim();
    let (neg, v) = match v.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, v),
    };
    let (radix, digits) = if let Some(rest) = v.strip_prefix("0x").or_else(|| v.strip_prefix("0X"))
    {
        (16, rest)
    } else if let Some(rest) = v.strip_prefix("0o").or_else(|| v.strip_prefix("0O")) {
        (8, rest)
    } else if let Some(rest) = v.strip_prefix("0b").or_else(|| v.strip_prefix("0B")) {
        (2, rest)
    } else {
        (10, v)
    };
    let end = digits
        .char_indices()
        .find(|(_, c)| !c.is_digit(radix))
        .map(|(i, _)| i)
        .unwrap_or(digits.len());
    if end == 0 {
        return None;
    }
    let mag = u32::from_str_radix(&digits[..end], radix).ok()?;
    Some(if neg { mag.wrapping_neg() } else { mag })
}

/// A signed knob in C `strtol(..., 0)` form: optional sign, then decimal or a
/// `0x`/`0o`/`0b` prefix (the C's base-0 parse).
pub fn i64_auto(name: &str) -> Option<i64> {
    let v = std::env::var(name).ok()?;
    let v = v.trim();
    let (neg, v) = match v.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, v.strip_prefix('+').unwrap_or(v)),
    };
    let (radix, digits) = if let Some(rest) = v.strip_prefix("0x").or_else(|| v.strip_prefix("0X"))
    {
        (16, rest)
    } else if let Some(rest) = v.strip_prefix("0o").or_else(|| v.strip_prefix("0O")) {
        (8, rest)
    } else if let Some(rest) = v.strip_prefix("0b").or_else(|| v.strip_prefix("0B")) {
        (2, rest)
    } else {
        (10, v)
    };
    let end = digits
        .char_indices()
        .find(|(_, c)| !c.is_digit(radix))
        .map(|(i, _)| i)
        .unwrap_or(digits.len());
    if end == 0 {
        return None;
    }
    let mag = i64::from_str_radix(&digits[..end], radix).ok()?;
    Some(if neg { -mag } else { mag })
}

/// A positive integer knob, ignoring zero/negative values (as the C timing knobs do).
pub fn positive(name: &str) -> Option<i64> {
    int(name).filter(|v| *v > 0)
}

/// A list of CPU ids such as `4-7`, `4,5,6,7` or `0,4-7`. Out-of-range ids are
/// dropped, matching `rocket_affinity.c`; an unparsable tail stops the scan.
pub fn cpu_list(name: &str) -> Option<Vec<usize>> {
    let s = string(name)?;
    let mut out = Vec::new();
    let bytes = s.as_bytes();
    let mut i = 0;
    const CPU_SETSIZE: usize = 1024;
    while i < bytes.len() {
        while i < bytes.len() && matches!(bytes[i], b',' | b' ') {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        let start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if start == i {
            break;
        }
        let a: usize = s[start..i].parse().ok()?;
        let mut b = a;
        if i < bytes.len() && bytes[i] == b'-' {
            i += 1;
            let start = i;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            if start == i {
                break;
            }
            b = s[start..i].parse().ok()?;
        }
        let b = b.min(CPU_SETSIZE - 1);
        for cpu in a..=b {
            if out.len() < CPU_SETSIZE {
                out.push(cpu);
            }
        }
    }
    Some(out)
}

/// A process-wide lock for tests that mutate the environment (the knobs are global,
/// and the LUT/regcmd builders read them).
#[cfg(test)]
pub(crate) static TEST_ENV: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    // The environment is process-wide, so these tests use distinct names.
    #[test]
    fn boolean_spellings() {
        unsafe { std::env::set_var("ROCKET_TEST_BOOL_A", "0") };
        assert_eq!(boolean("ROCKET_TEST_BOOL_A"), Some(false));
        unsafe { std::env::set_var("ROCKET_TEST_BOOL_A", "1") };
        assert_eq!(boolean("ROCKET_TEST_BOOL_A"), Some(true));
        unsafe { std::env::remove_var("ROCKET_TEST_BOOL_A") };
        assert_eq!(boolean("ROCKET_TEST_BOOL_A"), None);
    }

    #[test]
    fn int_accepts_trailing_junk() {
        unsafe { std::env::set_var("ROCKET_TEST_INT_A", "12ms") };
        assert_eq!(int("ROCKET_TEST_INT_A"), Some(12));
        unsafe { std::env::set_var("ROCKET_TEST_INT_A", "-3") };
        assert_eq!(int("ROCKET_TEST_INT_A"), Some(-3));
        unsafe { std::env::remove_var("ROCKET_TEST_INT_A") };
    }

    #[test]
    fn cpu_lists() {
        unsafe { std::env::set_var("ROCKET_TEST_CPUS_A", "4-7") };
        assert_eq!(cpu_list("ROCKET_TEST_CPUS_A"), Some(vec![4, 5, 6, 7]));
        unsafe { std::env::set_var("ROCKET_TEST_CPUS_A", "0,4-6") };
        assert_eq!(cpu_list("ROCKET_TEST_CPUS_A"), Some(vec![0, 4, 5, 6]));
        unsafe { std::env::remove_var("ROCKET_TEST_CPUS_A") };
    }

    #[test]
    fn atof_takes_the_longest_numeric_prefix() {
        assert_eq!(atof("-0.5"), -0.5);
        assert_eq!(atof(" 16"), 16.0);
        assert_eq!(atof("1e-3"), 1e-3);
        assert_eq!(atof("2.5x"), 2.5);
        assert_eq!(atof("1e"), 1.0);
        assert!(atof("nan").is_nan());
        assert_eq!(atof("junk"), 0.0);
    }

    #[test]
    fn float_knob_is_present_or_not() {
        unsafe { std::env::set_var("ROCKET_TEST_FLOAT_A", "6.5") };
        assert_eq!(float("ROCKET_TEST_FLOAT_A"), Some(6.5));
        unsafe { std::env::set_var("ROCKET_TEST_FLOAT_A", "") };
        assert_eq!(float("ROCKET_TEST_FLOAT_A"), None);
        unsafe { std::env::remove_var("ROCKET_TEST_FLOAT_A") };
        assert_eq!(float("ROCKET_TEST_FLOAT_A"), None);
    }
}
