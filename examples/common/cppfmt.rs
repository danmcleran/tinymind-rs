//! C++ iostream number formatting, shared by the examples' drivers
//! (`#[path]`-included; a directory under `examples/` without `main.rs` is not an
//! example of its own).

// Each binary compiles its own copy and uses a subset of these.
#![allow(dead_code)]

/// Format like C++ `std::ostream << double` with default flags, which is
/// `printf("%g")` with precision 6. Rust's `{}` prints the shortest round-trip
/// form instead (`0.08288574218750001`-style), so the C++ rule is rebuilt here:
/// round to 6 significant digits, use scientific notation when the exponent is
/// below -4 or at least 6, and strip trailing zeros.
pub fn fmt_g(v: f64) -> String {
    const PRECISION: i32 = 6;
    if v.is_nan() {
        return "nan".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "inf" } else { "-inf" }.into();
    }
    if v == 0.0 {
        return if v.is_sign_negative() { "-0" } else { "0" }.into();
    }

    // `{:.5e}` rounds to 6 significant digits and tells us the decimal exponent
    // *after* rounding (e.g. 9.999995 -> "1.00000e1"), which is what %g uses.
    let sci = format!("{:.*e}", (PRECISION - 1) as usize, v);
    let (mantissa, exponent) = sci.split_once('e').expect("{:e} always has an exponent");
    let exponent: i32 = exponent.parse().expect("exponent is an integer");

    if !(-4..PRECISION).contains(&exponent) {
        let sign = if exponent < 0 { '-' } else { '+' };
        format!(
            "{}e{sign}{:02}",
            strip_trailing_zeros(mantissa),
            exponent.abs()
        )
    } else {
        let decimals = (PRECISION - 1 - exponent) as usize;
        strip_trailing_zeros(&format!("{v:.decimals$}")).to_string()
    }
}

/// Format like C++ `std::ostream << std::fixed << std::setprecision(digits)`,
/// i.e. `printf("%.*f")`. Rust's `{:.*}` also rounds the exact binary value
/// correctly, so it gives the same digits.
pub fn fmt_fixed(v: f64, digits: usize) -> String {
    format!("{v:.digits$}")
}

fn strip_trailing_zeros(s: &str) -> &str {
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.')
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::{fmt_fixed, fmt_g};

    #[test]
    fn fmt_g_matches_printf() {
        // Expected strings are what glibc printf("%g") produces.
        let cases = [
            (0.0, "0"),
            (1.0, "1"),
            (0.5, "0.5"),
            (0.025, "0.025"),
            (0.0828857421875, "0.0828857"),
            (0.911819458, "0.911819"),
            (123456.0, "123456"),
            (1234567.0, "1.23457e+06"),
            (0.0001, "0.0001"),
            (0.00001234, "1.234e-05"),
            (9.9999996, "10"),
            (-2.5, "-2.5"),
        ];
        for (v, expected) in cases {
            assert_eq!(fmt_g(v), expected, "fmt_g({v})");
        }
    }

    #[test]
    fn fmt_fixed_matches_printf() {
        // Expected strings are what C++ `std::fixed << setprecision(4)` (glibc
        // printf("%.4f") on a double) produces, including values whose decimal form
        // is a tie but whose binary value is not. Do not take expectations from
        // bash's printf: it parses arguments as long double and rounds 0.99995 down.
        let cases = [
            (0.0659741, "0.0660"),
            (0.01485, "0.0149"),
            (1.0, "1.0000"),
            (0.0, "0.0000"),
            (0.12345, "0.1235"),
            (0.00005, "0.0001"),
            (0.99995, "1.0000"),
            (0.00015, "0.0001"),
            (2.675, "2.6750"),
        ];
        for (v, expected) in cases {
            assert_eq!(fmt_fixed(v, 4), expected, "fmt_fixed({v})");
        }
    }
}
