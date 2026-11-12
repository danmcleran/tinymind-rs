// Differential oracle for `QValue`, ported from tinymind's
// `unit_test/qformat_oracle/qformat_oracle_unit_test.cpp`.
//
// Every public `QValue` operation is run against an exact reference computed in
// wide integers. The port has signed formats that fill their storage (`i16`,
// `i32`) with tinymind's default TruncatePolicy and WrapPolicy, so this covers
// the signed full-width formats of the C++ oracle's 16- and 32-bit matrices
// under that policy pair: Q8.8, Q1.15, Q7.9, Q15.1, Q16.16, Q24.8, Q8.24, Q1.31
// and Q31.1. Inputs are edge values plus a fixed-seed sample from the same
// splitmix64 generator, per-format seeds and value mix as the C++ oracle, and
// random chains of operations.
//
// The C++ oracle keeps a list of known defects (behaviour QValue is known to get
// wrong, modelled exactly so the test still pins it down). That list is empty at
// the TINYMIND_REF this crate is verified against, so the reference here is
// simply the intended arithmetic: any mismatch is a bug in the port.
//
// The reference uses i64 rather than the C++ oracle's 128-bit integers: with at
// most 32-bit storage every exact intermediate (a product is at most 2^62) fits.
//
// TINYMIND_ORACLE_SEED and TINYMIND_ORACLE_SCALE work as in the C++ oracle: a
// soak run sets a fresh seed and a larger scale, and the seed and scale are
// printed so a failure can be reproduced. `q8_8_exhaustive` (ignored by default;
// run with `cargo test --release --test qformat_oracle -- --ignored`) checks the
// binary operations over every Q8.8 operand pair.
//
// The oracle itself is in tests/sweeps/oracle.rs, which bench/ also runs on
// Cortex-M (default sample).

#[path = "sweeps/oracle.rs"]
mod oracle;

use oracle::{Oracle, Tally, Wide, check_int, check_pair, check_value, int_operands};
use tinymind::QValue;

// === Seed, scale and report =================================================

struct RunConfig {
    seed: u64,
    scale: u64,
}

fn env_number(name: &str, fallback: u64, max: u64) -> u64 {
    match std::env::var(name) {
        Ok(text) if !text.is_empty() => {
            let parsed = if let Some(hex) = text.strip_prefix("0x") {
                u64::from_str_radix(hex, 16)
            } else {
                text.parse()
            };
            match parsed {
                Ok(v) if v <= max => v,
                // A soak that silently fell back to the default sample would pass
                // while testing nothing new.
                _ => panic!("{name}={text:?} is not a whole number in [0, {max}]"),
            }
        }
        _ => fallback,
    }
}

fn run_config() -> RunConfig {
    let scale = env_number("TINYMIND_ORACLE_SCALE", 1, 100_000);
    assert!(scale >= 1, "TINYMIND_ORACLE_SCALE must be at least 1");
    RunConfig {
        seed: env_number("TINYMIND_ORACLE_SEED", 0, u64::MAX),
        scale,
    }
}

fn report(t: &Tally, what: &str) {
    let config = run_config();
    println!(
        "{what}: {} checks, {} failures (seed {}, scale {})",
        t.checks, t.failures, config.seed, config.scale
    );
    let first: Vec<String> = t.first_failures().map(ToString::to_string).collect();
    assert!(
        t.failures == 0,
        "{what}: {} of {} checks failed; first ones:\n  {}",
        t.failures,
        t.checks,
        first.join("\n  ")
    );
}

// === Tests ==================================================================

#[test]
fn oracle_16bit_storage() {
    let c = run_config();
    let t = oracle::run_16bit(c.seed, c.scale);
    report(&t, "16-bit storage (Q8.8, Q1.15, Q7.9, Q15.1)");
    if (c.seed, c.scale) == (0, 1) {
        assert_eq!(t.checks, oracle::CHECKS_16BIT, "default sample size");
    }
}

#[test]
fn oracle_32bit_storage() {
    let c = run_config();
    let t = oracle::run_32bit(c.seed, c.scale);
    report(&t, "32-bit storage (Q16.16, Q24.8, Q8.24, Q1.31, Q31.1)");
    if (c.seed, c.scale) == (0, 1) {
        assert_eq!(t.checks, oracle::CHECKS_32BIT, "default sample size");
    }
}

/// Every Q8.8 operand pair through every binary operation, and every Q8.8 value
/// through the single-value checks and the integer divisors. About 4.3 billion
/// pairs: run it in release (`cargo test --release --test qformat_oracle --
/// --ignored`); CI runs it nightly.
#[test]
#[ignore = "exhaustive: minutes in release, far longer in debug"]
fn q8_8_exhaustive() {
    type Q = QValue<i16, 8>;
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let all: Vec<Wide> = (Wide::from(i16::MIN)..=Wide::from(i16::MAX)).collect();
    let chunk = all.len().div_ceil(threads);
    let ints = int_operands(Q::FORMAT);

    let tallies: Vec<Tally> = std::thread::scope(|scope| {
        let handles: Vec<_> = all
            .chunks(chunk)
            .map(|lhs| {
                let (all, ints) = (&all, &ints);
                scope.spawn(move || {
                    let mut t = Tally::default();
                    for &a in lhs {
                        for &b in all {
                            check_pair::<Q>(&mut t, a, b);
                        }
                        check_value::<Q>(&mut t, a);
                        for n in ints.iter() {
                            check_int::<Q>(&mut t, a, n);
                        }
                    }
                    t
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut total = Tally::default();
    for t in tallies {
        total.merge(t);
    }
    report(&total, "Q8.8 exhaustive");
}
