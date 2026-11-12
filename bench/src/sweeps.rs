//! The host test sweeps (tests/sweeps/) on the device: each writes the same
//! report as on the host, and the report must equal the committed golden, byte
//! for byte. The goldens come from tinymind (C++)'s programs (scripts/cpp/), so
//! a match proves the layers compute on this instruction set what the C++
//! computes on the host. A correctness check, run under QEMU by
//! scripts/qemu-bench.sh (it also runs on the F411, which reports the total
//! cycles; the image does not fit the L031).

use core::fmt::{self, Write};

use crate::board::{self, Uart};

#[path = "../../tests/sweeps/conv.rs"]
mod conv;
#[path = "../../tests/sweeps/lstm.rs"]
mod lstm;
#[path = "../../tests/sweeps/oracle.rs"]
mod oracle;
#[path = "../../tests/sweeps/qlearn.rs"]
mod qlearn;
#[path = "../../tests/sweeps/tree.rs"]
mod tree;

/// A `fmt::Write` sink that compares what is written with `golden`, as it is
/// written, so no report has to be held in RAM.
struct Expect {
    golden: &'static str,
    at: usize,
    /// Byte offset of the first difference, if any.
    first_diff: Option<usize>,
}

impl Expect {
    fn new(golden: &'static str) -> Self {
        Expect {
            golden,
            at: 0,
            first_diff: None,
        }
    }

    /// Compare with the golden after its first `lines` lines (the part of the
    /// report the device does not write).
    fn after_lines(golden: &'static str, lines: usize) -> Self {
        let start = golden
            .match_indices('\n')
            .nth(lines.wrapping_sub(1))
            .map_or(0, |(i, _)| i + 1);
        Self::new(&golden[start..])
    }

    /// Report `<name>_match=1` if everything written equals the golden, and
    /// where it went wrong otherwise.
    fn report(&mut self, out: &mut Uart, name: &str) {
        if self.first_diff.is_none() && self.at != self.golden.len() {
            self.first_diff = Some(self.at);
        }
        let _ = writeln!(out, "{name}_bytes={}", self.at);
        match self.first_diff {
            None => {
                let _ = writeln!(out, "{name}_match=1");
            }
            Some(at) => {
                // The golden line the difference falls in, to find it on the host.
                let start = self.golden[..at.min(self.golden.len())]
                    .rfind('\n')
                    .map_or(0, |i| i + 1);
                let line = self.golden[start..].lines().next().unwrap_or("");
                let number = self.golden[..start].matches('\n').count() + 1;
                let _ = writeln!(out, "{name}_first_diff_line={number}");
                let _ = writeln!(out, "{name}_expected_line={line}");
                let _ = writeln!(out, "{name}_match=0");
            }
        }
    }
}

impl Write for Expect {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        if self.first_diff.is_none() {
            let end = self.at + s.len();
            match self.golden.as_bytes().get(self.at..end) {
                Some(expected) if expected == s.as_bytes() => {}
                Some(expected) => {
                    let i = expected
                        .iter()
                        .zip(s.bytes())
                        .take_while(|(a, b)| **a == *b);
                    self.first_diff = Some(self.at + i.count());
                }
                None => self.first_diff = Some(self.at.min(self.golden.len())),
            }
        }
        self.at += s.len();
        Ok(())
    }
}

pub fn run(out: &mut Uart) {
    let _ = writeln!(out, "name=sweeps");
    let start = board::cycles();

    let mut e = Expect::new(include_str!("../../tests/golden/conv/stdout.txt"));
    conv::run(&mut e);
    e.report(out, "conv");

    let mut e = Expect::new(include_str!("../../tests/golden/lstm/stdout.txt"));
    lstm::report(&mut e);
    e.report(out, "lstm");

    let mut e = Expect::new(include_str!("../../tests/golden/tree/stdout.txt"));
    tree::report(&mut e);
    e.report(out, "tree");

    // The maze (the golden's first line) runs on the host only.
    let mut e = Expect::after_lines(include_str!("../../tests/golden/qlearn/stdout.txt"), 1);
    qlearn::report(&mut e);
    e.report(out, "qlearn");

    // The Q-format oracle checks itself (an exact reference in the same
    // program): every check must pass, and all of the default sample must run.
    for (name, t, expected) in [
        (
            "oracle_16bit",
            oracle::run_16bit(0, 1),
            oracle::CHECKS_16BIT,
        ),
        (
            "oracle_32bit",
            oracle::run_32bit(0, 1),
            oracle::CHECKS_32BIT,
        ),
    ] {
        let _ = writeln!(out, "{name}_checks={}", t.checks);
        let _ = writeln!(out, "{name}_failures={}", t.failures);
        for failure in t.first_failures() {
            let _ = writeln!(out, "{name}_failure={failure}");
        }
        let ok = t.failures == 0 && t.checks == expected;
        let _ = writeln!(out, "{name}_match={}", u8::from(ok));
    }
    // All of the above, on a board (0 under QEMU, which has no cycle counter).
    let _ = writeln!(out, "cycles={}", board::cycles().wrapping_sub(start));
}
