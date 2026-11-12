//! The host test sweeps (tests/sweeps/) on the device: each writes the same
//! report as on the host, and the report must equal the committed golden, byte
//! for byte. The goldens come from tinymind (C++)'s programs (scripts/cpp/), so
//! a match proves the layers compute on this instruction set what the C++
//! computes on the host. A correctness check, run under QEMU by
//! scripts/qemu-bench.sh. It also runs on the F411 at opt-level s (at 3 the
//! image exceeds its 512 KB of flash), which reports the total cycles; it does
//! not fit the L031. What the device cannot reproduce (calibration, which needs
//! std) is described at `GoldenLines` and `NoCalibration`.

use core::fmt::{self, Write};

use tinymind::int8::{QAdd, QSoftmax, Requantizer};

use crate::board::{self, Uart};

#[path = "../../tests/sweeps/add.rs"]
mod add;
#[path = "../../tests/sweeps/compose.rs"]
mod compose;
#[path = "../../tests/sweeps/conv.rs"]
mod conv;
#[path = "../../tests/sweeps/lstm.rs"]
mod lstm;
#[path = "../../tests/sweeps/oracle.rs"]
mod oracle;
#[path = "../../tests/sweeps/qlearn.rs"]
mod qlearn;
#[path = "../../tests/sweeps/recurrent_int8.rs"]
mod recurrent_int8;
#[path = "../../tests/sweeps/softmax.rs"]
mod softmax;
#[path = "../../tests/sweeps/tree.rs"]
mod tree;

/// Longest report line a sweep writes (the int8 recurrent sweep's are about
/// 600 bytes).
const LINE: usize = 1024;

/// A `fmt::Write` sink that compares what is written with `golden`, line by
/// line as it is written, so no report has to be held in RAM. Only the lines
/// `keep` accepts are compared, on both sides: the rest are the host's part of
/// the report.
struct Expect {
    golden: core::str::Lines<'static>,
    /// Golden lines read so far (for the line number of a difference).
    golden_line: usize,
    keep: fn(&str) -> bool,
    buf: [u8; LINE],
    len: usize,
    compared: usize,
    /// The first golden line that differs: its number and text ("" at the end
    /// of the golden).
    first_diff: Option<(usize, &'static str)>,
}

impl Expect {
    fn new(golden: &'static str) -> Self {
        Self::keeping(golden, |_| true)
    }

    fn keeping(golden: &'static str, keep: fn(&str) -> bool) -> Self {
        Expect {
            golden: golden.lines(),
            golden_line: 0,
            keep,
            buf: [0; LINE],
            len: 0,
            compared: 0,
            first_diff: None,
        }
    }

    /// The next golden line `keep` accepts, with its number.
    fn next_golden(&mut self) -> Option<(usize, &'static str)> {
        for line in self.golden.by_ref() {
            self.golden_line += 1;
            if (self.keep)(line) {
                return Some((self.golden_line, line));
            }
        }
        None
    }

    /// One complete line of the device's report.
    fn line(&mut self, line: &[u8]) {
        let Ok(text) = core::str::from_utf8(line) else {
            return self.differ(None);
        };
        if self.first_diff.is_some() || !(self.keep)(text) {
            return;
        }
        self.compared += 1;
        match self.next_golden() {
            Some((_, want)) if want.as_bytes() == line => {}
            other => self.differ(other),
        }
    }

    fn differ(&mut self, at: Option<(usize, &'static str)>) {
        if self.first_diff.is_none() {
            self.first_diff = Some(at.unwrap_or((self.golden_line + 1, "")));
        }
    }

    /// Report `<name>_match=1` if every kept line equals the golden's, and the
    /// first golden line that differs otherwise.
    fn report(&mut self, out: &mut Uart, name: &str) {
        if self.len != 0 {
            // A last line without its newline.
            let (buf, len) = (self.buf, self.len);
            self.line(&buf[..len]);
            self.len = 0;
        }
        if self.first_diff.is_none()
            && let Some(left) = self.next_golden()
        {
            // The device wrote fewer lines than the golden has.
            self.differ(Some(left));
        }
        let _ = writeln!(out, "{name}_lines={}", self.compared);
        match self.first_diff {
            None => {
                let _ = writeln!(out, "{name}_match=1");
            }
            Some((number, line)) => {
                let _ = writeln!(out, "{name}_first_diff_line={number}");
                let _ = writeln!(out, "{name}_expected_line={line}");
                let _ = writeln!(out, "{name}_match=0");
            }
        }
    }
}

impl Write for Expect {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for &b in s.as_bytes() {
            if b == b'\n' {
                let (buf, len) = (self.buf, self.len);
                self.line(&buf[..len]);
                self.len = 0;
            } else if let Some(slot) = self.buf.get_mut(self.len) {
                *slot = b;
                self.len += 1;
            } else {
                // Longer than any golden line: cannot match.
                self.differ(None);
            }
        }
        Ok(())
    }
}

/// Lines of a golden that start with `prefix`, in order: where the device
/// reads the parameters the host computes with `int8::calibration` (std only).
/// The golden is tinymind (C++)'s output, so these are the C++'s parameters.
struct GoldenLines {
    lines: core::str::Lines<'static>,
    prefix: &'static str,
}

impl GoldenLines {
    fn new(golden: &'static str, prefix: &'static str) -> Self {
        GoldenLines {
            lines: golden.lines(),
            prefix,
        }
    }

    /// The next line with the prefix. A missing one is a broken golden: panic,
    /// which fails the run.
    fn next(&mut self) -> &'static str {
        let prefix = self.prefix;
        self.lines
            .by_ref()
            .find(|l| l.starts_with(prefix))
            .expect("golden line with the expected prefix")
    }
}

/// The integers of the `occurrence`-th `key` field of `line` (`key=1,-2;3 ...`:
/// separated by `,` or `;`, ending at a space). A malformed field panics.
fn ints<const N: usize>(line: &str, key: &str, occurrence: usize) -> [i32; N] {
    let field = line
        .split(' ')
        .filter_map(|t| t.strip_prefix(key))
        .nth(occurrence)
        .expect("golden field");
    let mut values = field
        .split([',', ';'])
        .map(|v| v.parse::<i32>().expect("integer"));
    let out = core::array::from_fn(|_| values.next().expect("enough values"));
    assert!(values.next().is_none(), "no extra values");
    out
}

/// A requantizer from a golden's `rq=multiplier,shift,zero_point,qmin,qmax`.
fn requantizer(line: &str, occurrence: usize) -> Requantizer {
    let [multiplier, shift, zero_point, qmin, qmax] = ints(line, "rq=", occurrence);
    Requantizer {
        multiplier,
        shift,
        zero_point: narrow(zero_point),
        qmin: narrow(qmin),
        qmax: narrow(qmax),
    }
}

fn narrow(v: i32) -> i8 {
    i8::try_from(v).expect("int8 field")
}

/// The add sweep's calibrated layers, from the golden's `calibrated` lines.
/// The zero points and range come from the sweep's generator too: they must
/// agree with the line, or the device is reading another trial's parameters.
struct AddParams(GoldenLines);

impl add::Calibrate for AddParams {
    fn add(&mut self, a: (f32, i8), b: (f32, i8), out: (f32, i8), qmin: i8, qmax: i8) -> QAdd {
        let line = self.0.next();
        let [za, zb] = ints(line, "zp=", 0);
        let [left_shift] = ints(line, "ls=", 0);
        let [input_a_multiplier, input_a_shift] = ints(line, "a=", 0);
        let [input_b_multiplier, input_b_shift] = ints(line, "b=", 0);
        let output_requantizer = requantizer(line, 0);
        let drawn = (a.1, b.1, out.1, qmin, qmax);
        let read = (
            narrow(za),
            narrow(zb),
            output_requantizer.zero_point,
            output_requantizer.qmin,
            output_requantizer.qmax,
        );
        assert!(drawn == read, "add: golden line out of step");
        QAdd {
            input_a_zero_point: read.0,
            input_b_zero_point: read.1,
            left_shift,
            input_a_multiplier,
            input_a_shift,
            input_b_multiplier,
            input_b_shift,
            output_requantizer,
        }
    }
}

/// A requantizer read for a layer whose zero point and range the sweep drew:
/// they must agree with the golden line (see `AddParams`).
fn checked(rq: Requantizer, zo: i8, qmin: i32, qmax: i32, what: &str) -> Requantizer {
    let read = (rq.zero_point, i32::from(rq.qmin), i32::from(rq.qmax));
    assert!(read == (zo, qmin, qmax), "{what}: golden line out of step");
    rq
}

/// The compose sweep's calibrated requantizers: one per `mul calibrated` line,
/// two per `concat` line (input a, then b).
struct ComposeParams {
    mul: GoldenLines,
    concat: GoldenLines,
    /// The concat line being read and which of its requantizers is next.
    line: &'static str,
    next: usize,
}

impl compose::Calibrate for ComposeParams {
    fn mul_requantizer(
        &mut self,
        _sa: f32,
        _sb: f32,
        _so: f32,
        zo: i8,
        qmin: i32,
        qmax: i32,
    ) -> Requantizer {
        let rq = requantizer(self.mul.next(), 0);
        checked(rq, zo, qmin, qmax, "mul")
    }

    fn rescaler(&mut self, _s: f32, _so: f32, zo: i8, qmin: i32, qmax: i32) -> Requantizer {
        if self.next == 0 {
            self.line = self.concat.next();
        }
        let rq = requantizer(self.line, self.next);
        self.next = (self.next + 1) % 2;
        checked(rq, zo, qmin, qmax, "concat")
    }
}

/// The softmax sweep's calibrations: `params=multiplier,left_shift,diff_min`
/// on every line, in order.
struct SoftmaxParams(GoldenLines);

impl softmax::Calibrate for SoftmaxParams {
    fn softmax(&mut self, _input_scale: f64, _beta: f64) -> QSoftmax {
        let [input_multiplier, input_left_shift, diff_min] = ints(self.0.next(), "params=", 0);
        QSoftmax {
            input_multiplier,
            input_left_shift,
            diff_min,
        }
    }
}

/// The int8 recurrent sweep's calibrated trials need weights, biases and LUTs
/// that the golden prints only as hashes: they run here only to keep the
/// generator in step, and only the direct-parameter lines are compared.
struct NoCalibration;

impl recurrent_int8::Calibrate for NoCalibration {
    fn lstm<const I: usize, const H: usize>(
        &mut self,
        _: recurrent_int8::Scales,
        _: &[[[f32; I]; H]; 4],
        _: &[[[f32; H]; H]; 4],
        _: &[[f32; H]; 4],
    ) -> Option<recurrent_int8::LstmCalibration<I, H>> {
        None
    }

    fn gru<const I: usize, const H: usize>(
        &mut self,
        _: recurrent_int8::Scales,
        _: &[[[f32; I]; H]; 3],
        _: &[[[f32; H]; H]; 3],
        _: &[[f32; H]; 3],
    ) -> Option<recurrent_int8::GruCalibration<I, H>> {
        None
    }
}

pub fn run(out: &mut Uart) {
    let _ = writeln!(out, "name=sweeps");
    let start = board::cycles();

    let mut e = Expect::new(include_str!("../../tests/golden/conv/stdout.txt"));
    conv::run(&mut e);
    e.report(out, "conv");

    const ADD: &str = include_str!("../../tests/golden/add/stdout.txt");
    let mut e = Expect::new(ADD);
    add::report(&mut e, &mut AddParams(GoldenLines::new(ADD, "calibrated ")));
    e.report(out, "add");

    const COMPOSE: &str = include_str!("../../tests/golden/compose/stdout.txt");
    let mut e = Expect::new(COMPOSE);
    let mut cal = ComposeParams {
        mul: GoldenLines::new(COMPOSE, "mul calibrated "),
        concat: GoldenLines::new(COMPOSE, "concat "),
        line: "",
        next: 0,
    };
    compose::report(&mut e, &mut cal);
    e.report(out, "compose");

    const SOFTMAX: &str = include_str!("../../tests/golden/softmax/stdout.txt");
    let mut e = Expect::new(SOFTMAX);
    softmax::report(&mut e, &mut SoftmaxParams(GoldenLines::new(SOFTMAX, "e=")));
    e.report(out, "softmax");

    let mut e = Expect::keeping(
        include_str!("../../tests/golden/recurrent_int8/stdout.txt"),
        |l| !l.contains(" calibrated "),
    );
    recurrent_int8::report(&mut e, &mut NoCalibration);
    e.report(out, "recurrent_int8");

    let mut e = Expect::new(include_str!("../../tests/golden/lstm/stdout.txt"));
    lstm::report(&mut e);
    e.report(out, "lstm");

    let mut e = Expect::new(include_str!("../../tests/golden/tree/stdout.txt"));
    tree::report(&mut e);
    e.report(out, "tree");

    // The maze (the golden's first line) runs on the host only.
    let mut e = Expect::keeping(include_str!("../../tests/golden/qlearn/stdout.txt"), |l| {
        !l.starts_with("maze ")
    });
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
