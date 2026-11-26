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

#[cfg(feature = "sweeps-a")]
use tinymind::int8::{QAdd, QSoftmax, Requantizer};

use crate::board::{self, Uart};

#[cfg(feature = "sweeps-a")]
#[path = "../../tests/sweeps/add.rs"]
mod add;
#[cfg(feature = "sweeps-a")]
#[path = "../../tests/sweeps/compose.rs"]
mod compose;
#[cfg(feature = "sweeps-a")]
#[path = "../../tests/sweeps/conv.rs"]
mod conv;
// sweeps-b uses only its twiddle reader (for the log-mel sweep).
#[cfg_attr(not(feature = "sweeps-a"), allow(dead_code))]
#[path = "../../tests/sweeps/fft.rs"]
mod fft;
#[cfg(feature = "sweeps-b")]
#[path = "../../tests/sweeps/int16x8.rs"]
mod int16x8;
#[cfg(feature = "sweeps-b")]
#[path = "../../tests/sweeps/int4.rs"]
mod int4;
#[cfg(feature = "sweeps-a")]
#[path = "../../tests/sweeps/lstm.rs"]
mod lstm;
#[cfg(feature = "sweeps-b")]
#[path = "../../tests/sweeps/mel.rs"]
mod mel;
#[cfg(feature = "sweeps-a")]
#[path = "../../tests/sweeps/oracle.rs"]
mod oracle;
#[cfg(feature = "sweeps-b")]
#[path = "../../tests/sweeps/qanomaly.rs"]
mod qanomaly;
#[cfg(feature = "sweeps-a")]
#[path = "../../tests/sweeps/qlearn.rs"]
mod qlearn;
#[cfg(feature = "sweeps-a")]
#[path = "../../tests/sweeps/recurrent_int8.rs"]
mod recurrent_int8;
#[cfg(feature = "sweeps-a")]
#[path = "../../tests/sweeps/softmax.rs"]
mod softmax;
#[cfg(feature = "sweeps-a")]
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
    #[cfg(feature = "sweeps-a")]
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
#[cfg(feature = "sweeps-a")]
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
#[cfg(feature = "sweeps-a")]
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

#[cfg(feature = "sweeps-a")]
fn narrow(v: i32) -> i8 {
    i8::try_from(v).expect("int8 field")
}

/// The add sweep's calibrated layers, from the golden's `calibrated` lines.
/// The zero points and range come from the sweep's generator too: they must
/// agree with the line, or the device is reading another trial's parameters.
#[cfg(feature = "sweeps-a")]
struct AddParams(GoldenLines);

#[cfg(feature = "sweeps-a")]
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
#[cfg(feature = "sweeps-a")]
fn checked(rq: Requantizer, zo: i8, qmin: i32, qmax: i32, what: &str) -> Requantizer {
    let read = (rq.zero_point, i32::from(rq.qmin), i32::from(rq.qmax));
    assert!(read == (zo, qmin, qmax), "{what}: golden line out of step");
    rq
}

/// The compose sweep's calibrated requantizers: one per `mul calibrated` line,
/// two per `concat` line (input a, then b).
#[cfg(feature = "sweeps-a")]
struct ComposeParams {
    mul: GoldenLines,
    concat: GoldenLines,
    /// The concat line being read and which of its requantizers is next.
    line: &'static str,
    next: usize,
}

#[cfg(feature = "sweeps-a")]
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
#[cfg(feature = "sweeps-a")]
struct SoftmaxParams(GoldenLines);

#[cfg(feature = "sweeps-a")]
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

/// The FFT sweep's twiddle tables: `twiddles n=N k=K cos=... sin=...` lines,
/// up to 16 entries each, in order.
struct FftTwiddles(GoldenLines);

impl fft::Twiddles for FftTwiddles {
    fn twiddles<const N: usize, const HALF: usize>(&mut self) -> ([i16; HALF], [i16; HALF]) {
        let (mut cos, mut sin) = ([0i16; HALF], [0i16; HALF]);
        for k in (0..HALF).step_by(16) {
            let line = self.0.next();
            let mut fields = line.split(' ');
            assert!(fields.nth(1) == Some(itoa_eq(N)), "twiddles out of step");
            for f in fields {
                let (table, values) = if let Some(v) = f.strip_prefix("cos=") {
                    (&mut cos, v)
                } else if let Some(v) = f.strip_prefix("sin=") {
                    (&mut sin, v)
                } else {
                    continue;
                };
                for (slot, v) in table[k..].iter_mut().zip(values.split(',')) {
                    *slot = v.parse().expect("integer");
                }
            }
        }
        (cos, sin)
    }
}

/// The anomaly sweep's built Gaussians (trials 3 to 5) are only hashed in the
/// golden: they run on the host only.
#[cfg(feature = "sweeps-b")]
struct NoGaussian;

#[cfg(feature = "sweeps-b")]
impl qanomaly::Calibrate for NoGaussian {
    fn gaussian<const F: usize, const K: usize>(
        &mut self,
        _means: &[[f64; F]; K],
        _variances: &[[f64; F]; K],
        _weights: &[f64; K],
        _scale: f32,
        _zero_point: i32,
        _floor: f64,
    ) -> Option<qanomaly::Gaussian<F, K>> {
        None
    }
}

/// The int4 sweep's float quantizer needs std: its lines run on the host only.
#[cfg(feature = "sweeps-b")]
struct NoQuantize;

#[cfg(feature = "sweeps-b")]
impl int4::Quantize for NoQuantize {
    fn quantize(&mut self, _w: &[[f32; 9]; 4]) -> Option<([[i8; 9]; 4], [f32; 4])> {
        None
    }
}

/// The 16x8 sweep's calibrated additions (`add calibrated T: ls am as bm bs
/// om os`) and softmax parameters (`softmax N T params m s`), from the golden.
/// The zero points are 0 and the range int16's, as the sweep builds them. The
/// built activation tables are only hashed in the golden: those lines run on
/// the host only.
#[cfg(feature = "sweeps-b")]
struct Int16x8Params {
    add: GoldenLines,
    softmax: core::iter::Filter<core::str::Lines<'static>, fn(&&'static str) -> bool>,
}

/// The integers after `: ` (an addition) or after ` params ` (a softmax).
#[cfg(feature = "sweeps-b")]
fn trailing_ints<const N: usize>(line: &str, after: &str) -> [i32; N] {
    let (_, rest) = line.split_once(after).expect("golden field");
    let mut values = rest.split(' ').map(|v| v.parse::<i32>().expect("integer"));
    let out = core::array::from_fn(|_| values.next().expect("enough values"));
    assert!(values.next().is_none(), "no extra values");
    out
}

#[cfg(feature = "sweeps-b")]
impl int16x8::Calibrate for Int16x8Params {
    fn add(&mut self, _sa: f32, _sb: f32, _so: f32) -> tinymind::int16x8::QAdd {
        let [left_shift, am, ash, bm, bsh, om, osh] = trailing_ints(self.add.next(), ": ");
        tinymind::int16x8::QAdd {
            input_a_zero_point: 0,
            input_b_zero_point: 0,
            left_shift,
            input_a_multiplier: am,
            input_a_shift: ash,
            input_b_multiplier: bm,
            input_b_shift: bsh,
            output_requantizer: tinymind::int16x8::Requantizer32 {
                multiplier: om,
                shift: osh,
                zero_point: 0,
                qmin: -32768,
                qmax: 32767,
            },
        }
    }

    fn lut(
        &mut self,
        _sigmoid: bool,
        _in_scale: f64,
        _in_zp: i32,
    ) -> Option<[i16; tinymind::int16x8::LUT_SIZE]> {
        None
    }

    fn softmax(&mut self, _scale: f64, _beta: f64) -> tinymind::int16x8::QSoftmax {
        let line = self.softmax.next().expect("golden softmax parameters");
        let [input_multiplier, input_shift] = trailing_ints(line, " params ");
        tinymind::int16x8::QSoftmax {
            input_multiplier,
            input_shift,
        }
    }
}

const FFT: &str = include_str!("../../tests/golden/fft/stdout.txt");

/// The log-mel sweep on the device: twiddles from the FFT golden's tables for
/// that length; the built window, filterbank and DCT need libm and are only
/// hashed in the golden, so those configurations run on the host only.
#[cfg(feature = "sweeps-b")]
struct MelTables;

#[cfg(feature = "sweeps-b")]
impl mel::Calibrate for MelTables {
    fn twiddles<const N: usize, const HALF: usize>(&mut self) -> ([i16; HALF], [i16; HALF]) {
        let first = FFT
            .lines()
            .find(|l| l.starts_with("twiddles ") && l.split(' ').nth(1) == Some(itoa_eq(N)))
            .expect("FFT golden twiddles for this length");
        let offset = first.as_ptr() as usize - FFT.as_ptr() as usize;
        let rest = FFT.get(offset..).expect("offset inside the golden");
        fft::Twiddles::twiddles::<N, HALF>(&mut FftTwiddles(GoldenLines::new(rest, "twiddles ")))
    }

    fn tables<const N: usize, const BANDS: usize, const COEFFS: usize, const CAP: usize>(
        &mut self,
        _periodic: bool,
        _sample_rate: f64,
        _f_min: f64,
        _f_max: f64,
    ) -> Option<mel::Tables<N, BANDS, COEFFS, CAP>> {
        None
    }
}

/// `n=<n>`, for comparing a golden field without formatting.
fn itoa_eq(n: usize) -> &'static str {
    match n {
        2 => "n=2",
        4 => "n=4",
        8 => "n=8",
        16 => "n=16",
        32 => "n=32",
        64 => "n=64",
        128 => "n=128",
        256 => "n=256",
        512 => "n=512",
        1024 => "n=1024",
        _ => "",
    }
}

/// The int8 recurrent sweep's calibrated trials need weights, biases and LUTs
/// that the golden prints only as hashes: they run here only to keep the
/// generator in step, and only the direct-parameter lines are compared.
#[cfg(feature = "sweeps-a")]
struct NoCalibration;

#[cfg(feature = "sweeps-a")]
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

/// Cycles per sweep, summed in 64 bits. The F411's cycle counter (DWT_CYCCNT)
/// is 32 bits wide and wraps every 2^32 cycles (43 s at 100 MHz); the whole
/// run takes longer than that, so a single difference of two readings would
/// be the total modulo 2^32. Each sweep must stay under 2^32 cycles (the
/// longest, softmax, took 2.9e9 on 2026-11-14). UART output between sweeps is
/// not counted.
struct Laps {
    start: u32,
    total: u64,
}

impl Laps {
    fn start() -> Self {
        Laps {
            start: board::cycles(),
            total: 0,
        }
    }

    /// Report `<name>_cycles=`, the cycles since the last report (or the
    /// start), then restart the count once `report` has written its lines.
    fn lap(&mut self, out: &mut Uart, name: &str, report: impl FnOnce(&mut Uart)) {
        let cycles = board::cycles().wrapping_sub(self.start);
        self.total += u64::from(cycles);
        report(out);
        let _ = writeln!(out, "{name}_cycles={cycles}");
        self.start = board::cycles();
    }
}

pub fn run(out: &mut Uart) {
    let _ = writeln!(out, "name=sweeps");
    let mut laps = Laps::start();
    #[cfg(feature = "sweeps-a")]
    run_a(out, &mut laps);
    #[cfg(feature = "sweeps-b")]
    run_b(out, &mut laps);
    // All of the above, on a board (0 under QEMU, which has no cycle counter).
    let _ = writeln!(out, "cycles={}", laps.total);
}

/// The sweeps up to tinymind (C++) v18 (convolutions to the Q-format oracle).
#[cfg(feature = "sweeps-a")]
fn run_a(out: &mut Uart, laps: &mut Laps) {
    let mut e = Expect::new(include_str!("../../tests/golden/conv/stdout.txt"));
    conv::run(&mut e);
    laps.lap(out, "conv", |out| e.report(out, "conv"));

    const ADD: &str = include_str!("../../tests/golden/add/stdout.txt");
    let mut e = Expect::new(ADD);
    add::report(&mut e, &mut AddParams(GoldenLines::new(ADD, "calibrated ")));
    laps.lap(out, "add", |out| e.report(out, "add"));

    const COMPOSE: &str = include_str!("../../tests/golden/compose/stdout.txt");
    let mut e = Expect::new(COMPOSE);
    let mut cal = ComposeParams {
        mul: GoldenLines::new(COMPOSE, "mul calibrated "),
        concat: GoldenLines::new(COMPOSE, "concat "),
        line: "",
        next: 0,
    };
    compose::report(&mut e, &mut cal);
    laps.lap(out, "compose", |out| e.report(out, "compose"));

    const SOFTMAX: &str = include_str!("../../tests/golden/softmax/stdout.txt");
    let mut e = Expect::new(SOFTMAX);
    softmax::report(&mut e, &mut SoftmaxParams(GoldenLines::new(SOFTMAX, "e=")));
    laps.lap(out, "softmax", |out| e.report(out, "softmax"));

    let mut e = Expect::new(FFT);
    fft::report(&mut e, &mut FftTwiddles(GoldenLines::new(FFT, "twiddles ")));
    laps.lap(out, "fft", |out| e.report(out, "fft"));

    let mut e = Expect::keeping(
        include_str!("../../tests/golden/recurrent_int8/stdout.txt"),
        |l| !l.contains(" calibrated "),
    );
    recurrent_int8::report(&mut e, &mut NoCalibration);
    laps.lap(out, "recurrent_int8", |out| e.report(out, "recurrent_int8"));

    let mut e = Expect::new(include_str!("../../tests/golden/lstm/stdout.txt"));
    lstm::report(&mut e);
    laps.lap(out, "lstm", |out| e.report(out, "lstm"));

    let mut e = Expect::new(include_str!("../../tests/golden/tree/stdout.txt"));
    tree::report(&mut e);
    laps.lap(out, "tree", |out| e.report(out, "tree"));

    // The maze (the golden's first line) runs on the host only.
    let mut e = Expect::keeping(include_str!("../../tests/golden/qlearn/stdout.txt"), |l| {
        !l.starts_with("maze ")
    });
    qlearn::report(&mut e);
    laps.lap(out, "qlearn", |out| e.report(out, "qlearn"));

    // The Q-format oracle checks itself (an exact reference in the same
    // program): every check must pass, and all of the default sample must run.
    let oracles = [
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
    ];
    laps.lap(out, "oracle", |_| {});
    for (name, t, expected) in oracles {
        let _ = writeln!(out, "{name}_checks={}", t.checks);
        let _ = writeln!(out, "{name}_failures={}", t.failures);
        for failure in t.first_failures() {
            let _ = writeln!(out, "{name}_failure={failure}");
        }
        let ok = t.failures == 0 && t.checks == expected;
        let _ = writeln!(out, "{name}_match={}", u8::from(ok));
    }
}

/// tinymind (C++) v19.0's sweeps: log-mel, 16x8, anomaly scores, int4.
#[cfg(feature = "sweeps-b")]
fn run_b(out: &mut Uart, laps: &mut Laps) {
    // The random-table configurations (rand16, rand64).
    let mut e = Expect::keeping(include_str!("../../tests/golden/mel/stdout.txt"), |l| {
        l.split(' ')
            .nth(1)
            .is_some_and(|name| name.starts_with("rand"))
    });
    mel::run(&mut e, &mut MelTables);
    laps.lap(out, "mel", |out| e.report(out, "mel"));

    const INT16X8: &str = include_str!("../../tests/golden/int16x8/stdout.txt");
    // Tables 0 and 1 are random, 2 to 5 built (host only).
    let mut e = Expect::keeping(INT16X8, |l| {
        !(l.starts_with("lut table ")
            || l.starts_with("lut apply ")
                && !(l.starts_with("lut apply 0 ") || l.starts_with("lut apply 1 ")))
    });
    let mut cal = Int16x8Params {
        add: GoldenLines::new(INT16X8, "add calibrated "),
        softmax: INT16X8
            .lines()
            .filter((|l: &&str| l.contains(" params ")) as fn(&&'static str) -> bool),
    };
    int16x8::run(&mut e, &mut cal);
    laps.lap(out, "int16x8", |out| e.report(out, "int16x8"));

    let mut e = Expect::keeping(
        include_str!("../../tests/golden/qanomaly/stdout.txt"),
        |l| {
            !(l.starts_with("gaussian ")
                && (l.contains(" trial=3 ") || l.contains(" trial=4 ") || l.contains(" trial=5 ")))
        },
    );
    qanomaly::run(&mut e, &mut NoGaussian);
    laps.lap(out, "qanomaly", |out| e.report(out, "qanomaly"));

    let mut e = Expect::keeping(include_str!("../../tests/golden/int4/stdout.txt"), |l| {
        !l.starts_with("quantize ")
    });
    int4::run(&mut e, &mut NoQuantize);
    laps.lap(out, "int4", |out| e.report(out, "int4"));
}
