// QLogMelSpectrogram, QMfcc and calibration::build_hann_window,
// build_mel_filterbank and build_dct_matrix against tinymind (C++)
// (cpp/qmel.hpp): frame lengths 4 to 1024, band counts that leave bands
// empty, output Q formats 0, 8 and 14, log floors from -32768 to 32767; ten
// integer test signals with pre-emphasis off and on; then random tables.
// tests/golden/mel/stdout.txt is the output of scripts/cpp/mel_sweep.cpp
// (scripts/compare-with-cpp.sh rebuilds it and checks the golden is
// current); this file computes the same lines.
use std::fmt::Write;

use tinymind::int8::calibration::{
    build_dct_matrix, build_fft_twiddles, build_hann_window, build_mel_filterbank,
};
use tinymind::int8::{QLogMelSpectrogram, QMelWorkspace, QMfcc};

struct Rng(u32);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.0
    }
    fn i16(&mut self) -> i16 {
        (self.next() >> 16) as u16 as i16
    }
}

fn hash<T: Copy, const W: usize>(xs: &[T], bytes: impl Fn(T) -> [u8; W]) -> u32 {
    let mut h = 0x811C_9DC5u32;
    for &x in xs {
        for b in bytes(x) {
            h = (h ^ u32::from(b)).wrapping_mul(0x0100_0193);
        }
    }
    h
}

fn h16(xs: &[i16]) -> u32 {
    hash(xs, i16::to_le_bytes)
}

fn hu16(xs: &[u16]) -> u32 {
    hash(xs, u16::to_le_bytes)
}

const SIGNALS: usize = 10;

fn make_signal<const N: usize>(s: usize, rng: &mut Rng) -> ([i16; N], i16) {
    let mut previous = 0;
    let mut x = [0i16; N];
    for (n, v) in x.iter_mut().enumerate() {
        *v = match s {
            0 => 0,
            1 => i16::from(n == N / 2),
            2 => (rng.next() >> 30) as i16 - 2,
            3 => rng.i16(),
            4 => {
                if rng.next() >> 31 != 0 {
                    32767
                } else {
                    -32768
                }
            }
            5 => {
                let p = if N >= 8 { N / 8 } else { 2 };
                ((n % p) * 6000 / p) as i16 - 3000
            }
            6 => {
                if (n / 4) % 2 == 0 {
                    3
                } else {
                    -3
                }
            }
            7 => 1000,
            8 => -32768,
            _ => {
                previous = -32768;
                20000
            }
        };
    }
    (x, previous)
}

#[allow(clippy::too_many_arguments)]
fn run_frames<
    const N: usize,
    const HALF: usize,
    const BINS: usize,
    const BANDS: usize,
    const FRAC: u32,
    const COEFFS: usize,
>(
    out: &mut String,
    rng: &mut Rng,
    name: &str,
    mel: &QLogMelSpectrogram<'_, N, HALF, BANDS, FRAC>,
    mfcc: &QMfcc<'_, BANDS, COEFFS>,
    extreme_preemphasis: bool,
) {
    let mut workspace = QMelWorkspace::<N, BINS>::new();
    let mut layer = *mel;
    let variants = if extreme_preemphasis { 4 } else { 2 };
    for s in 0..SIGNALS {
        let (frame, signal_previous) = make_signal::<N>(s, rng);
        for v in 0..variants {
            let mut previous = signal_previous;
            match v {
                0 => layer.preemphasis = 0,
                1 => {
                    layer.preemphasis = 31785;
                    previous = if s == 9 { signal_previous } else { rng.i16() };
                }
                2 => {
                    layer.preemphasis = -32768;
                    previous = 32767;
                }
                _ => {
                    layer.preemphasis = 1;
                    previous = 1;
                }
            }
            let mut bands = [0i16; BANDS];
            let mut coeffs = [0i32; COEFFS];
            layer.forward(&frame, previous, &mut bands, &mut workspace);
            mfcc.forward(&bands, &mut coeffs);
            let first: Vec<String> = bands.iter().take(4).map(|v| v.to_string()).collect();
            writeln!(
                out,
                "mel {name} signal={s} pre={} prev={previous} out={} hash={:08x} mfcc={},{} hash={:08x}",
                layer.preemphasis,
                first.join(","),
                h16(&bands),
                coeffs[0],
                coeffs[if COEFFS > 1 { 1 } else { 0 }],
                hash(&coeffs, i32::to_le_bytes)
            )
            .unwrap();
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn built<
    const N: usize,
    const HALF: usize,
    const BINS: usize,
    const BANDS: usize,
    const FRAC: u32,
    const COEFFS: usize,
    const CAP: usize,
>(
    out: &mut String,
    rng: &mut Rng,
    name: &str,
    sample_rate: f64,
    f_min: f64,
    f_max: f64,
    log_floor: i16,
    periodic: bool,
    extreme_preemphasis: bool,
) {
    let window = build_hann_window::<N>(periodic);
    let (cos, sin) = build_fft_twiddles::<N, HALF>();
    let fb = build_mel_filterbank::<N, BANDS, CAP>(sample_rate, f_min, f_max).unwrap();
    // The C++ reports 0 from a call with too little room; here that is None.
    let short = build_mel_filterbank::<N, BANDS, 1>(sample_rate, f_min, f_max).map_or(0, |f| f.len);
    let dct = build_dct_matrix::<BANDS, COEFFS>();
    let empty = fb.band_length.iter().filter(|&&l| l == 0).count();
    let dct_flat: Vec<i16> = dct.iter().flatten().copied().collect();
    writeln!(
        out,
        "tables {name} window={:08x} first={},{},{} used={} short={short} empty={empty} start={:08x} length={:08x} weights={:08x} dct={:08x} dct0={},{}",
        h16(&window),
        window[0],
        window[1],
        window[N / 2],
        fb.len,
        hu16(&fb.band_start),
        hu16(&fb.band_length),
        hu16(fb.weights()),
        h16(&dct_flat),
        dct[0][0],
        dct[if COEFFS > 1 { 1 } else { 0 }][0]
    )
    .unwrap();
    let mel: QLogMelSpectrogram<'_, N, HALF, BANDS, FRAC> = QLogMelSpectrogram {
        window: &window,
        twiddle_cos: &cos,
        twiddle_sin: &sin,
        band_start: &fb.band_start,
        band_length: &fb.band_length,
        weights: fb.weights(),
        preemphasis: 0,
        log_floor,
    };
    let mfcc = QMfcc { dct: &dct };
    run_frames::<N, HALF, BINS, BANDS, FRAC, COEFFS>(
        out,
        rng,
        name,
        &mel,
        &mfcc,
        extreme_preemphasis,
    );
}

fn random_tables<
    const N: usize,
    const HALF: usize,
    const BINS: usize,
    const BANDS: usize,
    const FRAC: u32,
>(
    out: &mut String,
    rng: &mut Rng,
    name: &str,
) {
    let (cos, sin) = build_fft_twiddles::<N, HALF>();
    let mut window = [0i16; N];
    for w in window.iter_mut() {
        *w = rng.i16();
    }
    window[0] = if N == 64 { -32768 } else { 100 };
    let mut band_start = [0u16; BANDS];
    let mut band_length = [0u16; BANDS];
    let mut weights = Vec::new();
    for m in 0..BANDS {
        band_start[m] = ((rng.next() >> 16) as usize % BINS) as u16;
        band_length[m] =
            ((rng.next() >> 16) as usize % (BINS - usize::from(band_start[m]) + 1)) as u16;
        for _ in 0..band_length[m] {
            weights.push(if m == 0 {
                65535
            } else {
                (rng.next() >> 16) as u16
            });
        }
    }
    let mut dct = [[0i16; BANDS]; BANDS];
    for d in dct.iter_mut().flatten() {
        *d = rng.i16();
    }
    writeln!(
        out,
        "tables {name} random used={} start={:08x} length={:08x}",
        weights.len(),
        hu16(&band_start),
        hu16(&band_length)
    )
    .unwrap();
    let mel: QLogMelSpectrogram<'_, N, HALF, BANDS, FRAC> = QLogMelSpectrogram {
        window: &window,
        twiddle_cos: &cos,
        twiddle_sin: &sin,
        band_start: &band_start,
        band_length: &band_length,
        weights: &weights,
        preemphasis: 0,
        log_floor: -32768,
    };
    let mfcc = QMfcc { dct: &dct };
    run_frames::<N, HALF, BINS, BANDS, FRAC, BANDS>(out, rng, name, &mel, &mfcc, true);
}

#[test]
fn mel_sweep_matches_cpp() {
    let mut out = String::new();
    let rng = &mut Rng(7);
    built::<4, 2, 3, 2, 8, 1, 6>(&mut out, rng, "n4", 8.0, 0.0, 4.0, -32768, true, true);
    built::<64, 32, 33, 10, 8, 5, 66>(
        &mut out, rng, "n64", 8000.0, 20.0, 4000.0, -1500, true, false,
    );
    built::<128, 64, 65, 64, 8, 13, 130>(
        &mut out,
        rng,
        "n128-narrow",
        16000.0,
        0.0,
        8000.0,
        -32768,
        true,
        false,
    );
    built::<256, 128, 129, 32, 8, 13, 258>(
        &mut out, rng, "n256", 16000.0, 20.0, 8000.0, -2048, true, true,
    );
    built::<256, 128, 129, 32, 0, 32, 258>(
        &mut out, rng, "n256-q0", 16000.0, 20.0, 8000.0, -128, false, false,
    );
    built::<512, 256, 257, 26, 14, 13, 514>(
        &mut out, rng, "n512-q14", 44100.0, 30.0, 11000.0, -32768, true, false,
    );
    built::<1024, 512, 513, 40, 8, 20, 1026>(
        &mut out, rng, "n1024", 48000.0, 0.0, 24000.0, 32767, true, false,
    );
    random_tables::<16, 8, 9, 5, 8>(&mut out, rng, "rand16");
    random_tables::<64, 32, 33, 7, 14>(&mut out, rng, "rand64");

    let golden = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/golden/mel/stdout.txt"
    ))
    .unwrap();
    for (n, (got, want)) in out.lines().zip(golden.lines()).enumerate() {
        assert_eq!(got, want, "line {}", n + 1);
    }
    assert_eq!(out.lines().count(), golden.lines().count());
}
