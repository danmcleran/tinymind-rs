// The log-mel sweep, shared by tests/mel_sweep.rs (on the host, against
// tinymind (C++)'s output, with the calibration builders) and
// bench/src/sweeps.rs (on Cortex-M, against the same golden). The built
// window, filterbank and DCT need libm and are only hashed in the golden, so
// the device runs the random-table configurations (twiddles read from the FFT
// golden) and only makes the built ones' draws. no_std and allocation-free.
// The C++ is scripts/cpp/mel_sweep.cpp.

use core::fmt::Write;

use tinymind::int8::{QLogMelSpectrogram, QMelWorkspace, QMfcc};

/// The tables `int8::calibration` builds for one configuration.
pub struct Tables<const N: usize, const BANDS: usize, const COEFFS: usize, const CAP: usize> {
    /// `build_hann_window::<N>(periodic)`.
    pub window: [i16; N],
    /// The filterbank's band starts.
    pub band_start: [u16; BANDS],
    /// The filterbank's band lengths.
    pub band_length: [u16; BANDS],
    /// The filterbank's weights, `used` of them.
    pub weights: [u16; CAP],
    /// How many weights the filterbank uses.
    pub used: usize,
    /// The weights a filterbank with room for one reports (0 if none).
    pub short: usize,
    /// `build_dct_matrix::<BANDS, COEFFS>()`.
    pub dct: [[i16; BANDS]; COEFFS],
}

/// What the sweep needs from `int8::calibration` (std only).
pub trait Calibrate {
    /// `build_fft_twiddles::<N, HALF>()`.
    fn twiddles<const N: usize, const HALF: usize>(&mut self) -> ([i16; HALF], [i16; HALF]);
    /// The window, filterbank and DCT; `None` skips the configuration's lines.
    fn tables<const N: usize, const BANDS: usize, const COEFFS: usize, const CAP: usize>(
        &mut self,
        periodic: bool,
        sample_rate: f64,
        f_min: f64,
        f_max: f64,
    ) -> Option<Tables<N, BANDS, COEFFS, CAP>>;
}

/// The first four values, comma-separated.
struct First4<'a>(&'a [i16]);

impl core::fmt::Display for First4<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        for (i, v) in self.0.iter().take(4).enumerate() {
            if i > 0 {
                f.write_str(",")?;
            }
            write!(f, "{v}")?;
        }
        Ok(())
    }
}

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
    out: &mut impl Write,
    rng: &mut Rng,
    name: &str,
    layers: Option<(
        &QLogMelSpectrogram<'_, N, HALF, BANDS, FRAC>,
        &QMfcc<'_, BANDS, COEFFS>,
    )>,
    extreme_preemphasis: bool,
) {
    let mut workspace = QMelWorkspace::<N, BINS>::new();
    let variants = if extreme_preemphasis { 4 } else { 2 };
    for s in 0..SIGNALS {
        let (frame, signal_previous) = make_signal::<N>(s, rng);
        for v in 0..variants {
            let (preemphasis, previous) = match v {
                0 => (0, signal_previous),
                1 => (31785, if s == 9 { signal_previous } else { rng.i16() }),
                2 => (-32768, 32767),
                _ => (1, 1),
            };
            // Without the tables (the device, for the built ones) only the
            // draws are made, so the later configurations stay in step.
            let Some((mel, mfcc)) = layers else {
                continue;
            };
            let mut layer = *mel;
            layer.preemphasis = preemphasis;
            let mut bands = [0i16; BANDS];
            let mut coeffs = [0i32; COEFFS];
            layer.forward(&frame, previous, &mut bands, &mut workspace);
            mfcc.forward(&bands, &mut coeffs);
            writeln!(
                out,
                "mel {name} signal={s} pre={} prev={previous} out={} hash={:08x} mfcc={},{} hash={:08x}",
                layer.preemphasis,
                First4(&bands),
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
    out: &mut impl Write,
    rng: &mut Rng,
    cal: &mut impl Calibrate,
    name: &str,
    sample_rate: f64,
    f_min: f64,
    f_max: f64,
    log_floor: i16,
    periodic: bool,
    extreme_preemphasis: bool,
) {
    let Some(t) = cal.tables::<N, BANDS, COEFFS, CAP>(periodic, sample_rate, f_min, f_max) else {
        run_frames::<N, HALF, BINS, BANDS, FRAC, COEFFS>(out, rng, name, None, extreme_preemphasis);
        return;
    };
    let (cos, sin) = cal.twiddles::<N, HALF>();
    let (window, dct, short) = (&t.window, &t.dct, t.short);
    let weights = t.weights.get(..t.used).unwrap_or(&[]);
    let empty = t.band_length.iter().filter(|&&l| l == 0).count();
    writeln!(
        out,
        "tables {name} window={:08x} first={},{},{} used={} short={short} empty={empty} start={:08x} length={:08x} weights={:08x} dct={:08x} dct0={},{}",
        h16(window),
        window[0],
        window[1],
        window[N / 2],
        t.used,
        hu16(&t.band_start),
        hu16(&t.band_length),
        hu16(weights),
        h16(dct.as_flattened()),
        dct[0][0],
        dct[if COEFFS > 1 { 1 } else { 0 }][0]
    )
    .unwrap();
    let mel: QLogMelSpectrogram<'_, N, HALF, BANDS, FRAC> = QLogMelSpectrogram {
        window,
        twiddle_cos: &cos,
        twiddle_sin: &sin,
        band_start: &t.band_start,
        band_length: &t.band_length,
        weights,
        preemphasis: 0,
        log_floor,
    };
    let mfcc = QMfcc { dct };
    run_frames::<N, HALF, BINS, BANDS, FRAC, COEFFS>(
        out,
        rng,
        name,
        Some((&mel, &mfcc)),
        extreme_preemphasis,
    );
}

fn random_tables<
    const N: usize,
    const HALF: usize,
    const BINS: usize,
    const BANDS: usize,
    const FRAC: u32,
    const CAP: usize,
>(
    out: &mut impl Write,
    rng: &mut Rng,
    cal: &mut impl Calibrate,
    name: &str,
) {
    let (cos, sin) = cal.twiddles::<N, HALF>();
    let mut window = [0i16; N];
    for w in window.iter_mut() {
        *w = rng.i16();
    }
    window[0] = if N == 64 { -32768 } else { 100 };
    let mut band_start = [0u16; BANDS];
    let mut band_length = [0u16; BANDS];
    // At most BANDS * BINS weights: CAP.
    let mut weights = [0u16; CAP];
    let mut used = 0;
    for m in 0..BANDS {
        band_start[m] = ((rng.next() >> 16) as usize % BINS) as u16;
        band_length[m] =
            ((rng.next() >> 16) as usize % (BINS - usize::from(band_start[m]) + 1)) as u16;
        for _ in 0..band_length[m] {
            let w = if m == 0 {
                65535
            } else {
                (rng.next() >> 16) as u16
            };
            if let Some(slot) = weights.get_mut(used) {
                *slot = w;
            }
            used += 1;
        }
    }
    let mut dct = [[0i16; BANDS]; BANDS];
    for d in dct.iter_mut().flatten() {
        *d = rng.i16();
    }
    writeln!(
        out,
        "tables {name} random used={} start={:08x} length={:08x}",
        used,
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
        weights: weights.get(..used).unwrap_or(&weights),
        preemphasis: 0,
        log_floor: -32768,
    };
    let mfcc = QMfcc { dct: &dct };
    run_frames::<N, HALF, BINS, BANDS, FRAC, BANDS>(out, rng, name, Some((&mel, &mfcc)), true);
}

/// The whole sweep, in the golden's order.
pub fn run(out: &mut impl Write, cal: &mut impl Calibrate) {
    let rng = &mut Rng(7);
    built::<4, 2, 3, 2, 8, 1, 6>(out, rng, cal, "n4", 8.0, 0.0, 4.0, -32768, true, true);
    built::<64, 32, 33, 10, 8, 5, 66>(
        out, rng, cal, "n64", 8000.0, 20.0, 4000.0, -1500, true, false,
    );
    built::<128, 64, 65, 64, 8, 13, 130>(
        out,
        rng,
        cal,
        "n128-narrow",
        16000.0,
        0.0,
        8000.0,
        -32768,
        true,
        false,
    );
    built::<256, 128, 129, 32, 8, 13, 258>(
        out, rng, cal, "n256", 16000.0, 20.0, 8000.0, -2048, true, true,
    );
    built::<256, 128, 129, 32, 0, 32, 258>(
        out, rng, cal, "n256-q0", 16000.0, 20.0, 8000.0, -128, false, false,
    );
    built::<512, 256, 257, 26, 14, 13, 514>(
        out, rng, cal, "n512-q14", 44100.0, 30.0, 11000.0, -32768, true, false,
    );
    built::<1024, 512, 513, 40, 8, 20, 1026>(
        out, rng, cal, "n1024", 48000.0, 0.0, 24000.0, 32767, true, false,
    );
    random_tables::<16, 8, 9, 5, 8, 45>(out, rng, cal, "rand16");
    random_tables::<64, 32, 33, 7, 14, 231>(out, rng, cal, "rand64");
}
