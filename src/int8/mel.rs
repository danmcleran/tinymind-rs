//! Fixed-point log-mel spectrogram and MFCC front end for audio, vibration
//! and IMU frames, ported from tinymind (C++) `cpp/qmel.hpp`. Integer
//! arithmetic only, on [`QRFft1D`].
//!
//! One frame of `N` `i16` samples becomes `BANDS` natural-log mel energies:
//!
//! 1. pre-emphasis: `y[n] = x[n] - a * x[n - 1]` (`a` in Q1.15, 0 = off)
//! 2. window: `z[n] = y[n] * w[n]` (`w` in Q1.15)
//! 3. block scaling: `u[n] = z[n] * 2^-r`, `r` chosen per frame so the
//!    largest `|u[n]|` lies in `[2^14, 2^15)`; steps 1 and 2 are exact in 64
//!    bits, so this is the only rounding before the FFT
//! 4. real FFT: `X = QRFft1D<N>(u)`, scaled by `1/N`
//! 5. mel energies: `E[m] = sum_k W[m][k] * |X[k]|^2` (`W` in Q0.16, 64-bit sum)
//! 6. log: `log_mel[m] = ln(E[m])` with the scaling of steps 2 to 5 removed,
//!    in Q(`LOG_FRAC_BITS`)
//!
//! The block scaling keeps a quiet frame from vanishing into the 16-bit FFT's
//! rounding: every frame enters the FFT at full scale, and the `2^-r` it was
//! scaled by comes back out as an exact offset of the base-2 logarithm. The
//! result is the float definition
//!
//! ```text
//! log_mel[m] = ln( sum_k W[m][k] * |DFT(y * w / 32768)[k]|^2 )
//! ```
//!
//! the natural log of the mel-weighted power spectrum of the windowed frame,
//! samples read as `i16 / 32768`. This is torchaudio's `MelSpectrogram`
//! (power 2, HTK mel scale, no filter normalization, no centering) followed by
//! a natural log, and
//! [`calibration::build_mel_filterbank`](super::calibration::build_mel_filterbank)
//! builds torchaudio's filters.

use super::QRFft1D;

/// `log2(1 + i / 32)` in Q2.30 for `i` in `0..=32`. Linear interpolation
/// between entries is within 1.8e-4 of `log2` everywhere, a twentieth of the
/// Q8 output's step once converted to a natural log.
const LOG2_TABLE: [u32; 33] = [
    0, 47667823, 93912511, 138816582, 182455581, 224898839, 266210141, 306448299, 345667660,
    383918542, 421247625, 457698295, 493310944, 528123241, 562170370, 595485245, 628098702,
    660039669, 691335320, 722011213, 752091421, 781598637, 810554283, 838978604, 866890747,
    894308843, 921250079, 947730758, 973766362, 999371606, 1024560487, 1049346328, 1073741824,
];

/// `ln(2)` in Q1.31.
const LN2_Q31: i64 = 1_488_522_236;

/// `log2(x)` in Q.30 for `x > 0` (0 gives 0): the integer part from the
/// highest set bit, the fraction from [`LOG2_TABLE`] interpolated on the 32
/// bits below the 5 that select the entry.
pub(crate) fn log2_q30(x: u64) -> i64 {
    if x == 0 {
        return 0;
    }
    let msb = 63 - x.leading_zeros();
    let normalized = x << (63 - msb);
    let index = ((normalized >> 58) & 31) as usize;
    let remainder = (normalized >> 26) & 0xFFFF_FFFF;
    let low = LOG2_TABLE[index];
    let high = LOG2_TABLE.get(index + 1).copied().unwrap_or(1 << 30);
    let fraction = low.wrapping_add(((u64::from(high.wrapping_sub(low)) * remainder) >> 32) as u32);
    (i64::from(msb) << 30) + i64::from(fraction)
}

/// `ln(2^(log2_q30 / 2^30))` in Q(`FRAC`), rounded to nearest; `FRAC` in
/// `0..=14`.
pub(crate) fn natural_log_from_log2_q30<const FRAC: u32>(log2_q30: i64) -> i32 {
    let log2_q16 = log2_q30 >> 14;
    let shift = 16 + 31 - FRAC;
    ((log2_q16 * LN2_Q31 + (1i64 << (shift - 1))) >> shift) as i32
}

/// Caller-owned scratch for one [`QLogMelSpectrogram::forward`] call: the
/// block-scaled frame and the FFT's `BINS = N / 2 + 1` bins (C++
/// `QMelWorkspace`). Its contents need not be kept between frames. `BINS` is
/// a parameter because stable Rust cannot compute `N / 2 + 1` from `N`; a
/// wrong value does not compile.
#[derive(Clone, Debug)]
pub struct QMelWorkspace<const N: usize, const BINS: usize> {
    /// The pre-emphasized, windowed, block-scaled frame.
    pub frame: [i16; N],
    /// Real parts of the FFT's bins.
    pub real: [i16; BINS],
    /// Imaginary parts of the FFT's bins.
    pub imag: [i16; BINS],
}

impl<const N: usize, const BINS: usize> QMelWorkspace<N, BINS> {
    /// A zeroed workspace.
    pub const fn new() -> Self {
        QMelWorkspace {
            frame: [0; N],
            real: [0; BINS],
            imag: [0; BINS],
        }
    }
}

impl<const N: usize, const BINS: usize> Default for QMelWorkspace<N, BINS> {
    fn default() -> Self {
        Self::new()
    }
}

/// Log-mel spectrogram of one frame (C++ `QLogMelSpectrogram`; see the module
/// documentation).
///
/// - `N`: frame and FFT length, a power of two, at least 4
/// - `HALF`: `N / 2`, the twiddle tables' length (stable Rust cannot compute
///   it from `N`)
/// - `BANDS`: mel bands
/// - `LOG_FRAC_BITS`: fractional bits of the `i16` output, `0..=14` (default
///   8: steps of 1/256 nat, range -128 to +128)
///
/// A band whose energy is 0 (a silent frame, or a band too narrow to cover
/// any FFT bin) and any band below `log_floor` reads `log_floor`, which plays
/// the role of the epsilon in `log(E + eps)`. Every table is built on the
/// host (`calibration::build_hann_window`, `build_fft_twiddles`,
/// `build_mel_filterbank`) and can live in flash. A band that reaches past
/// the last bin or past the end of `weights` (which the builder never
/// produces) sums only the bins and weights that exist.
///
/// ```
/// use tinymind::int8::{QLogMelSpectrogram, QMelWorkspace};
///
/// // Four-point frames, one band covering bin 1 with weight 1.0 (65535).
/// static WINDOW: [i16; 4] = [32767; 4];
/// static COS: [i16; 2] = [32767, 0];
/// static SIN: [i16; 2] = [0, -32768];
/// let mel: QLogMelSpectrogram<'_, 4, 2, 1> = QLogMelSpectrogram {
///     window: &WINDOW,
///     twiddle_cos: &COS,
///     twiddle_sin: &SIN,
///     band_start: &[1],
///     band_length: &[1],
///     weights: &[65535],
///     preemphasis: 0,
///     log_floor: -32768,
/// };
/// let mut ws = QMelWorkspace::<4, 3>::new();
/// let mut out = [0i16; 1];
/// mel.forward(&[0; 4], 0, &mut out, &mut ws);
/// assert_eq!(out, [-32768]); // silence reads the floor
/// mel.forward(&[16384, 0, -16384, 0], 0, &mut out, &mut ws);
/// // |DFT(x / 32768)[1]| = 1: ln(1) = 0 (within the window's 1/32768).
/// assert!(out[0].abs() <= 1);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct QLogMelSpectrogram<
    'a,
    const N: usize,
    const HALF: usize,
    const BANDS: usize,
    const LOG_FRAC_BITS: u32 = 8,
> {
    /// `N` window values in Q1.15 (`build_hann_window`).
    pub window: &'a [i16; N],
    /// `cos(-2πk/N)` in Q1.15 (`build_fft_twiddles`).
    pub twiddle_cos: &'a [i16; HALF],
    /// `sin(-2πk/N)` in Q1.15 (`build_fft_twiddles`).
    pub twiddle_sin: &'a [i16; HALF],
    /// The first FFT bin of each band.
    pub band_start: &'a [u16; BANDS],
    /// How many bins each band covers.
    pub band_length: &'a [u16; BANDS],
    /// The bands' Q0.16 weights, band after band: `band_length` entries each.
    pub weights: &'a [u16],
    /// Pre-emphasis coefficient in Q1.15, 0 for none (0.97 is 31785).
    pub preemphasis: i16,
    /// The smallest output, in Q(`LOG_FRAC_BITS`).
    pub log_floor: i16,
}

impl<const N: usize, const HALF: usize, const BANDS: usize, const LOG_FRAC_BITS: u32>
    QLogMelSpectrogram<'_, N, HALF, BANDS, LOG_FRAC_BITS>
{
    /// Checks the C++ `static_assert`s and `HALF`, `BINS`.
    const fn check<const BINS: usize>() {
        assert!(N >= 4, "frame length must be at least 4");
        assert!(N.is_power_of_two(), "frame length must be a power of two");
        assert!(HALF == N / 2, "HALF must be N / 2");
        assert!(BINS == HALF + 1, "the workspace must hold N / 2 + 1 bins");
        assert!(BINS <= 65535, "bins must fit the u16 band tables");
        assert!(BANDS > 0, "at least one mel band");
        assert!(LOG_FRAC_BITS <= 14, "LOG_FRAC_BITS must be in 0..=14");
    }

    /// The `BANDS` log-mel energies of `frame` into `log_mel`.
    /// `previous_sample` is the sample before `frame[0]` (the last sample of
    /// the previous hop's input, or 0), used only by pre-emphasis.
    pub fn forward<const BINS: usize>(
        &self,
        frame: &[i16; N],
        previous_sample: i16,
        log_mel: &mut [i16; BANDS],
        workspace: &mut QMelWorkspace<N, BINS>,
    ) {
        const { Self::check::<BINS>() };
        let a = i64::from(self.preemphasis);
        // y in Q15 (x * 2^15 - a * x_prev, |y| <= 2^31) and z = y * w in 64
        // bits (|z| < 2^46): both exact.
        let windowed =
            |x: i16, prev: i16, w: i16| (i64::from(x) * 32768 - a * i64::from(prev)) * i64::from(w);

        // Pass 1: the largest windowed sample, which sets the block scale.
        let mut prev = previous_sample;
        let mut max_abs = 0u64;
        for (&x, &w) in frame.iter().zip(self.window) {
            max_abs = max_abs.max(windowed(x, prev, w).unsigned_abs());
            prev = x;
        }

        let floor = i32::from(self.log_floor);
        if max_abs == 0 {
            log_mel.fill(self.log_floor);
            return;
        }

        // Pass 2: z[n] shifted right by r so that the largest |u| is in
        // [2^14, 2^15). r is in -14..=31.
        let shift = (63 - max_abs.leading_zeros() as i32) - 14;
        let mut prev = previous_sample;
        for ((u, &x), &w) in workspace.frame.iter_mut().zip(frame).zip(self.window) {
            let z = windowed(x, prev, w);
            prev = x;
            *u = if shift >= 0 {
                (z >> shift) as i16
            } else {
                (z << -shift) as i16
            };
        }

        let fft: QRFft1D<'_, N, HALF> = QRFft1D {
            twiddle_cos: self.twiddle_cos,
            twiddle_sin: self.twiddle_sin,
        };
        fft.forward(&workspace.frame, &mut workspace.real, &mut workspace.imag);

        // In float terms u = (y * w / 32768) * 2^(45 - r), so
        // E_int = sum W * |X|^2 = E * 2^(16 + 90 - 2r) / N^2 and
        // log2(E) = log2(E_int) - 106 + 2r + 2 log2(N).
        let offset_q30 = i64::from(2 * N.trailing_zeros() as i32 - 106 + 2 * shift) << 30;
        let mut weights = self.weights;
        for ((out, &start), &length) in log_mel
            .iter_mut()
            .zip(self.band_start)
            .zip(self.band_length)
        {
            let start = usize::from(start);
            let length = usize::from(length);
            let real = workspace.real.get(start..).unwrap_or(&[]);
            let imag = workspace.imag.get(start..).unwrap_or(&[]);
            let mut energy = 0u64;
            for ((&w, &re), &im) in weights.iter().zip(real).zip(imag).take(length) {
                let (re, im) = (i32::from(re), i32::from(im));
                // Each square is at most 2^30.
                let power = ((re * re) as u32).wrapping_add((im * im) as u32);
                energy = energy.wrapping_add(u64::from(w) * u64::from(power));
            }
            weights = weights.get(length..).unwrap_or(&[]);

            let mut value = floor;
            if energy != 0 {
                value = natural_log_from_log2_q30::<LOG_FRAC_BITS>(log2_q30(energy) + offset_q30)
                    .max(floor)
                    .min(32767);
            }
            *out = value as i16;
        }
    }
}

/// MFCCs from log-mel bands (C++ `QMfcc`): an orthonormal DCT-II, `COEFFS`
/// coefficients of `BANDS` bands, in the log-mel input's Q format, as `i32`
/// because coefficient 0 is `sqrt(BANDS)` times the mean log energy.
/// `calibration::build_dct_matrix` builds the table (torchaudio's `create_dct`
/// with `norm="ortho"`).
///
/// ```
/// use tinymind::int8::QMfcc;
///
/// // Two bands, one coefficient: (a + b) / sqrt(2) with 1/sqrt(2) in Q1.15.
/// let mfcc: QMfcc<'_, 2, 1> = QMfcc { dct: &[[23170, 23170]] };
/// let mut out = [0i32; 1];
/// mfcc.forward(&[256, 256], &mut out);
/// assert_eq!(out, [362]); // 512 / sqrt(2) = 362.04
/// ```
#[derive(Clone, Copy, Debug)]
pub struct QMfcc<'a, const BANDS: usize, const COEFFS: usize> {
    /// `COEFFS` rows of `BANDS` Q1.15 values.
    pub dct: &'a [[i16; BANDS]; COEFFS],
}

impl<const BANDS: usize, const COEFFS: usize> QMfcc<'_, BANDS, COEFFS> {
    /// The `COEFFS` MFCCs of `log_mel` into `mfcc`, rounded to nearest.
    pub fn forward(&self, log_mel: &[i16; BANDS], mfcc: &mut [i32; COEFFS]) {
        const { assert!(COEFFS > 0 && COEFFS <= BANDS, "1 to BANDS coefficients") };
        for (out, row) in mfcc.iter_mut().zip(self.dct) {
            let mut acc = 0i64;
            for (&d, &v) in row.iter().zip(log_mel) {
                acc = acc.wrapping_add(i64::from(d) * i64::from(v));
            }
            *out = (acc.wrapping_add(1 << 14) >> 15) as i32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log2_is_close() {
        for &x in &[1u64, 2, 3, 5, 1000, 123_456_789, u64::MAX, 1 << 63] {
            let got = log2_q30(x) as f64 / (1u64 << 30) as f64;
            assert!((got - (x as f64).log2()).abs() < 2e-4, "x={x}: {got}");
        }
        assert_eq!(log2_q30(1), 0);
        assert_eq!(log2_q30(1 << 40), 40 << 30);
    }

    #[test]
    fn natural_log_rounds() {
        // ln(2^1) = 0.6931 nat: 177.4 in Q8.
        assert_eq!(natural_log_from_log2_q30::<8>(1 << 30), 177);
        assert_eq!(natural_log_from_log2_q30::<8>(-(1 << 30)), -177);
        assert_eq!(natural_log_from_log2_q30::<0>(10 << 30), 7); // 6.93
    }
}
