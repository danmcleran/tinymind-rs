//! The log-mel example's program, shared by `examples/logmel_vibration.rs` and
//! `tests/logmel_golden.rs` (which runs it in-process against the C++
//! goldens). Port of tinymind (C++) `examples/logmel_vibration/logmel_vibration.cpp`.
//!
//! A synthetic machine-vibration recording (3,200 samples per second, 12
//! seconds, int16: a motor at 29.5 Hz with four harmonics and noise; from
//! halfway, a bearing fault ringing a 900 Hz resonance; the level ramping from
//! -50 dBFS to -6 dBFS) goes frame by frame (256 samples, hop 128) through
//! [`QLogMelSpectrogram`] (Hann window, 32 HTK mel bands from 5 to 1,600 Hz)
//! and [`QMfcc`] (13 coefficients), and is compared with the same definition
//! computed in `f64`: `ln(sum_k W[m][k] |DFT(x / 32768 * hann)[k]|^2)`, then
//! the orthonormal DCT.

use std::f64::consts::PI;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use tinymind::int8::calibration::{
    build_dct_matrix, build_fft_twiddles, build_hann_window, build_mel_filterbank,
    mel_filter_weight,
};
use tinymind::int8::{QLogMelSpectrogram, QMelWorkspace, QMfcc};

#[allow(dead_code)]
#[path = "std_random.rs"]
mod std_random;
use std_random::{Mt19937, NormalF64};

const SAMPLE_RATE: f64 = 3200.0;
const SECONDS: f64 = 12.0;
const N: usize = 256;
const HALF: usize = N / 2;
const BINS: usize = N / 2 + 1;
const HOP: usize = 128;
const BANDS: usize = 32;
const COEFFS: usize = 13;
const F_MIN: f64 = 5.0;
const F_MAX: f64 = 1600.0;
const FRAC_BITS: u32 = 8;

/// The recording, with each sample's level in dBFS and fault flag.
fn make_recording() -> (Vec<i16>, Vec<f64>, Vec<i32>) {
    let total = (SAMPLE_RATE * SECONDS) as usize;
    let mut rng = Mt19937::new(7);
    let mut gauss = NormalF64::new(0.0, 1.0);
    let mut x = vec![0i16; total];
    let mut level_db = vec![0f64; total];
    let mut fault = vec![0i32; total];
    let mut ring = 0.0f64;
    let mut ring_velocity = 0.0f64;
    let omega = 2.0 * PI * 900.0 / SAMPLE_RATE;
    let damping = 0.985;
    for n in 0..total {
        let t = n as f64 / SAMPLE_RATE;
        let mut s = 0.0;
        for h in 1..=5 {
            let h = f64::from(h);
            s += (2.0 * PI * 29.5 * h * t + 0.7 * h).sin() / h;
        }
        s += 0.05 * gauss.sample(&mut rng);
        fault[n] = i32::from(t >= SECONDS / 2.0);
        if fault[n] != 0 {
            // A damped 900 Hz oscillator kicked at 107 Hz.
            let period = SAMPLE_RATE / 107.0;
            if (n as f64) % period < 1.0 {
                ring_velocity += 1.0;
            }
            let next = 2.0 * omega.cos() * damping * ring - damping * damping * ring_velocity;
            ring_velocity = ring;
            ring = next;
            s += 0.6 * ring;
        }
        level_db[n] = -50.0 + 44.0 * t / SECONDS;
        let amplitude = 10f64.powf(level_db[n] / 20.0) * 32767.0 / 2.6;
        x[n] = (s * amplitude).round().clamp(-32768.0, 32767.0) as i16;
    }
    (x, level_db, fault)
}

/// Runs the example: writes `logmel_vibration.csv`, `logmel_vibration_mfcc.csv`
/// and `logmel_vibration_error.csv` into `out_dir` and the C++ example's
/// report to `stdout`.
pub fn run(out_dir: &Path, stdout: &mut impl Write) -> io::Result<()> {
    let (x, level_db, fault) = make_recording();

    let window = build_hann_window::<N>(true);
    let (cos, sin) = build_fft_twiddles::<N, HALF>();
    let fb = build_mel_filterbank::<N, BANDS, { 2 * BINS }>(SAMPLE_RATE, F_MIN, F_MAX)
        .expect("2 * BINS weights always suffice");
    let dct = build_dct_matrix::<BANDS, COEFFS>();

    // ln(E) floor of -20 (Q8: -5120); no band of this recording is that quiet.
    let mel: QLogMelSpectrogram<'_, N, HALF, BANDS> = QLogMelSpectrogram {
        window: &window,
        twiddle_cos: &cos,
        twiddle_sin: &sin,
        band_start: &fb.band_start,
        band_length: &fb.band_length,
        weights: fb.weights(),
        preemphasis: 0,
        log_floor: -5120,
    };
    let mfcc = QMfcc { dct: &dct };
    let mut workspace = QMelWorkspace::<N, BINS>::new();

    let mut filter = vec![[0f64; BINS]; BANDS];
    for (m, row) in filter.iter_mut().enumerate() {
        for (k, w) in row.iter_mut().enumerate() {
            *w = mel_filter_weight(N, SAMPLE_RATE, BANDS, F_MIN, F_MAX, m, k);
        }
    }

    let mut bands_csv = BufWriter::new(File::create(out_dir.join("logmel_vibration.csv"))?);
    let mut mfcc_csv = BufWriter::new(File::create(out_dir.join("logmel_vibration_mfcc.csv"))?);
    write!(bands_csv, "frame,time_s,fault,level_dbfs")?;
    for m in 0..BANDS {
        write!(bands_csv, ",fixed_{m}")?;
    }
    for m in 0..BANDS {
        write!(bands_csv, ",float_{m}")?;
    }
    writeln!(bands_csv)?;
    write!(mfcc_csv, "frame,time_s")?;
    for c in 0..COEFFS {
        write!(mfcc_csv, ",fixed_c{c},float_c{c}")?;
    }
    writeln!(mfcc_csv)?;

    // |error| buckets by level below the frame's loudest band: 0-10, ..., 40+ dB.
    let mut buckets: [Vec<f64>; 5] = Default::default();
    let mut worst_mfcc_error = 0.0f64;
    let mut frames = 0usize;
    let mut start = 0usize;
    while start + N <= x.len() {
        let previous = if start > 0 { x[start - 1] } else { 0 };
        let frame: &[i16; N] = x[start..start + N].try_into().unwrap();
        let mut log_mel = [0i16; BANDS];
        mel.forward(frame, previous, &mut log_mel, &mut workspace);
        let mut coeffs = [0i32; COEFFS];
        mfcc.forward(&log_mel, &mut coeffs);

        let mut y = [0f64; N];
        for (n, v) in y.iter_mut().enumerate() {
            *v = f64::from(frame[n]) / 32768.0
                * (0.5 - 0.5 * (2.0 * PI * n as f64 / N as f64).cos());
        }
        let mut power = [0f64; BINS];
        for (k, p) in power.iter_mut().enumerate() {
            let mut re = 0.0;
            let mut im = 0.0;
            for (n, &v) in y.iter().enumerate() {
                re += v * (2.0 * PI * k as f64 * n as f64 / N as f64).cos();
                im -= v * (2.0 * PI * k as f64 * n as f64 / N as f64).sin();
            }
            *p = re * re + im * im;
        }
        let mut reference = [0f64; BANDS];
        for (r, row) in reference.iter_mut().zip(&filter) {
            let mut e = 0.0;
            for (&w, &p) in row.iter().zip(&power) {
                e += w * p;
            }
            *r = e.ln();
        }
        // std::max_element: the first of the largest.
        let peak = reference
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, |a, b| if b > a { b } else { a });
        for (m, &r) in reference.iter().enumerate() {
            let below = (peak - r) * 10.0 / 10f64.ln();
            let bucket = ((below / 10.0) as usize).min(4);
            buckets[bucket].push((f64::from(log_mel[m]) / 256.0 - r).abs());
        }

        let centre = start + N / 2;
        write!(
            bands_csv,
            "{frames},{:.4},{},{:.2}",
            centre as f64 / SAMPLE_RATE,
            fault[centre],
            level_db[centre]
        )?;
        for &v in &log_mel {
            write!(bands_csv, ",{:.5}", f64::from(v) / 256.0)?;
        }
        for &r in &reference {
            write!(bands_csv, ",{r:.5}")?;
        }
        writeln!(bands_csv)?;

        write!(mfcc_csv, "{frames},{:.4}", centre as f64 / SAMPLE_RATE)?;
        for (c, &coeff) in coeffs.iter().enumerate() {
            let mut r = 0.0;
            for (m, &reference_m) in reference.iter().enumerate() {
                r += (if c == 0 { 1.0 } else { 2.0 } / BANDS as f64).sqrt()
                    * (PI / BANDS as f64 * (m as f64 + 0.5) * c as f64).cos()
                    * reference_m;
            }
            let got = f64::from(coeff) / f64::from(1u32 << FRAC_BITS);
            worst_mfcc_error = worst_mfcc_error.max((got - r).abs());
            write!(mfcc_csv, ",{got:.5},{r:.5}")?;
        }
        writeln!(mfcc_csv)?;

        start += HOP;
        frames += 1;
    }
    bands_csv.flush()?;
    mfcc_csv.flush()?;

    let mut error_csv = BufWriter::new(File::create(out_dir.join("logmel_vibration_error.csv"))?);
    let labels = ["0-10", "10-20", "20-30", "30-40", "40+"];
    writeln!(
        error_csv,
        "db_below_peak,bands,mean_abs_error,p99_abs_error,max_abs_error"
    )?;
    writeln!(
        stdout,
        "frames: {frames} of {N} samples (hop {HOP}), {} filter weights, workspace {} bytes",
        fb.len,
        size_of::<QMelWorkspace<N, BINS>>()
    )?;
    writeln!(
        stdout,
        "|log-mel error| in nats, by band level below the frame's loudest band:"
    )?;
    for (e, label) in buckets.iter_mut().zip(labels) {
        if e.is_empty() {
            continue;
        }
        e.sort_by(f64::total_cmp);
        let mut mean = 0.0;
        for &v in e.iter() {
            mean += v;
        }
        mean /= e.len() as f64;
        let p99 = e[(0.99 * (e.len() - 1) as f64) as usize];
        let max = e[e.len() - 1];
        writeln!(error_csv, "{label},{},{mean:.5},{p99:.5},{max:.5}", e.len())?;
        writeln!(
            stdout,
            "  {label:>5} dB: {:>5} bands  mean {mean:.4}  p99 {p99:.4}  max {max:.4}",
            e.len()
        )?;
    }
    error_csv.flush()?;
    writeln!(
        stdout,
        "worst MFCC error against the float DCT of the float log-mel: {worst_mfcc_error:.4}"
    )?;
    writeln!(
        stdout,
        "wrote logmel_vibration.csv, logmel_vibration_mfcc.csv and logmel_vibration_error.csv"
    )?;
    Ok(())
}
