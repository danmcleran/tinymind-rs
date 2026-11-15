// QFft1D and calibration::build_fft_twiddles against tinymind's C++ (QFFT1D,
// buildQFFTTwiddles) for lengths 2 to 1024: the twiddle tables, then nine test
// signals (impulses, DC, tones, random, full-scale rails) through forward,
// magnitude_squared, inverse and the round trip, on the built tables and on
// random ones. tests/golden/fft/stdout.txt is the output of
// scripts/cpp/fft_sweep.cpp (scripts/compare-with-cpp.sh rebuilds it and checks
// the golden is current).

// The sweep itself is in tests/sweeps/fft.rs, which bench/ also runs on
// Cortex-M.

#[path = "sweeps/fft.rs"]
mod fft;

use tinymind::int8::calibration::build_fft_twiddles;

struct Build;

impl fft::Twiddles for Build {
    fn twiddles<const N: usize, const HALF: usize>(&mut self) -> ([i16; HALF], [i16; HALF]) {
        build_fft_twiddles::<N, HALF>()
    }
}

#[test]
fn fft_sweep_matches_cpp() {
    let mut out = String::new();
    fft::report(&mut out, &mut Build);
    let golden = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/golden/fft/stdout.txt"
    ))
    .unwrap();
    for (n, (got, want)) in out.lines().zip(golden.lines()).enumerate() {
        assert_eq!(got, want, "line {}", n + 1);
    }
    assert_eq!(out.lines().count(), golden.lines().count());
}

/// The real FFT against an exact DFT divided by N, and against `QFft1D` on the
/// same input with a zeroed imaginary part, for N = 4 to 256 (the bounds
/// tinymind (C++)'s unit tests use): every bin of noise, tones and full-scale
/// rail patterns within `2 * log2(N) + 1` LSB of the exact value and within
/// `log2(N)` LSB of `QFft1D`. Also: DC, Nyquist and an impulse land in the
/// right bins.
#[test]
fn rfft_is_close_to_the_exact_dft() {
    use tinymind::int8::{QFft1D, QRFft1D};

    fn check<const N: usize, const HALF: usize, const BINS: usize>() {
        let (c, s) = build_fft_twiddles::<N, HALF>();
        let rfft: QRFft1D<'_, N, HALF> = QRFft1D {
            twiddle_cos: &c,
            twiddle_sin: &s,
        };
        let fft: QFft1D<'_, N, HALF> = QFft1D {
            twiddle_cos: &c,
            twiddle_sin: &s,
        };
        let bits = N.trailing_zeros() as f64;
        let mut state = 12345u32 ^ N as u32;
        let mut draw = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 16) as u16 as i16
        };
        let mut inputs: Vec<[i16; N]> = Vec::new();
        for _ in 0..8 {
            inputs.push(core::array::from_fn(|_| draw()));
            inputs.push(core::array::from_fn(|_| {
                if draw() & 1 != 0 { 32767 } else { -32768 }
            }));
        }
        for k in [1, N / 4, N / 2 - 1] {
            inputs.push(core::array::from_fn(|i| {
                let ph = core::f64::consts::TAU * (k * i) as f64 / N as f64;
                (30000.0 * ph.cos()).round() as i16
            }));
        }
        inputs.push([-32768; N]);
        inputs.push([32767; N]);
        inputs.push(core::array::from_fn(|i| {
            if i % 2 == 0 { 32767 } else { -32768 }
        }));
        for x in &inputs {
            let mut re = [0i16; BINS];
            let mut im = [0i16; BINS];
            let before = *x;
            rfft.forward(x, &mut re, &mut im);
            assert_eq!(*x, before);
            assert_eq!((im[0], im[HALF]), (0, 0), "N={N}");
            let (mut cre, mut cim) = (*x, [0i16; N]);
            fft.forward(&mut cre, &mut cim);
            for k in 0..BINS {
                let (mut er, mut ei) = (0.0, 0.0);
                for (n, &v) in x.iter().enumerate() {
                    let ph = -core::f64::consts::TAU * ((k * n) % N) as f64 / N as f64;
                    er += f64::from(v) * ph.cos();
                    ei += f64::from(v) * ph.sin();
                }
                let (er, ei) = (er / N as f64, ei / N as f64);
                let dft_err = (f64::from(re[k]) - er)
                    .abs()
                    .max((f64::from(im[k]) - ei).abs());
                assert!(
                    dft_err <= 2.0 * bits + 1.0,
                    "N={N} bin {k}: {dft_err} LSB from the DFT"
                );
                let fft_err = (i32::from(re[k]) - i32::from(cre[k]))
                    .abs()
                    .max((i32::from(im[k]) - i32::from(cim[k])).abs());
                assert!(
                    f64::from(fft_err) <= bits,
                    "N={N} bin {k}: {fft_err} LSB from QFft1D"
                );
            }
        }

        // An impulse is flat, DC lands in bin 0 and the alternating rails in
        // the Nyquist bin.
        let mut re = [0i16; BINS];
        let mut im = [0i16; BINS];
        let mut impulse = [0i16; N];
        impulse[0] = 16384;
        rfft.forward(&impulse, &mut re, &mut im);
        let flat = 16384 / N as i32;
        assert!(
            re.iter()
                .all(|&r| (i32::from(r) - flat).abs() <= bits as i32),
            "N={N}"
        );
        rfft.forward(&[-32768; N], &mut re, &mut im);
        assert_eq!(re[0], -32768, "N={N}");
        assert!(
            re[1..].iter().chain(&im).all(|&v| v.abs() <= bits as i16),
            "N={N}"
        );
        let alt: [i16; N] = core::array::from_fn(|i| if i % 2 == 0 { 20000 } else { -20000 });
        rfft.forward(&alt, &mut re, &mut im);
        assert!((i32::from(re[HALF]) - 20000).abs() <= bits as i32, "N={N}");
        assert!(
            re[..HALF]
                .iter()
                .chain(&im)
                .all(|&v| v.abs() <= bits as i16),
            "N={N}"
        );
    }
    check::<4, 2, 3>();
    check::<8, 4, 5>();
    check::<16, 8, 9>();
    check::<32, 16, 17>();
    check::<64, 32, 33>();
    check::<128, 64, 65>();
    check::<256, 128, 129>();
}
