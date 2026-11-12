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
