// QLogMelSpectrogram, QMfcc and calibration::build_hann_window,
// build_mel_filterbank and build_dct_matrix against tinymind (C++)
// (cpp/qmel.hpp): frame lengths 4 to 1024, band counts that leave bands
// empty, output Q formats 0, 8 and 14, log floors from -32768 to 32767; ten
// integer test signals with pre-emphasis off and on; then random tables.
// tests/golden/mel/stdout.txt is the output of scripts/cpp/mel_sweep.cpp
// (scripts/compare-with-cpp.sh rebuilds it and checks the golden is
// current); this file computes the same lines.
// The sweep itself is in tests/sweeps/mel.rs, which bench/ also runs on
// Cortex-M.

#[path = "sweeps/mel.rs"]
mod mel;

use tinymind::int8::calibration::{
    build_dct_matrix, build_fft_twiddles, build_hann_window, build_mel_filterbank,
};

struct Host;

impl mel::Calibrate for Host {
    fn twiddles<const N: usize, const HALF: usize>(&mut self) -> ([i16; HALF], [i16; HALF]) {
        build_fft_twiddles::<N, HALF>()
    }

    fn tables<const N: usize, const BANDS: usize, const COEFFS: usize, const CAP: usize>(
        &mut self,
        periodic: bool,
        sample_rate: f64,
        f_min: f64,
        f_max: f64,
    ) -> Option<mel::Tables<N, BANDS, COEFFS, CAP>> {
        let fb = build_mel_filterbank::<N, BANDS, CAP>(sample_rate, f_min, f_max)?;
        Some(mel::Tables {
            window: build_hann_window::<N>(periodic),
            band_start: fb.band_start,
            band_length: fb.band_length,
            weights: fb.weight_storage,
            used: fb.len,
            // The C++ reports 0 from a call with too little room; here that is None.
            short: build_mel_filterbank::<N, BANDS, 1>(sample_rate, f_min, f_max)
                .map_or(0, |f| f.len),
            dct: build_dct_matrix::<BANDS, COEFFS>(),
        })
    }
}

#[test]
fn mel_sweep_matches_cpp() {
    let mut out = String::new();
    mel::run(&mut out, &mut Host);
    let golden = include_str!("golden/mel/stdout.txt");
    for (n, (got, want)) in out.lines().zip(golden.lines()).enumerate() {
        assert_eq!(got, want, "line {}", n + 1);
    }
    assert_eq!(out.lines().count(), golden.lines().count());
}
