//! The quantized FFT on the board: one 256-point forward transform and the
//! magnitudes of its bins, on the FFT sweep's random signal, with the twiddle
//! tables of tinymind (C++). The spectrum's hash must equal the forward hash
//! that tinymind (C++) printed for the same signal (tests/golden/fft/stdout.txt,
//! read by build.rs).

use core::fmt::Write;
use core::hint::black_box;

use tinymind::int8::QFft1D;

use crate::board::{self, Uart};
use crate::{REPS, median};

mod twiddles {
    include!(concat!(env!("OUT_DIR"), "/fft_twiddles.rs"));
}

const N: usize = 256;

static FFT: QFft1D<'static, N, { N / 2 }> = QFft1D {
    twiddle_cos: &twiddles::COS,
    twiddle_sin: &twiddles::SIN,
};

/// The FFT sweep's `random` signal (tests/sweeps/fft.rs): interleaved draws
/// for the real and imaginary parts from its generator, seeded with `N`.
fn random_signal() -> ([i16; N], [i16; N]) {
    let mut state = N as u32;
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 16) as u16 as i16
    };
    let (mut re, mut im) = ([0i16; N], [0i16; N]);
    for (r, i) in re.iter_mut().zip(im.iter_mut()) {
        *r = next();
        *i = next();
    }
    (re, im)
}

/// FNV-1a over the bytes of `re`, then `im`, little-endian (the sweep's hash).
fn hash_spectrum(re: &[i16; N], im: &[i16; N]) -> u32 {
    re.iter()
        .chain(im)
        .flat_map(|v| v.to_le_bytes())
        .fold(0x811C_9DC5, |h, b| {
            (h ^ u32::from(b)).wrapping_mul(0x0100_0193)
        })
}

pub fn run(out: &mut Uart) {
    let _ = writeln!(out, "name=fft\nfft_n={N}");
    let (x, y) = random_signal();

    let (mut re, mut im) = (x, y);
    FFT.forward(&mut re, &mut im);
    let hash = hash_spectrum(&re, &im);
    let expect = twiddles::FORWARD_HASH;
    let _ = writeln!(
        out,
        "hash=0x{hash:08X}\nexpect_hash=0x{expect:08X}\nhash_match={}",
        u32::from(hash == expect)
    );

    let mut t = [0u32; REPS];
    for slot in t.iter_mut() {
        let (mut re, mut im) = black_box((x, y));
        let a = board::cycles();
        FFT.forward(&mut re, &mut im);
        *slot = board::cycles().wrapping_sub(a);
        black_box((&re, &im));
    }
    let (min, med, max) = median(t);
    let _ = writeln!(
        out,
        "forward_min={min}\nforward_med={med}\nforward_max={max}"
    );

    let mut mag = [0i32; N];
    for slot in t.iter_mut() {
        let (r, i) = black_box((&re, &im));
        let a = board::cycles();
        FFT.magnitude_squared(r, i, &mut mag);
        *slot = board::cycles().wrapping_sub(a);
        black_box(&mag);
    }
    let (min, med, max) = median(t);
    let _ = writeln!(
        out,
        "magnitude_min={min}\nmagnitude_med={med}\nmagnitude_max={max}"
    );
}
