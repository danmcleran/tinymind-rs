//! Quantized radix-2 FFT (C++ `cpp/qfft1d.hpp`).

/// Saturate an `i32` to `i16` (C++ `qfft_detail::saturateInt16`).
#[inline(always)]
pub(crate) fn saturate_i16(v: i32) -> i16 {
    if v < i32::from(i16::MIN) {
        i16::MIN
    } else if v > i32::from(i16::MAX) {
        i16::MAX
    } else {
        v as i16
    }
}

/// Q1.15 multiply, `(a * b) >> 15` in `i32` (C++ `qfft_detail::q15Mul`). The
/// shift is arithmetic, so the result floors. With both operands from `i16`
/// the product is at most `2^30` and cannot overflow.
#[inline(always)]
pub(crate) fn q15_mul(a: i32, b: i32) -> i32 {
    (a * b) >> 15
}

/// A quantized radix-2 decimation-in-time FFT of length `N`, in place on `i16`
/// real and imaginary buffers (C++ `QFFT1D<N>`).
///
/// For vibration, audio or IMU front ends on cores without a floating-point
/// unit: butterflies are 16-bit with `i32` intermediates, twiddle factors are
/// Q1.15 (scale `1/32768`). The layer only borrows its two twiddle tables, so
/// they can live in flash as `static` or `const` data; the work buffers are
/// the caller's, and the transform needs no other memory beyond a few
/// registers.
///
/// `HALF` is `N / 2`, the length of each twiddle table: entry `k` holds
/// `cos(-2πk/N)` and `sin(-2πk/N)` in Q1.15. Stable Rust cannot write `[i16;
/// N / 2]` for a generic `N`, so the half length is a parameter of its own.
/// `N` must be a power of two of at least 2 and `HALF` must be `N / 2`, or
/// calling any method does not compile. `calibration::build_fft_twiddles`
/// (`std` feature) computes the tables as tinymind (C++) does, so generate them
/// on the host and embed them.
///
/// [`forward`](Self::forward) scales each of its `log2(N)` stages by 1/2, so
/// the output is the discrete Fourier transform divided by `N` and the working
/// values stay within 16 bits. [`inverse`](Self::inverse) does not scale, so
/// it undoes `forward` up to rounding. Each butterfly saturates its results to
/// `i16`.
///
/// ```
/// use tinymind::int8::QFft1D;
///
/// // Twiddles for N = 8: cos and sin of -2πk/8 for k = 0..4, in Q1.15.
/// static COS: [i16; 4] = [32767, 23170, 0, -23170];
/// static SIN: [i16; 4] = [0, -23170, -32768, -23170];
/// static FFT: QFft1D<'static, 8, 4> = QFft1D { twiddle_cos: &COS, twiddle_sin: &SIN };
///
/// // A cosine that completes one cycle over the eight samples.
/// let mut real = [8000, 5657, 0, -5657, -8000, -5657, 0, 5657];
/// let mut imag = [0; 8];
/// FFT.forward(&mut real, &mut imag);
/// // Scaled by 1/8, its energy lands in bins 1 and 7, each with half the
/// // amplitude; each stage's halving floors, so a few values are off by one.
/// assert_eq!(real, [-1, 3999, 0, 0, 0, 0, 0, 4000]);
/// assert_eq!(imag, [0, -1, 0, 0, 0, 0, 0, 0]);
/// let mut power = [0i32; 8];
/// FFT.magnitude_squared(&real, &imag, &mut power);
/// assert_eq!(power[7], 4000 * 4000);
/// ```
///
/// A table of the wrong length does not compile:
///
/// ```compile_fail
/// # use tinymind::int8::QFft1D;
/// static COS: [i16; 8] = [0; 8];
/// static SIN: [i16; 8] = [0; 8];
/// let fft: QFft1D<'_, 8, 8> = QFft1D { twiddle_cos: &COS, twiddle_sin: &SIN };
/// fft.forward(&mut [0; 8], &mut [0; 8]);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct QFft1D<'a, const N: usize, const HALF: usize> {
    /// `twiddle_cos[k]` = `cos(-2πk/N)` in Q1.15.
    pub twiddle_cos: &'a [i16; HALF],
    /// `twiddle_sin[k]` = `sin(-2πk/N)` in Q1.15.
    pub twiddle_sin: &'a [i16; HALF],
}

impl<const N: usize, const HALF: usize> QFft1D<'_, N, HALF> {
    /// `log2(N)`, after checking `N` and `HALF` (the C++ `static_assert`s).
    const STAGES: u32 = {
        assert!(N >= 2, "FFT length must be at least 2");
        assert!(N.is_power_of_two(), "FFT length must be a power of two");
        assert!(HALF == N / 2, "HALF must be N / 2");
        N.trailing_zeros()
    };

    /// Forward transform in place, scaled by `1/N`.
    pub fn forward(&self, real: &mut [i16; N], imag: &mut [i16; N]) {
        self.transform::<true>(real, imag);
    }

    /// Inverse transform in place, unscaled, so it cancels the `1/N` of
    /// [`forward`](Self::forward). Computed as the conjugate of the forward
    /// transform of the conjugate; negating `-32768` wraps to `-32768`, as the
    /// C++ cast does.
    pub fn inverse(&self, real: &mut [i16; N], imag: &mut [i16; N]) {
        for x in imag.iter_mut() {
            *x = x.wrapping_neg();
        }
        self.transform::<false>(real, imag);
        for x in imag.iter_mut() {
            *x = x.wrapping_neg();
        }
    }

    /// Squared magnitude of each bin, `real^2 + imag^2`, in `i32`, into the
    /// caller's `mag_sq` (C++ `magnitudeSquared`). The output is in the square
    /// of the working scale;
    /// a [`Requantizer`](super::Requantizer) can rescale it to an int8 grid.
    /// The only sum that does not fit, `(-32768)^2 + (-32768)^2 = 2^31`,
    /// saturates to `i32::MAX`.
    pub fn magnitude_squared(&self, real: &[i16; N], imag: &[i16; N], mag_sq: &mut [i32; N]) {
        let _ = Self::STAGES;
        for ((o, &r), &m) in mag_sq.iter_mut().zip(real).zip(imag) {
            let (r, m) = (i64::from(r), i64::from(m));
            let sq = r * r + m * m;
            *o = if sq > i64::from(i32::MAX) {
                i32::MAX
            } else {
                sq as i32
            };
        }
    }

    /// The radix-2 stages (C++ `forwardImpl`), with or without the per-stage
    /// halving.
    fn transform<const SCALE: bool>(&self, real: &mut [i16; N], imag: &mut [i16; N]) {
        // The checks on N and HALF (a post-monomorphization error if they fail).
        let _ = Self::STAGES;
        bit_reverse_permute(real, imag);

        // Each stage runs with its half size as a constant, so offsets,
        // strides and trip counts are immediates and more registers stay free
        // for the butterfly: on a Cortex-M0+ this took a 256-point forward
        // transform from 153,698 cycles to 117,640 at opt-level 3, and from
        // 157,939 to 135,233 at "s", for about 1.5 KB more code. Stages that a length N
        // does not have compile to nothing. The list reaches past any array
        // that fits a 32-bit address space.
        macro_rules! stages {
            ($($h:literal)*) => {$(
                if $h < N {
                    self.stage::<SCALE, $h>(real, imag);
                }
            )*};
        }
        stages!(
            1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144
            524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728
            268435456 536870912 1073741824
        );
    }

    /// One stage with half size `H` (C++ `halfSize`). The C++ runs group by
    /// group and loads the twiddle for every butterfly; the butterflies of a
    /// stage are independent, so this runs twiddle by twiddle instead, loading
    /// each pair once per stage. The results are the same. Every index is
    /// already in range; the masks only let the compiler see it, so no bounds
    /// check (and no panic) is emitted. `N` and `HALF` are powers of two.
    #[inline(always)]
    fn stage<const SCALE: bool, const H: usize>(&self, real: &mut [i16; N], imag: &mut [i16; N]) {
        let mask = N - 1;
        let tw_mask = HALF - 1;
        let stride = HALF / H;
        for pair in 0..H {
            let k = (pair * stride) & tw_mask;
            let w_cos = i32::from(self.twiddle_cos[k]);
            let w_sin = i32::from(self.twiddle_sin[k]);
            let mut top = pair;
            while top < N {
                butterfly::<SCALE, N>(real, imag, top & mask, (top + H) & mask, w_cos, w_sin);
                top += 2 * H;
            }
        }
    }
}

/// One radix-2 butterfly on `top` and `bot` with the twiddle `(w_cos, w_sin)`
/// (the body of the C++ `forwardImpl` loop).
#[inline(always)]
fn butterfly<const SCALE: bool, const N: usize>(
    real: &mut [i16; N],
    imag: &mut [i16; N],
    top: usize,
    bot: usize,
    w_cos: i32,
    w_sin: i32,
) {
    let bot_real = i32::from(real[bot]);
    let bot_imag = i32::from(imag[bot]);
    let t_real = q15_mul(w_cos, bot_real) - q15_mul(w_sin, bot_imag);
    let t_imag = q15_mul(w_cos, bot_imag) + q15_mul(w_sin, bot_real);

    let top_real = i32::from(real[top]);
    let top_imag = i32::from(imag[top]);
    if SCALE {
        real[top] = saturate_i16((top_real + t_real) >> 1);
        imag[top] = saturate_i16((top_imag + t_imag) >> 1);
        real[bot] = saturate_i16((top_real - t_real) >> 1);
        imag[bot] = saturate_i16((top_imag - t_imag) >> 1);
    } else {
        real[top] = saturate_i16(top_real + t_real);
        imag[top] = saturate_i16(top_imag + t_imag);
        real[bot] = saturate_i16(top_real - t_real);
        imag[bot] = saturate_i16(top_imag - t_imag);
    }
}

/// Swap each index with its bit reversal (C++ `bitReversalPermute`). The C++
/// reads the reversed index from an `N`-entry table; here `j` steps through
/// the reversed sequence by adding 1 at its top bit and carrying downwards, so
/// the permutation is the same without a table in flash or RAM.
fn bit_reverse_permute<const N: usize>(real: &mut [i16; N], imag: &mut [i16; N]) {
    let mask = N - 1;
    let mut j = 0;
    for i in 0..N {
        if i < j {
            // Not `slice::swap`: when it is not inlined (opt-level z) its
            // bounds checks stay.
            let (a, b) = (i & mask, j & mask);
            (real[a], real[b]) = (real[b], real[a]);
            (imag[a], imag[b]) = (imag[b], imag[a]);
        }
        // j = reverse(reverse(j) + 1).
        let mut bit = N >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reverse(i: usize, bits: u32) -> usize {
        (0..bits).fold(0, |r, b| (r << 1) | ((i >> b) & 1))
    }

    fn check_permutation<const N: usize>() {
        let mut real: [i16; N] = core::array::from_fn(|i| i as i16);
        let mut imag: [i16; N] = core::array::from_fn(|i| !(i as i16));
        bit_reverse_permute(&mut real, &mut imag);
        let bits = N.trailing_zeros();
        for i in 0..N {
            assert_eq!(real[i] as usize, reverse(i, bits), "N={N} i={i}");
            assert_eq!(imag[i], !(reverse(i, bits) as i16));
        }
    }

    #[test]
    fn bit_reversal_matches_the_table() {
        check_permutation::<2>();
        check_permutation::<4>();
        check_permutation::<8>();
        check_permutation::<64>();
        check_permutation::<1024>();
        check_permutation::<4096>();
    }

    #[test]
    fn saturate_and_q15() {
        assert_eq!(saturate_i16(40_000), i16::MAX);
        assert_eq!(saturate_i16(-40_000), i16::MIN);
        assert_eq!(saturate_i16(-32_768), i16::MIN);
        assert_eq!(q15_mul(-32768, -32768), 32768);
        assert_eq!(q15_mul(1, -1), -1);
    }

    #[test]
    fn magnitude_saturates_only_at_both_rails() {
        static T: [i16; 1] = [32767];
        let fft: QFft1D<'_, 2, 1> = QFft1D {
            twiddle_cos: &T,
            twiddle_sin: &[0],
        };
        let mut m = [0; 2];
        fft.magnitude_squared(&[-32768, -32768], &[-32768, 32767], &mut m);
        assert_eq!(m, [i32::MAX, 32768 * 32768 + 32767 * 32767]);
    }
}
