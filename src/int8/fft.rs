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

/// Whether the forward transform checks once per butterfly that its two
/// products fit in `i16`, and then skips the four saturations, which cannot
/// change anything then. Both forms give the same results. On a Cortex-M0+
/// the check pays (a 256-point transform took 91,430 cycles instead of
/// 117,640 at opt-level 3): Thumb-1 has no saturating instruction, and each
/// saturation loads its bounds from the literal pool. On a Cortex-M4 it costs
/// (48,363 against 41,991), since `ssat` saturates in one instruction. Stable
/// Rust names no instruction-set features in `cfg`, so this uses the closest
/// available sign: ARMv6-M has no 32-bit atomic instructions, Armv7-M does.
const SKIP_SATURATION: bool = cfg!(not(target_has_atomic = "32"));

/// True when `v` is in the `i16` range: `v >> 15` is -1 or 0 exactly then.
/// A shift and a compare with small immediates, no 32-bit constants.
#[inline(always)]
pub(crate) fn fits_i16(v: i32) -> bool {
    ((v >> 15) + 1) as u32 <= 1
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
        self.transform::<true, SKIP_SATURATION>(real, imag);
    }

    /// Inverse transform in place, unscaled, so it cancels the `1/N` of
    /// [`forward`](Self::forward). Computed as the conjugate of the forward
    /// transform of the conjugate; negating `-32768` wraps to `-32768`, as the
    /// C++ cast does.
    pub fn inverse(&self, real: &mut [i16; N], imag: &mut [i16; N]) {
        for x in imag.iter_mut() {
            *x = x.wrapping_neg();
        }
        self.transform::<false, SKIP_SATURATION>(real, imag);
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
        magnitude_squared(real, imag, mag_sq);
    }

    /// The radix-2 stages (C++ `forwardImpl`), with or without the per-stage
    /// halving.
    //
    // Never inlined: with the stages inlined into it, it is large, and when
    // LLVM inlined it into a caller anyway (once the stages became a free
    // function shared with QRFft1D) a 256-point forward transform took
    // 103,209 cycles on a Cortex-M0+ instead of 91,430, and 43,604 instead of
    // 41,991 on a Cortex-M4, for 4 KB more code.
    #[inline(never)]
    pub(crate) fn transform<const SCALE: bool, const SKIP: bool>(
        &self,
        real: &mut [i16; N],
        imag: &mut [i16; N],
    ) {
        // The checks on N and HALF (a post-monomorphization error if they fail).
        let _ = Self::STAGES;
        bit_reverse_permute(real, imag);
        run_stages::<SCALE, SKIP, N, HALF, 1>(self.twiddle_cos, self.twiddle_sin, real, imag);
    }
}

/// The radix-2 stages with half sizes `FIRST`, `2 * FIRST`, ... below `L` (the
/// C++ stage loop), on buffers of `L` values with twiddle tables of `T`
/// entries; stage `H` reads entry `pair * T / H`. [`QFft1D`] runs every stage
/// (`L = N`, `T = N / 2`); [`QRFft1D`] runs those after its fused first stage
/// on its `N / 2`-point inner transform with the `N`-point tables
/// (`L = T = N / 2`), so it reads every second entry.
///
/// Each stage runs with its half size as a constant, so offsets, strides and
/// trip counts are immediates and more registers stay free for the butterfly:
/// on a Cortex-M0+ this took a 256-point forward transform from 153,698 cycles
/// to 117,640 at opt-level 3, and from 157,939 to 135,233 at "s", for about
/// 1.5 KB more code (before the saturation skip). Stages that a length does
/// not have compile to nothing. The list reaches past any array that fits a
/// 32-bit address space.
#[inline(always)]
fn run_stages<
    const SCALE: bool,
    const SKIP: bool,
    const L: usize,
    const T: usize,
    const FIRST: usize,
>(
    cos: &[i16; T],
    sin: &[i16; T],
    real: &mut [i16; L],
    imag: &mut [i16; L],
) {
    macro_rules! stages {
        ($($h:literal)*) => {$(
            if $h >= FIRST && $h < L {
                stage::<SCALE, SKIP, L, T, $h>(cos, sin, real, imag);
            }
        )*};
    }
    stages!(
        1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144
        524288 1048576 2097152 4194304 8388608 16777216 33554432 67108864 134217728
        268435456 536870912 1073741824
    );
}

/// One stage with half size `H` (C++ `halfSize`). The C++ runs group by group
/// and loads the twiddle for every butterfly; the butterflies of a stage are
/// independent, so this runs twiddle by twiddle instead, loading each pair
/// once per stage. The results are the same. Every index is already in range;
/// the masks only let the compiler see it, so no bounds check (and no panic)
/// is emitted. `L` and `T` are powers of two.
#[inline(always)]
fn stage<const SCALE: bool, const SKIP: bool, const L: usize, const T: usize, const H: usize>(
    cos: &[i16; T],
    sin: &[i16; T],
    real: &mut [i16; L],
    imag: &mut [i16; L],
) {
    let mask = L - 1;
    let tw_mask = T - 1;
    let stride = T / H;
    for pair in 0..H {
        let k = (pair * stride) & tw_mask;
        let w_cos = i32::from(cos[k]);
        let w_sin = i32::from(sin[k]);
        let mut top = pair;
        while top < L {
            butterfly::<SCALE, SKIP, L>(real, imag, top & mask, (top + H) & mask, w_cos, w_sin);
            top += 2 * H;
        }
    }
}

/// One radix-2 butterfly on `top` and `bot` with the twiddle `(w_cos, w_sin)`
/// (the body of the C++ `forwardImpl` loop).
#[inline(always)]
fn butterfly<const SCALE: bool, const SKIP: bool, const N: usize>(
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
        // With both products in the i16 range, every halved sum is too:
        // top + t lies in [-65536, 65534] and top - t in [-65535, 65535], so
        // after `>> 1` all four fit and the saturation can be skipped (see
        // SKIP_SATURATION). The products leave that range only near full
        // scale.
        if SKIP && fits_i16(t_real) && fits_i16(t_imag) {
            real[top] = ((top_real + t_real) >> 1) as i16;
            imag[top] = ((top_imag + t_imag) >> 1) as i16;
            real[bot] = ((top_real - t_real) >> 1) as i16;
            imag[bot] = ((top_imag - t_imag) >> 1) as i16;
        } else {
            real[top] = saturate_i16((top_real + t_real) >> 1);
            imag[top] = saturate_i16((top_imag + t_imag) >> 1);
            real[bot] = saturate_i16((top_real - t_real) >> 1);
            imag[bot] = saturate_i16((top_imag - t_imag) >> 1);
        }
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

/// `real^2 + imag^2` per entry in `i64`, saturated to `i32::MAX` (C++
/// `magnitudeSquared`, the same in `QFFT1D` and `QRFFT1D`).
fn magnitude_squared<const L: usize>(real: &[i16; L], imag: &[i16; L], mag_sq: &mut [i32; L]) {
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

/// A quantized real-input FFT of length `N` (C++ `QRFFT1D<N>`): `N` real `i16`
/// samples in, the `N / 2 + 1` bins that are not redundant out, scaled by
/// `1/N` like [`QFft1D::forward`].
///
/// The spectrum of a real signal is conjugate-symmetric, so a complex FFT of
/// it, with the imaginary part zeroed, spends half its input and half its
/// output on nothing. This one packs the `N` samples into `N / 2` complex
/// values (even samples real, odd samples imaginary), runs an `N / 2`-point
/// complex FFT on them and separates the two halves in a final split step. At
/// `N = 256` that is 448 butterflies (64 of them without a multiply) and 128
/// split outputs instead of [`QFft1D`]'s 1,024 butterflies.
///
/// It takes the same twiddle tables as `QFft1D<N>`: `HALF` is `N / 2`, and
/// entry `k` holds `cos(-2πk/N)` and `sin(-2πk/N)` in Q1.15
/// (`calibration::build_fft_twiddles::<N, HALF>()`, `std` feature). The inner
/// transform reads every second entry. `N` must be a power of two of at least
/// 2 and `HALF` must be `N / 2`, or calling any method does not compile.
///
/// The bins go to two caller arrays of [`BINS`](Self::BINS) `= N / 2 + 1`
/// entries, which are also the inner transform's work space: stable Rust
/// cannot write `[i16; N / 2 + 1]` for a generic `N`, so their length is a
/// parameter of each method, and a length other than `BINS` does not compile.
/// Bin 0 (DC) and bin `N / 2` (Nyquist) are real, so their imaginary parts are
/// 0. Bin `k` for `k > N / 2` is the conjugate of bin `N - k`.
///
/// Scaling and headroom: packing two real samples into one complex value
/// lets a value of the inner transform reach `sqrt(2)` times full scale. So
/// the first inner stage, whose twiddle is exactly 1, computes `(a ± b) >> 2`
/// from the samples instead of `>> 1`, and the split step does not halve its
/// result. No input then saturates an intermediate value; the split step
/// saturates only where rounding reaches the `i16` rails. The cost is
/// precision: against an exact DFT divided by `N`, the largest bin error is
/// about 1.3 to 1.7 times [`QFft1D`]'s on the same real input (about 9 LSB
/// at `N = 256` on random and full-scale input, where `QFft1D`'s is about 6). Use
/// `QFft1D` with a zeroed imaginary buffer when that matters more than the
/// halved work. Rounding is `QFft1D`'s: every shift floors, and every Q1.15
/// product is `(a * b) >> 15`.
///
/// ```
/// use tinymind::int8::QRFft1D;
///
/// // Twiddles for N = 8 (the same tables QFft1D<8, 4> takes).
/// static COS: [i16; 4] = [32767, 23170, 0, -23170];
/// static SIN: [i16; 4] = [0, -23170, -32768, -23170];
/// static FFT: QRFft1D<'static, 8, 4> = QRFft1D { twiddle_cos: &COS, twiddle_sin: &SIN };
///
/// // A cosine that completes one cycle over the eight samples.
/// let input = [8000, 5657, 0, -5657, -8000, -5657, 0, 5657];
/// let mut real = [0; QRFft1D::<8, 4>::BINS];
/// let mut imag = [0; QRFft1D::<8, 4>::BINS];
/// FFT.forward(&input, &mut real, &mut imag);
/// // Scaled by 1/8, bin 1 holds half the amplitude (bin 7, its mirror, is
/// // not computed); the floored shifts leave a few values off by a few LSB.
/// assert_eq!(real, [0, 3998, 0, -1, 0]);
/// assert_eq!(imag, [0, -1, 0, 0, 0]);
/// let mut power = [0i32; 5];
/// FFT.magnitude_squared(&real, &imag, &mut power);
/// assert_eq!(power[1], 3998 * 3998 + 1);
/// ```
///
/// Output arrays of the wrong length do not compile:
///
/// ```compile_fail
/// # use tinymind::int8::QRFft1D;
/// static COS: [i16; 4] = [0; 4];
/// static SIN: [i16; 4] = [0; 4];
/// let fft: QRFft1D<'_, 8, 4> = QRFft1D { twiddle_cos: &COS, twiddle_sin: &SIN };
/// fft.forward(&[0; 8], &mut [0; 4], &mut [0; 4]);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct QRFft1D<'a, const N: usize, const HALF: usize> {
    /// `twiddle_cos[k]` = `cos(-2πk/N)` in Q1.15.
    pub twiddle_cos: &'a [i16; HALF],
    /// `twiddle_sin[k]` = `sin(-2πk/N)` in Q1.15.
    pub twiddle_sin: &'a [i16; HALF],
}

impl<const N: usize, const HALF: usize> QRFft1D<'_, N, HALF> {
    /// The number of bins, `N / 2 + 1`: the length of the output arrays.
    pub const BINS: usize = HALF + 1;

    /// Checks `N` and `HALF` (the C++ `static_assert`s) and that `BINS` is
    /// `HALF + 1`; a post-monomorphization error if they fail.
    const fn check<const BINS: usize>() {
        assert!(N >= 2, "FFT length must be at least 2");
        assert!(N.is_power_of_two(), "FFT length must be a power of two");
        assert!(HALF == N / 2, "HALF must be N / 2");
        assert!(
            BINS == HALF + 1,
            "the output arrays must hold N / 2 + 1 bins"
        );
    }

    /// Forward real FFT, scaled by `1/N`: bins 0 to `N / 2` of `input`'s
    /// spectrum into `real` and `imag`. `input` is left unchanged.
    pub fn forward<const BINS: usize>(
        &self,
        input: &[i16; N],
        real: &mut [i16; BINS],
        imag: &mut [i16; BINS],
    ) {
        self.forward_impl::<BINS, SKIP_SATURATION>(input, real, imag);
    }

    /// Squared magnitude of each of the `BINS` bins, `real^2 + imag^2`, in
    /// `i32`, saturated to `i32::MAX` as in [`QFft1D::magnitude_squared`].
    pub fn magnitude_squared<const BINS: usize>(
        &self,
        real: &[i16; BINS],
        imag: &[i16; BINS],
        mag_sq: &mut [i32; BINS],
    ) {
        const { Self::check::<BINS>() };
        magnitude_squared(real, imag, mag_sq);
    }

    /// [`forward`](Self::forward), with or without the saturation skip in the
    /// inner stages (see `SKIP_SATURATION`; both give the same results).
    #[inline(never)]
    pub(crate) fn forward_impl<const BINS: usize, const SKIP: bool>(
        &self,
        input: &[i16; N],
        real: &mut [i16; BINS],
        imag: &mut [i16; BINS],
    ) {
        const { Self::check::<BINS>() };
        // Bins 0 to HALF - 1 hold the inner transform, bin HALF the Nyquist
        // bin. BINS == HALF + 1, so none of these splits fails.
        let Some((zr, tail_r)) = real.split_first_chunk_mut::<HALF>() else {
            return;
        };
        let Some((zi, tail_i)) = imag.split_first_chunk_mut::<HALF>() else {
            return;
        };
        let (Some(nyquist_r), Some(nyquist_i)) = (tail_r.first_mut(), tail_i.first_mut()) else {
            return;
        };
        let (cos, sin) = (self.twiddle_cos, self.twiddle_sin);
        let zmask = HALF - 1;

        if HALF >= 2 {
            // Bit-reversed load of the packed samples, fused with the first
            // stage (twiddle 1): (a ± b) >> 2 from the full-precision samples,
            // so the inner transform has the headroom packing costs. For even
            // n, reverse(n) over log2(HALF) bits is reverse(n / 2) over one bit
            // fewer, and reverse(n + 1) adds HALF / 2; `j` steps through the
            // first sequence by reverse-carry, as in `bit_reverse_permute`.
            let mask = N - 1;
            let quarter = HALF / 2;
            let mut j = 0;
            for m in 0..quarter {
                let (j0, j1) = (j, j + quarter);
                let a_real = i32::from(input[(2 * j0) & mask]);
                let a_imag = i32::from(input[(2 * j0 + 1) & mask]);
                let b_real = i32::from(input[(2 * j1) & mask]);
                let b_imag = i32::from(input[(2 * j1 + 1) & mask]);
                // |a ± b| <= 65535, so each quarter fits in i16.
                let n = 2 * m;
                zr[n & zmask] = ((a_real + b_real) >> 2) as i16;
                zi[n & zmask] = ((a_imag + b_imag) >> 2) as i16;
                zr[(n + 1) & zmask] = ((a_real - b_real) >> 2) as i16;
                zi[(n + 1) & zmask] = ((a_imag - b_imag) >> 2) as i16;
                let mut bit = quarter >> 1;
                while j & bit != 0 {
                    j ^= bit;
                    bit >>= 1;
                }
                j |= bit;
            }
            // The remaining stages of the inner transform, QFft1D's scaled
            // butterflies; stage twiddle W_{2h}^p = W_N^{p * (N / 2) / h}.
            run_stages::<true, SKIP, HALF, HALF, 2>(cos, sin, zr, zi);
        } else {
            // N = 2: one complex value, no inner stage.
            zr[0] = input[0];
            zi[0] = input[N - 1];
        }

        // Split step. DC and Nyquist come from Z[0] alone; for N = 2 the
        // first-stage halving is taken here instead.
        let z0_real = i32::from(zr[0]);
        let z0_imag = i32::from(zi[0]);
        let edge_shift = if HALF >= 2 { 0 } else { 1 };
        zr[0] = saturate_i16((z0_real + z0_imag) >> edge_shift);
        *nyquist_r = saturate_i16((z0_real - z0_imag) >> edge_shift);
        zi[0] = 0;
        *nyquist_i = 0;

        for k in 1..=HALF / 2 {
            let (a, b) = (k & zmask, (HALF - k) & zmask);
            let (a_real, a_imag) = (i32::from(zr[a]), i32::from(zi[a]));
            let (b_real, b_imag) = (i32::from(zr[b]), i32::from(zi[b]));
            (zr[a], zi[a]) = split_bin(a_real, a_imag, b_real, b_imag, cos[a], sin[a]);
            if a != b {
                (zr[b], zi[b]) = split_bin(b_real, b_imag, a_real, a_imag, cos[b], sin[b]);
            }
        }
    }
}

/// One bin of the real FFT's split step (C++ `qfft_detail::qrfftSplitBin`):
/// `X[k] = E[k] + W * O[k]` from `a = Z[k]` and `b = Z[M - k]`, with
/// `E = (a + conj(b)) / 2` and `O = (a - conj(b)) / (2i)`.
#[inline(always)]
fn split_bin(
    a_real: i32,
    a_imag: i32,
    b_real: i32,
    b_imag: i32,
    w_cos: i16,
    w_sin: i16,
) -> (i16, i16) {
    let (w_cos, w_sin) = (i32::from(w_cos), i32::from(w_sin));
    let even_real = (a_real + b_real) >> 1;
    let even_imag = (a_imag - b_imag) >> 1;
    let odd_real = (a_imag + b_imag) >> 1;
    let odd_imag = (b_real - a_real) >> 1;
    let t_real = q15_mul(w_cos, odd_real) - q15_mul(w_sin, odd_imag);
    let t_imag = q15_mul(w_cos, odd_imag) + q15_mul(w_sin, odd_real);
    (
        saturate_i16(even_real + t_real),
        saturate_i16(even_imag + t_imag),
    )
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

    /// The forward transform gives the same bits with and without the
    /// saturation skip, at every power-of-two length up to 1024, on random
    /// samples and twiddles that reach the rails (where the skip does not
    /// apply) and on smaller ones (where it does). The host sweep runs the
    /// form the host gets; QEMU's Cortex-M0 runs the other against the C++.
    #[test]
    fn saturation_skip_changes_nothing() {
        fn check<const N: usize, const HALF: usize>(rng: &mut crate::rand::GlibcRand) {
            for trial in 0..20 {
                let scale = [0, 1, 6][trial % 3];
                let mut draw = || (rng.next_i32() as i16) >> scale;
                let cos: [i16; HALF] = core::array::from_fn(|_| draw());
                let sin: [i16; HALF] = core::array::from_fn(|_| draw());
                let real: [i16; N] = core::array::from_fn(|_| draw());
                let imag: [i16; N] = core::array::from_fn(|_| draw());
                let fft: QFft1D<'_, N, HALF> = QFft1D {
                    twiddle_cos: &cos,
                    twiddle_sin: &sin,
                };
                let (mut r0, mut i0) = (real, imag);
                fft.transform::<true, false>(&mut r0, &mut i0);
                let (mut r1, mut i1) = (real, imag);
                fft.transform::<true, true>(&mut r1, &mut i1);
                assert_eq!((r0, i0), (r1, i1), "N={N} trial {trial}");
            }
        }
        let mut rng = crate::rand::GlibcRand::new(11);
        check::<2, 1>(&mut rng);
        check::<4, 2>(&mut rng);
        check::<8, 4>(&mut rng);
        check::<64, 32>(&mut rng);
        check::<256, 128>(&mut rng);
        check::<1024, 512>(&mut rng);
    }

    /// The real FFT's forward transform gives the same bits with and without
    /// the saturation skip in its inner stages, as `QFft1D`'s does.
    #[test]
    fn rfft_saturation_skip_changes_nothing() {
        fn check<const N: usize, const HALF: usize, const BINS: usize>(
            rng: &mut crate::rand::GlibcRand,
        ) {
            for trial in 0..20 {
                let scale = [0, 1, 6][trial % 3];
                let mut draw = || (rng.next_i32() as i16) >> scale;
                let cos: [i16; HALF] = core::array::from_fn(|_| draw());
                let sin: [i16; HALF] = core::array::from_fn(|_| draw());
                let input: [i16; N] = core::array::from_fn(|_| draw());
                let fft: QRFft1D<'_, N, HALF> = QRFft1D {
                    twiddle_cos: &cos,
                    twiddle_sin: &sin,
                };
                let (mut r0, mut i0) = ([0; BINS], [0; BINS]);
                fft.forward_impl::<BINS, false>(&input, &mut r0, &mut i0);
                let (mut r1, mut i1) = ([0; BINS], [0; BINS]);
                fft.forward_impl::<BINS, true>(&input, &mut r1, &mut i1);
                assert_eq!((r0, i0), (r1, i1), "N={N} trial {trial}");
            }
        }
        let mut rng = crate::rand::GlibcRand::new(12);
        check::<2, 1, 2>(&mut rng);
        check::<4, 2, 3>(&mut rng);
        check::<8, 4, 5>(&mut rng);
        check::<64, 32, 33>(&mut rng);
        check::<256, 128, 129>(&mut rng);
        check::<1024, 512, 513>(&mut rng);
    }

    /// N = 2 is the sum and the difference, halved (the scaling by 1/N).
    #[test]
    fn rfft_of_two_samples() {
        static T: [i16; 1] = [32767];
        let fft: QRFft1D<'_, 2, 1> = QRFft1D {
            twiddle_cos: &T,
            twiddle_sin: &[0],
        };
        for (a, b) in [(3, 1), (-32768, -32768), (32767, -32768), (-5, 2)] {
            let (mut re, mut im) = ([9; 2], [9; 2]);
            fft.forward(&[a, b], &mut re, &mut im);
            let (a, b) = (i32::from(a), i32::from(b));
            assert_eq!(re, [((a + b) >> 1) as i16, ((a - b) >> 1) as i16]);
            assert_eq!(im, [0, 0]);
        }
    }

    /// The split step saturates its sums. With tables from
    /// `build_fft_twiddles` only rounding can reach the rails; other tables
    /// reach past them. Expected values from tinymind (C++)'s
    /// `qrfftSplitBin`.
    #[test]
    fn rfft_split_saturates() {
        assert_eq!(
            split_bin(0, -23000, 23000, -23000, -32768, -32768),
            (i16::MAX, 11500)
        );
        assert_eq!(
            split_bin(0, 23000, -23000, 23000, -32768, -32768),
            (i16::MIN, -11500)
        );
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
