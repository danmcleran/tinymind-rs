//! Int8 elementwise addition, ported from tinymind (C++) `cpp/qadd.hpp`.
//! Re-exported from [`int8`](super).

use super::{FastRequantizer, Requantizer, multiply_by_quantized_multiplier, saturating_shl};

// The sum of the two rescaled inputs wraps where the C++ sum would overflow
// (outside its contract).

/// Elementwise addition of two int8 tensors with their own scales and zero
/// points, into a third (C++ `QAdd`, TFLite ADD): the skip connections of
/// ResNet and MobileNet-V2.
///
/// Each input, less its zero point, is shifted left by `left_shift` (20 in
/// TFLite) and rescaled by its own multiplier to a common scale, twice the
/// larger input scale; the `i32` sum is then requantized to the output.
/// A fused ReLU raises `output_requantizer.qmin` to the output zero point.
/// Build one from float scales with `calibration::build_add` (`std` feature),
/// or write the integers out as constants.
///
/// ```
/// use tinymind::int8::{QAdd, Requantizer};
///
/// // Both inputs and the output share one scale (left shift 20, then 1/2, then
/// // 2^-19 back): the sum of the real values, on the same grid.
/// const ADD: QAdd = QAdd {
///     input_a_zero_point: 0,
///     input_b_zero_point: 0,
///     left_shift: 20,
///     input_a_multiplier: 1 << 30,
///     input_a_shift: 0,
///     input_b_multiplier: 1 << 30,
///     input_b_shift: 0,
///     output_requantizer: Requantizer {
///         multiplier: 1 << 30,
///         shift: 18,
///         zero_point: 0,
///         qmin: -128,
///         qmax: 127,
///     },
/// };
/// let a = [[[10i8, -20]; 3]; 2];
/// let b = [[[5i8, 100]; 3]; 2];
/// let mut sum = [[[0i8; 2]; 3]; 2];
/// ADD.forward(&a, &b, &mut sum);
/// assert_eq!(sum[1][2], [15, 80]);
/// ```
///
/// Tensors of different shapes do not compile:
///
/// ```compile_fail
/// # use tinymind::int8::{QAdd, Requantizer};
/// # let rq = Requantizer { multiplier: 1 << 30, shift: 18, zero_point: 0, qmin: -128, qmax: 127 };
/// # let add = QAdd { input_a_zero_point: 0, input_b_zero_point: 0, left_shift: 20,
/// #     input_a_multiplier: 1 << 30, input_a_shift: 0, input_b_multiplier: 1 << 30,
/// #     input_b_shift: 0, output_requantizer: rq };
/// let a = [[[0i8; 2]; 3]; 2];
/// let b = [[[0i8; 2]; 2]; 3]; // 3 x 2, not 2 x 3
/// let mut sum = [[[0i8; 2]; 3]; 2];
/// add.forward(&a, &b, &mut sum);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QAdd {
    /// Zero point of the first input.
    pub input_a_zero_point: i8,
    /// Zero point of the second input.
    pub input_b_zero_point: i8,
    /// Left shift applied to each input less its zero point, before rescaling:
    /// head room for the multipliers. tinymind (C++)'s contract is `0..=23`
    /// (the shifted value fits an `i32`); a negative one is treated as 0, and a
    /// larger one saturates.
    pub left_shift: i32,
    /// Q0.31 multiplier rescaling the first input to the common scale.
    pub input_a_multiplier: i32,
    /// Shift for `input_a_multiplier` (positive divides, as in [`Requantizer`]).
    pub input_a_shift: i32,
    /// Q0.31 multiplier rescaling the second input to the common scale.
    pub input_b_multiplier: i32,
    /// Shift for `input_b_multiplier`.
    pub input_b_shift: i32,
    /// Rescales the `i32` sum to the output: its multiplier and shift, the
    /// output zero point and the saturation range.
    pub output_requantizer: Requantizer,
}

/// Elements from which [`QAdd::forward`] tabulates each input's rescaling:
/// building the two 256-entry tables costs about as much as rescaling 190
/// elements one by one.
const TABLE_MIN_ELEMENTS: usize = 256;

impl QAdd {
    /// `out = a + b` element by element. The three tensors have the same type,
    /// so the same shape: a flat `[i8; N]`, a row `[[i8; C]; W]` or an NHWC
    /// tensor `[[[i8; C]; W]; H]`.
    ///
    /// From 256 elements, each input's rescaling is tabulated first for all
    /// 256 int8 values (2 KB of stack), so an element costs two table reads, an
    /// add and the requantizer: about a ninth of the cycles on a Cortex-M4.
    /// The table holds the same values the element-wise form computes.
    pub fn forward<T: Tensor>(&self, a: &T, b: &T, out: &mut T) {
        let (a, b, out) = (a.as_flat(), b.as_flat(), out.as_flat_mut());
        if out.len() < TABLE_MIN_ELEMENTS {
            for ((o, &x), &y) in out.iter_mut().zip(a).zip(b) {
                *o = self
                    .output_requantizer
                    .apply(self.scaled_a(x).wrapping_add(self.scaled_b(y)));
            }
            return;
        }
        let mut table_a = [0i32; 256];
        let mut table_b = [0i32; 256];
        for (i, (ta, tb)) in table_a.iter_mut().zip(table_b.iter_mut()).enumerate() {
            let x = i as u8 as i8;
            *ta = self.scaled_a(x);
            *tb = self.scaled_b(x);
        }
        let sums = a.iter().zip(b).map(|(&x, &y)| {
            table_a[usize::from(x as u8)].wrapping_add(table_b[usize::from(y as u8)])
        });
        match FastRequantizer::new(&self.output_requantizer) {
            Some(rq) => {
                for (o, sum) in out.iter_mut().zip(sums) {
                    *o = rq.apply(sum);
                }
            }
            None => {
                for (o, sum) in out.iter_mut().zip(sums) {
                    *o = self.output_requantizer.apply(sum);
                }
            }
        }
    }

    /// `a = a + b` element by element: [`forward`](Self::forward) with its
    /// result written over the first input, as tinymind (C++)'s generated
    /// models run an addition whose first input dies there (its arena plan
    /// gives the output that input's bytes). Each element is read before it
    /// is written, so the result equals `forward`'s.
    ///
    /// ```
    /// # use tinymind::int8::{QAdd, Requantizer};
    /// # let rq = Requantizer { multiplier: 1 << 30, shift: 18, zero_point: 0, qmin: -128, qmax: 127 };
    /// # let add = QAdd { input_a_zero_point: 0, input_b_zero_point: 0, left_shift: 20,
    /// #     input_a_multiplier: 1 << 30, input_a_shift: 0, input_b_multiplier: 1 << 30,
    /// #     input_b_shift: 0, output_requantizer: rq };
    /// let (mut a, b) = ([10i8, -20, 7], [5i8, 100, -7]);
    /// let mut sum = [0i8; 3];
    /// add.forward(&a, &b, &mut sum);
    /// add.forward_in_place(&mut a, &b);
    /// assert_eq!(a, sum);
    /// ```
    pub fn forward_in_place<T: Tensor>(&self, a: &mut T, b: &T) {
        let a = a.as_flat_mut();
        let n = a.len();
        self.add_elements(
            n,
            a.iter_mut().zip(b.as_flat()).map(|(o, &y)| {
                let x = *o;
                (o, x, y)
            }),
        );
    }

    /// `b = a + b` element by element: [`forward`](Self::forward) with its
    /// result written over the second input (see
    /// [`forward_in_place`](Self::forward_in_place)).
    pub fn forward_in_place_b<T: Tensor>(&self, a: &T, b: &mut T) {
        let b = b.as_flat_mut();
        let n = b.len();
        self.add_elements(
            n,
            b.iter_mut().zip(a.as_flat()).map(|(o, &x)| {
                let y = *o;
                (o, x, y)
            }),
        );
    }

    /// The in-place forms' loop over `n` (output slot, a, b) triples: the
    /// arithmetic of [`forward`](Self::forward), tables included. `forward`
    /// keeps its own loop over three slices (its code on the boards is
    /// tuned).
    fn add_elements<'o>(&self, n: usize, elements: impl Iterator<Item = (&'o mut i8, i8, i8)>) {
        if n < TABLE_MIN_ELEMENTS {
            for (o, x, y) in elements {
                *o = self
                    .output_requantizer
                    .apply(self.scaled_a(x).wrapping_add(self.scaled_b(y)));
            }
            return;
        }
        let mut table_a = [0i32; 256];
        let mut table_b = [0i32; 256];
        for (i, (ta, tb)) in table_a.iter_mut().zip(table_b.iter_mut()).enumerate() {
            let x = i as u8 as i8;
            *ta = self.scaled_a(x);
            *tb = self.scaled_b(x);
        }
        let sum = |x: i8, y: i8| {
            table_a[usize::from(x as u8)].wrapping_add(table_b[usize::from(y as u8)])
        };
        match FastRequantizer::new(&self.output_requantizer) {
            Some(rq) => {
                for (o, x, y) in elements {
                    *o = rq.apply(sum(x, y));
                }
            }
            None => {
                for (o, x, y) in elements {
                    *o = self.output_requantizer.apply(sum(x, y));
                }
            }
        }
    }

    /// The first input less its zero point, shifted left and rescaled to the
    /// common scale.
    #[inline]
    fn scaled_a(&self, a: i8) -> i32 {
        let a_minus_zp = i32::from(a) - i32::from(self.input_a_zero_point);
        multiply_by_quantized_multiplier(
            saturating_shl(a_minus_zp, self.left_shift.max(0).unsigned_abs()),
            self.input_a_multiplier,
            self.input_a_shift,
        )
    }

    /// The same for the second input.
    #[inline]
    fn scaled_b(&self, b: i8) -> i32 {
        let b_minus_zp = i32::from(b) - i32::from(self.input_b_zero_point);
        multiply_by_quantized_multiplier(
            saturating_shl(b_minus_zp, self.left_shift.max(0).unsigned_abs()),
            self.input_b_multiplier,
            self.input_b_shift,
        )
    }
}

mod sealed {
    pub trait Sealed {}
}

/// An int8 tensor for the elementwise layers ([`QAdd`], [`QMul`](super::QMul)): `[i8; N]`,
/// `[[i8; C]; W]` or `[[[i8; C]; W]; H]`, read as its elements in memory
/// order. Sealed: the crate implements it.
pub trait Tensor: sealed::Sealed {
    /// The elements in memory order.
    fn as_flat(&self) -> &[i8];
    /// The elements in memory order, mutable.
    fn as_flat_mut(&mut self) -> &mut [i8];
}

impl<const N: usize> sealed::Sealed for [i8; N] {}
impl<const N: usize> Tensor for [i8; N] {
    fn as_flat(&self) -> &[i8] {
        self
    }
    fn as_flat_mut(&mut self) -> &mut [i8] {
        self
    }
}

impl<const W: usize, const C: usize> sealed::Sealed for [[i8; C]; W] {}
impl<const W: usize, const C: usize> Tensor for [[i8; C]; W] {
    fn as_flat(&self) -> &[i8] {
        self.as_flattened()
    }
    fn as_flat_mut(&mut self) -> &mut [i8] {
        self.as_flattened_mut()
    }
}

impl<const H: usize, const W: usize, const C: usize> sealed::Sealed for [[[i8; C]; W]; H] {}
impl<const H: usize, const W: usize, const C: usize> Tensor for [[[i8; C]; W]; H] {
    fn as_flat(&self) -> &[i8] {
        self.as_flattened().as_flattened()
    }
    fn as_flat_mut(&mut self) -> &mut [i8] {
        self.as_flattened_mut().as_flattened_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u32);

    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            self.0
        }
        fn i8(&mut self) -> i8 {
            (self.next() >> 24) as u8 as i8
        }
    }

    /// Both in-place forms equal `forward`, on the element-wise path (below
    /// 256 elements) and the table path, with and without a fast requantizer.
    #[test]
    fn in_place_forms_equal_forward() {
        fn check<const N: usize>(rng: &mut Lcg) {
            for trial in 0..8 {
                let add = QAdd {
                    input_a_zero_point: rng.i8(),
                    input_b_zero_point: rng.i8(),
                    left_shift: 20,
                    input_a_multiplier: (1 << 30) + (rng.next() >> 2) as i32,
                    input_a_shift: (rng.next() % 4) as i32,
                    input_b_multiplier: (1 << 30) + (rng.next() >> 2) as i32,
                    input_b_shift: (rng.next() % 4) as i32,
                    output_requantizer: Requantizer {
                        multiplier: (1 << 30) + (rng.next() >> 2) as i32,
                        // Shift 1 has no fast form: the general path.
                        shift: if trial % 2 == 0 {
                            1
                        } else {
                            18 + (rng.next() % 4) as i32
                        },
                        zero_point: rng.i8(),
                        qmin: -128,
                        qmax: 127,
                    },
                };
                let a: [i8; N] = core::array::from_fn(|_| rng.i8());
                let b: [i8; N] = core::array::from_fn(|_| rng.i8());
                let mut sum = [0i8; N];
                add.forward(&a, &b, &mut sum);
                let mut in_a = a;
                add.forward_in_place(&mut in_a, &b);
                assert_eq!(in_a, sum, "a, N = {N}");
                let mut in_b = b;
                add.forward_in_place_b(&a, &mut in_b);
                assert_eq!(in_b, sum, "b, N = {N}");
            }
        }
        let rng = &mut Lcg(7);
        check::<1>(rng);
        check::<255>(rng);
        check::<256>(rng);
        check::<700>(rng);
    }
}
