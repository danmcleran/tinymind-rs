//! Kani proofs: properties checked for *every* input, not a sample. Run with
//! `cargo kani` (see CLAUDE.md for setup). Compiled only under `cfg(kani)`.
//!
//! Every harness also gets Kani's built-in checks for free: no panic, no
//! arithmetic overflow, no out-of-range shift, no out-of-bounds index.
//!
//! The specifications avoid the operations they check. "Wraps" is stated as a
//! congruence modulo 2^bits; "floors" and "truncates towards zero" are stated as
//! inequalities on a remainder, with the exact quotient as a free (`kani::any`)
//! variable constrained to be the only value that satisfies them. That keeps a
//! proof from restating the implementation, and keeps division (expensive for
//! the solver) out of the specification.

use crate::activation::{Activation, Relu, Sigmoid, Tanh};
use crate::lut::{SIGMOID_Q8_8, SIGMOID_Q16_16, TANH_Q8_8, TANH_Q16_16};
use crate::qformat::{Q8_8, Q16_16};

/// `x ≡ y (mod 2^bits)`.
fn congruent(x: i64, y: i64, bits: u32) -> bool {
    (x.wrapping_sub(y)).rem_euclid(1i64 << bits) == 0
}

/// Proofs for one format, generated per storage type and fractional-bit count.
macro_rules! format_proofs {
    ($module:ident, $s:ty, $frac:expr) => {
        mod $module {
            use super::congruent;
            use crate::qformat::{QValue, Scalar};

            type Q = QValue<$s, $frac>;
            const FRAC: u32 = $frac;
            const BITS: u32 = <$s>::BITS;

            fn any() -> Q {
                Q::from_raw(kani::any())
            }

            fn wide(q: Q) -> i64 {
                i64::from(q.raw())
            }

            fn bound_for_sign(a: Q) -> i64 {
                if a.raw() > 0 {
                    i64::from(<$s>::MAX)
                } else if a.raw() < 0 {
                    i64::from(<$s>::MIN)
                } else {
                    0
                }
            }

            #[kani::proof]
            fn add_and_sub_wrap() {
                let (a, b) = (any(), any());
                let mut c = a;
                c += b;
                assert!(congruent(wide(a + b), wide(a) + wide(b), BITS));
                assert!(congruent(wide(c), wide(a) + wide(b), BITS));
                assert!(congruent(wide(a - b), wide(a) - wide(b), BITS));
            }

            /// TruncatePolicy: the exact product floored to the format's
            /// resolution, then wrapped.
            #[kani::proof]
            fn mul_floors_then_wraps() {
                let (a, b) = (any(), any());
                // In i128 so the specification itself cannot overflow.
                let p = i128::from(wide(a)) * i128::from(wide(b));
                // f is the floor of p / 2^FRAC: the only integer with
                // f * 2^FRAC <= p < (f + 1) * 2^FRAC.
                let f: i64 = kani::any();
                let scaled = i128::from(f) * (1i128 << FRAC);
                kani::assume(scaled <= p && p - scaled < (1i128 << FRAC));
                assert!(congruent(wide(a * b), f, BITS));
            }

            /// `mul_round_toward_zero` (C++ `multiplyRoundTowardZero`): the
            /// exact product rounded toward zero to the format's resolution,
            /// then wrapped; the `Scalar` method is the same function.
            #[kani::proof]
            fn mul_round_toward_zero_then_wraps() {
                let (a, b) = (any(), any());
                let p = i128::from(wide(a)) * i128::from(wide(b));
                // z is p / 2^FRAC rounded toward zero: for p >= 0 the integer
                // with z * 2^FRAC <= p < (z + 1) * 2^FRAC, for p < 0 the one
                // with (z - 1) * 2^FRAC < p <= z * 2^FRAC.
                let z: i64 = kani::any();
                let scaled = i128::from(z) * (1i128 << FRAC);
                if p >= 0 {
                    kani::assume(scaled <= p && p - scaled < (1i128 << FRAC));
                } else {
                    kani::assume(p <= scaled && scaled - p < (1i128 << FRAC));
                }
                let r = a.mul_round_toward_zero(b);
                assert!(congruent(wide(r), z, BITS));
                assert!(<Q as Scalar>::mul_round_toward_zero(a, b) == r);
            }

            /// `mul_round_nearest` (C++ `multiplyRoundNearest`): the exact
            /// product rounded to nearest, ties away from zero, to the
            /// format's resolution, then wrapped; the `Scalar` method is the
            /// same function.
            #[kani::proof]
            fn mul_round_nearest_then_wraps() {
                let (a, b) = (any(), any());
                let p = i128::from(wide(a)) * i128::from(wide(b));
                // n is p / 2^FRAC rounded to nearest, ties away from zero:
                // |p - n * 2^FRAC| <= 2^(FRAC - 1), and a tie (equality)
                // takes the n further from zero, the one with p's sign.
                let n: i64 = kani::any();
                let scaled = i128::from(n) * (1i128 << FRAC);
                let half = 1i128 << (FRAC - 1);
                if p >= 0 {
                    kani::assume(p - half < scaled && scaled <= p + half);
                } else {
                    kani::assume(p - half <= scaled && scaled < p + half);
                }
                let r = a.mul_round_nearest(b);
                assert!(congruent(wide(r), n, BITS));
                assert!(<Q as Scalar>::mul_round_nearest(a, b) == r);
            }

            /// Division never panics, for any operands (Kani's built-in overflow
            /// and divide-by-zero checks; no functional specification).
            #[kani::proof]
            fn div_never_panics() {
                let _ = any() / any();
                let _ = any().div_int(kani::any());
            }

            /// `from_int(n)` is n * 2^FRAC wrapped (the fixed part truncated).
            #[kani::proof]
            fn from_int_scales_then_wraps() {
                let n: i32 = kani::any();
                assert!(congruent(
                    wide(Q::from_int(n)),
                    i64::from(n) * (1i64 << FRAC),
                    BITS
                ));
            }

            /// `pow2_neg(n)` is 2^-n in the format: 2^(FRAC - n) raw, wrapped, or
            /// zero once 2^-n is below the resolution.
            #[kani::proof]
            fn pow2_neg_is_exact_or_zero() {
                let n: u32 = kani::any();
                let r = wide(Q::pow2_neg(n));
                if n > FRAC {
                    assert!(r == 0);
                } else {
                    assert!(congruent(r << n, 1i64 << FRAC, BITS + n));
                }
            }

            #[kani::proof]
            fn zero_tolerance_matches_definition() {
                let a = any();
                let tolerance = 1i64 << if FRAC > 7 { FRAC - 7 } else { 1 };
                assert!(a.is_within_zero_tolerance() == (wide(a).abs() < tolerance));
            }

            #[kani::proof]
            fn comparisons_are_on_the_raw_value() {
                let (a, b) = (any(), any());
                assert!((a < b) == (a.raw() < b.raw()));
                assert!((a == b) == (a.raw() == b.raw()));
            }
        }
    };
}

format_proofs!(q16_16, i32, 16);
format_proofs!(q8_8, i16, 8);
// Formats at the edges of the storage width: no fractional room, no integer room.
format_proofs!(q31_1, i32, 1);
format_proofs!(q1_31, i32, 31);
format_proofs!(q15_1, i16, 1);
format_proofs!(q1_15, i16, 15);

/// For every input, a table activation returns a value between its first and
/// last table entries (interpolation never overshoots), and the derivative
/// formula does not panic.
macro_rules! activation_proofs {
    ($name:ident, $q:ty, $s:ty, $act:ty, $table:expr) => {
        #[kani::proof]
        fn $name() {
            let x = <$q>::from_raw(kani::any());
            let y = <$act>::activate(x).raw();
            let first = $table[0] as $s;
            let last = $table[$table.len() - 1] as $s;
            assert!(first <= y && y <= last);
            let _ = <$act>::derivative(<$q>::from_raw(kani::any()));
        }
    };
}

activation_proofs!(tanh_q16_16_in_range, Q16_16, i32, Tanh, TANH_Q16_16);
activation_proofs!(
    sigmoid_q16_16_in_range,
    Q16_16,
    i32,
    Sigmoid,
    SIGMOID_Q16_16
);
activation_proofs!(tanh_q8_8_in_range, Q8_8, i16, Tanh, TANH_Q8_8);
activation_proofs!(sigmoid_q8_8_in_range, Q8_8, i16, Sigmoid, SIGMOID_Q8_8);

/// ReLU: max(0, x), derivative 0 or 1, for every input in both formats.
#[kani::proof]
fn relu_definition() {
    let x = Q16_16::from_raw(kani::any());
    let y = Relu::activate(x);
    assert!(y.raw() == x.raw().max(0));
    let d = Relu::derivative(y);
    assert!(d == Q16_16::ZERO || d == Q16_16::ONE);
    let x = Q8_8::from_raw(kani::any());
    assert!(Relu::activate(x).raw() == x.raw().max(0));
}

/// Int8 affine inference (`crate::int8`).
mod int8 {
    use crate::int8::{
        LUT_SIZE, QDense, QDensePerChannel, QSoftmax, Requantizer, apply_lut,
        exp_on_negative_values, multiply_by_quantized_multiplier, one_over_one_plus_x,
        rounding_divide_by_pot, rounding_high_mul_divide_by_pot, saturating_shl,
    };

    /// gemmlowp's two roundings, written as tinymind (C++)'s unit test writes them:
    /// the high multiply with a sign-dependent nudge and a division that
    /// truncates towards zero, then the remainder/threshold divide.
    fn two_step(x: i32, m: i32, e: i32) -> i32 {
        let h = if x == i32::MIN && m == i32::MIN {
            i32::MAX
        } else {
            let ab = i64::from(x) * i64::from(m);
            let nudge: i64 = if ab >= 0 { 1 << 30 } else { 1 - (1 << 30) };
            ((ab + nudge) / (1i64 << 31)) as i32
        };
        let mask = ((1u32 << e) - 1) as i32;
        let threshold = (mask >> 1) + i32::from(h < 0);
        (h >> e) + i32::from((h & mask) > threshold)
    }

    /// The requantizer's fused single-rounding path equals the two-step form
    /// wherever it is used (multiplier >= 0, 2 <= shift <= 31).
    #[kani::proof]
    fn fused_requantize_matches_two_step() {
        let (x, m, e): (i32, i32, i32) = (kani::any(), kani::any(), kani::any());
        kani::assume(m >= 0 && (2..=31).contains(&e));
        assert!(rounding_high_mul_divide_by_pot(x, m, e) == two_step(x, m, e));
    }

    /// The building blocks and the requantizer never panic, for any field
    /// values, including shifts far outside the documented range.
    #[kani::proof]
    fn requantizer_never_panics() {
        let (x, m, s): (i32, i32, i32) = (kani::any(), kani::any(), kani::any());
        let _ = rounding_divide_by_pot(x, s);
        let _ = multiply_by_quantized_multiplier(x, m, s);
        let rq = Requantizer {
            multiplier: m,
            shift: s,
            zero_point: kani::any(),
            qmin: kani::any(),
            qmax: kani::any(),
        };
        let y = rq.apply(x);
        // With qmin <= qmax the result is inside the bounds.
        if rq.qmin <= rq.qmax {
            assert!(rq.qmin <= y && y <= rq.qmax);
        }
    }

    /// The left shift in `multiply_by_quantized_multiplier` and `QAdd`:
    /// `x * 2^n` saturated to the `i32` range, for every `x` and every `n` up
    /// to 40 (each `n` above 31 takes the same branch as 31). The spec is in
    /// i128, where the product cannot overflow.
    #[kani::proof]
    fn saturating_shl_saturates() {
        let (x, n): (i32, u32) = (kani::any(), kani::any());
        kani::assume(n <= 40);
        let exact = i128::from(x) << n;
        let want = exact.clamp(i128::from(i32::MIN), i128::from(i32::MAX)) as i32;
        assert!(saturating_shl(x, n) == want);
    }

    /// Multiplication, padding and concatenation never panic, for any
    /// parameters and inputs.
    #[kani::proof]
    #[kani::unwind(4)]
    fn compose_never_panics() {
        let mul = crate::int8::QMul {
            input_a_zero_point: kani::any(),
            input_b_zero_point: kani::any(),
            requantizer: Requantizer {
                multiplier: kani::any(),
                shift: kani::any(),
                zero_point: kani::any(),
                qmin: kani::any(),
                qmax: kani::any(),
            },
        };
        let (a, b): ([i8; 2], [i8; 2]) = (kani::any(), kani::any());
        let mut y = [0i8; 2];
        mul.forward(&a, &b, &mut y);
        let x: [[[i8; 1]; 1]; 1] = kani::any();
        let mut padded = [[[0i8; 1]; 2]; 2];
        crate::int8::QPad2D::<1, 0, 0, 1> {
            pad_value: kani::any(),
        }
        .forward(&x, &mut padded);
        let concat = crate::int8::QConcat {
            input_a_zero_point: kani::any(),
            input_b_zero_point: kani::any(),
            requantizer_a: mul.requantizer,
            requantizer_b: Requantizer {
                multiplier: kani::any(),
                shift: kani::any(),
                zero_point: kani::any(),
                qmin: kani::any(),
                qmax: kani::any(),
            },
        };
        let xb: [[[i8; 2]; 1]; 1] = kani::any();
        let mut joined = [[[0i8; 3]; 1]; 1];
        concat.forward(&x, &xb, &mut joined);
    }

    fn any_requantizer() -> Requantizer {
        Requantizer {
            multiplier: kani::any(),
            shift: kani::any(),
            zero_point: kani::any(),
            qmin: kani::any(),
            qmax: kani::any(),
        }
    }

    /// One step of a 1-input, 2-unit int8 LSTM cell with cell state `C`, for
    /// any weights, biases, LUTs, multipliers, shifts, zero points, bounds,
    /// input and state: it never panics, and with `cell_qmin <= cell_qmax` the
    /// new cell state lies within them.
    fn lstm_step<C: crate::int8::CellStorage + kani::Arbitrary + PartialOrd>() {
        let (w_in, w_rec, biases): ([[[i8; 1]; 2]; 4], [[[i8; 2]; 2]; 4], [[i32; 2]; 4]) =
            (kani::any(), kani::any(), kani::any());
        let (sigmoid, tanh, tanh_cell): ([i8; 256], [i8; 256], [i8; 256]) =
            (kani::any(), kani::any(), kani::any());
        let cell = crate::int8::QLstmCell::<'_, 1, 2, C> {
            input_weights: &w_in,
            recurrent_weights: &w_rec,
            biases: &biases,
            input_zero_point: kani::any(),
            hidden_zero_point: kani::any(),
            cell_zero_point: kani::any(),
            input_to_lut_multiplier: kani::any(),
            input_to_lut_shift: kani::any(),
            recurrent_to_lut_multiplier: kani::any(),
            recurrent_to_lut_shift: kani::any(),
            sigmoid_lut: &sigmoid,
            tanh_lut: &tanh,
            f_times_c_multiplier: kani::any(),
            f_times_c_shift: kani::any(),
            i_times_g_multiplier: kani::any(),
            i_times_g_shift: kani::any(),
            cell_qmin: kani::any(),
            cell_qmax: kani::any(),
            cell_to_tanh_multiplier: kani::any(),
            cell_to_tanh_shift: kani::any(),
            tanh_cell_lut: &tanh_cell,
            output_requantizer: any_requantizer(),
        };
        let x: [i8; 1] = kani::any();
        let mut h: [i8; 2] = kani::any();
        let mut c: [C; 2] = kani::any();
        cell.forward(&x, &mut h, &mut c);
        if cell.cell_qmin <= cell.cell_qmax {
            assert!(
                c.iter()
                    .all(|v| cell.cell_qmin <= *v && *v <= cell.cell_qmax)
            );
        }
    }

    #[kani::proof]
    #[kani::unwind(5)]
    fn lstm_int8_cell_never_panics() {
        lstm_step::<i8>();
    }

    #[kani::proof]
    #[kani::unwind(5)]
    fn lstm_int16_cell_never_panics() {
        lstm_step::<i16>();
    }

    /// One step of a 1-input, 2-unit int8 GRU cell, for any weights, biases,
    /// LUTs, multipliers, shifts, zero points, bounds, input and state: it
    /// never panics, and with `output_qmin <= output_qmax` the new hidden state
    /// lies within them.
    #[kani::proof]
    #[kani::unwind(5)]
    fn gru_cell_never_panics() {
        let (w_in, w_rec, biases): ([[[i8; 1]; 2]; 3], [[[i8; 2]; 2]; 3], [[i32; 2]; 3]) =
            (kani::any(), kani::any(), kani::any());
        let (sigmoid, tanh): ([i8; 256], [i8; 256]) = (kani::any(), kani::any());
        let cell = crate::int8::QGruCell::<'_, 1, 2> {
            input_weights: &w_in,
            recurrent_weights: &w_rec,
            biases: &biases,
            input_zero_point: kani::any(),
            hidden_zero_point: kani::any(),
            input_to_lut_multiplier: kani::any(),
            input_to_lut_shift: kani::any(),
            recurrent_to_lut_multiplier: kani::any(),
            recurrent_to_lut_shift: kani::any(),
            sigmoid_lut: &sigmoid,
            tanh_lut: &tanh,
            r_times_h_multiplier: kani::any(),
            r_times_h_shift: kani::any(),
            one_minus_z_times_n_multiplier: kani::any(),
            one_minus_z_times_n_shift: kani::any(),
            z_times_h_multiplier: kani::any(),
            z_times_h_shift: kani::any(),
            output_qmin: kani::any(),
            output_qmax: kani::any(),
        };
        let x: [i8; 1] = kani::any();
        let mut h: [i8; 2] = kani::any();
        cell.forward(&x, &mut h);
        if cell.output_qmin <= cell.output_qmax {
            assert!(
                h.iter()
                    .all(|v| cell.output_qmin <= *v && *v <= cell.output_qmax)
            );
        }
    }

    /// The elementwise addition never panics, for any parameters and inputs,
    /// shifts far outside the documented range included.
    #[kani::proof]
    fn add_never_panics() {
        let add = crate::int8::QAdd {
            input_a_zero_point: kani::any(),
            input_b_zero_point: kani::any(),
            left_shift: kani::any(),
            input_a_multiplier: kani::any(),
            input_a_shift: kani::any(),
            input_b_multiplier: kani::any(),
            input_b_shift: kani::any(),
            output_requantizer: Requantizer {
                multiplier: kani::any(),
                shift: kani::any(),
                zero_point: kani::any(),
                qmin: kani::any(),
                qmax: kani::any(),
            },
        };
        let (a, b): ([i8; 1], [i8; 1]) = (kani::any(), kani::any());
        let mut y = [0i8; 1];
        add.forward(&a, &b, &mut y);
    }

    /// A dense layer never panics, for any weights, biases, inputs and
    /// requantizer.
    #[kani::proof]
    fn dense_never_panics() {
        let weights: [[i8; 3]; 2] = kani::any();
        let biases: [i32; 2] = kani::any();
        let with_bias: bool = kani::any();
        let layer = QDense {
            weights: &weights,
            biases: if with_bias { Some(&biases) } else { None },
            input_zero_point: kani::any(),
            requantizer: Requantizer {
                multiplier: kani::any(),
                shift: kani::any(),
                zero_point: kani::any(),
                qmin: kani::any(),
                qmax: kani::any(),
            },
        };
        let input: [i8; 3] = kani::any();
        let _ = layer.forward(&input);
    }

    /// The per-channel dense layer never panics either.
    #[kani::proof]
    fn dense_per_channel_never_panics() {
        let weights: [[i8; 3]; 2] = kani::any();
        let biases: [i32; 2] = kani::any();
        let requantizers: [Requantizer; 2] = [
            Requantizer {
                multiplier: kani::any(),
                shift: kani::any(),
                zero_point: kani::any(),
                qmin: kani::any(),
                qmax: kani::any(),
            },
            Requantizer {
                multiplier: kani::any(),
                shift: kani::any(),
                zero_point: kani::any(),
                qmin: kani::any(),
                qmax: kani::any(),
            },
        ];
        let layer = QDensePerChannel {
            weights: &weights,
            biases: if kani::any() { Some(&biases) } else { None },
            input_zero_point: kani::any(),
            requantizers: &requantizers,
        };
        let input: [i8; 3] = kani::any();
        let _ = layer.forward(&input);
    }

    /// The softmax's exponential and reciprocal never panic, for any input,
    /// in or out of their domains.
    #[kani::proof]
    fn softmax_parts_never_panic() {
        let _ = exp_on_negative_values(kani::any());
        let _ = one_over_one_plus_x(kani::any());
    }

    /// The softmax layer never panics, for any parameters and logits.
    #[kani::proof]
    fn softmax_never_panics() {
        let softmax = QSoftmax {
            input_multiplier: kani::any(),
            input_left_shift: kani::any(),
            diff_min: kani::any(),
        };
        let input: [i8; 2] = kani::any();
        let _ = softmax.forward(&input);
    }

    #[kani::proof]
    fn lut_never_panics() {
        let lut: [i8; LUT_SIZE] = [0; LUT_SIZE];
        let _ = apply_lut(kani::any(), &lut);
    }
}

/// Trees: the walk, the ensemble and the validity checks never panic, for any
/// pool (indices in or out of range, cycles), root, class, base score and input.
mod tree {
    use crate::tree::{QDecisionTree, QGbdt, QTreeNode, walk_tree};

    fn any_node() -> QTreeNode<i8> {
        QTreeNode {
            feature: kani::any(),
            threshold: kani::any(),
            left: kani::any(),
            right: kani::any(),
            leaf_value: kani::any(),
        }
    }

    #[kani::proof]
    #[kani::unwind(7)]
    fn walk_never_panics() {
        let nodes = [any_node(), any_node(), any_node(), any_node()];
        let x: [i8; 2] = kani::any();
        let max_nodes: usize = kani::any();
        kani::assume(max_nodes <= 6);
        let _ = walk_tree(&nodes, kani::any(), &x, max_nodes);
        let tree = QDecisionTree::<'_, i8, 4, 2>::new(&nodes);
        let _ = (tree.predict(&x), tree.is_valid());
    }

    #[kani::proof]
    #[kani::unwind(5)]
    fn ensemble_never_panics() {
        let nodes = [any_node(), any_node(), any_node()];
        let roots: [i16; 2] = kani::any();
        let classes: [i16; 2] = kani::any();
        let base: [i32; 2] = kani::any();
        let gbdt = QGbdt::<'_, i8, 2, 2, 3, 2> {
            nodes: &nodes,
            tree_root: &roots,
            tree_class: &classes,
            base_score: if kani::any() { Some(&base) } else { None },
        };
        let x: [i8; 2] = kani::any();
        let _ = (gbdt.forward(&x), gbdt.predict(&x), gbdt.is_valid());
    }
}

/// Q-learning: no panic for any Q values, rates, exploration percentage,
/// states, actions and rewards, whatever the environment returns (more valid
/// actions than `ACTIONS`, random integers out of range).
mod qlearn {
    use crate::qformat::Q8_8;
    use crate::qlearn::{Environment, Experience, QLearner, QTable};

    struct AnyEnvironment;

    impl Environment<3> for AnyEnvironment {
        fn valid_actions(&self, _state: usize, actions: &mut [usize; 3]) -> usize {
            *actions = kani::any();
            kani::any()
        }

        fn random_int(&mut self, _min: usize, _max: usize) -> usize {
            kani::any()
        }
    }

    fn any_q8_8() -> Q8_8 {
        Q8_8::from_raw(kani::any())
    }

    #[kani::proof]
    #[kani::unwind(5)]
    fn qlearner_never_panics() {
        let table = QTable::<Q8_8, 2, 3>::from_values([
            [any_q8_8(), any_q8_8(), any_q8_8()],
            [any_q8_8(), any_q8_8(), any_q8_8()],
        ]);
        let mut q = QLearner::<Q8_8, AnyEnvironment, 2, 3, _>::with_q_values(
            AnyEnvironment,
            table,
            any_q8_8(),
            any_q8_8(),
            kani::any(),
        );
        if kani::any() {
            q.start_new_episode();
        }
        let _ = q.take_action(kani::any());
        q.update_from_experience(&Experience {
            state: kani::any(),
            action: kani::any(),
            reward: any_q8_8(),
            new_state: kani::any(),
        });
        let _ = q.take_action(kani::any());
    }
}

/// Int8 convolutions and pooling: no panic for any weights, biases, inputs,
/// zero points and requantizers, with windows reaching into padding on every
/// side; the pooling divide rounds half away from zero.
mod conv {
    use crate::int8::pool::rounded_divide;
    use crate::int8::{
        FastRequantizer, QAvgPool2D, QConv2D, QConv2DPerChannel, QDepthwiseConv2D,
        QGlobalAvgPool2D, QMaxPool2D, QPointwiseConv2D, QPointwiseConv2DPerChannel, Requantizer,
    };

    fn any_requantizer() -> Requantizer {
        Requantizer {
            multiplier: kani::any(),
            shift: kani::any(),
            zero_point: kani::any(),
            qmin: kani::any(),
            qmax: kani::any(),
        }
    }

    /// A 2x2 kernel over a 2x2 input, stride 2 down with one row of padding
    /// on top, stride 1 across with one column on the right: two outputs,
    /// each with taps in the padding.
    #[kani::proof]
    #[kani::unwind(6)]
    fn conv_never_panics() {
        let weights: [[[[i8; 1]; 2]; 2]; 1] = kani::any();
        let biases: [i32; 1] = kani::any();
        let layer: QConv2D<'_, 1, 2, 2, 1, 2, 1, 1, 0, 0, 1> = QConv2D {
            weights: &weights,
            biases: if kani::any() { Some(&biases) } else { None },
            input_zero_point: kani::any(),
            requantizer: any_requantizer(),
        };
        let input: [[[i8; 1]; 2]; 2] = kani::any();
        let mut output = [[[0i8; 1]; 2]; 1];
        layer.forward(&input, &mut output);
    }

    /// Two input channels, so the layer takes its 16-bit band path: a 2x2
    /// kernel over a 2x2 input padded on the bottom and the left, so one
    /// window lies inside the input and the others reach into the padding.
    #[kani::proof]
    #[kani::unwind(6)]
    fn conv_band_never_panics() {
        let weights: [[[[i8; 2]; 2]; 2]; 1] = kani::any();
        let biases: [i32; 1] = kani::any();
        let requantizers = [any_requantizer()];
        let layer: QConv2DPerChannel<'_, 2, 2, 2, 1, 1, 1, 0, 1, 1, 0> = QConv2DPerChannel {
            weights: &weights,
            biases: if kani::any() { Some(&biases) } else { None },
            input_zero_point: kani::any(),
            requantizers: &requantizers,
        };
        let input: [[[i8; 2]; 2]; 2] = kani::any();
        let mut output = [[[0i8; 1]; 2]; 2];
        layer.forward(&input, &mut output);
    }

    #[kani::proof]
    #[kani::unwind(6)]
    fn conv_per_channel_never_panics() {
        let weights: [[[[i8; 1]; 2]; 2]; 2] = kani::any();
        let biases: [i32; 2] = kani::any();
        let requantizers = [any_requantizer(), any_requantizer()];
        let layer: QConv2DPerChannel<'_, 1, 2, 2, 2, 1, 1, 0, 1, 1, 0> = QConv2DPerChannel {
            weights: &weights,
            biases: if kani::any() { Some(&biases) } else { None },
            input_zero_point: kani::any(),
            requantizers: &requantizers,
        };
        let input: [[[i8; 1]; 1]; 1] = kani::any();
        let mut output = [[[0i8; 2]; 1]; 1];
        layer.forward(&input, &mut output);
    }

    #[kani::proof]
    #[kani::unwind(4)]
    fn depthwise_never_panics() {
        let weights: [[[i8; 2]; 2]; 2] = kani::any();
        let biases: [i32; 2] = kani::any();
        let requantizers = [any_requantizer(), any_requantizer()];
        let layer: QDepthwiseConv2D<'_, 2, 2, 2, 1, 1, 1, 0, 0, 1> = QDepthwiseConv2D {
            weights: &weights,
            biases: if kani::any() { Some(&biases) } else { None },
            input_zero_point: kani::any(),
            requantizers: &requantizers,
        };
        let input: [[[i8; 2]; 1]; 1] = kani::any();
        let mut output = [[[0i8; 2]; 1]; 1];
        layer.forward(&input, &mut output);
    }

    #[kani::proof]
    #[kani::unwind(4)]
    fn pointwise_never_panics() {
        let weights: [[i8; 2]; 2] = kani::any();
        let biases: [i32; 2] = kani::any();
        let input: [[[i8; 2]; 1]; 1] = kani::any();
        let mut output = [[[0i8; 2]; 1]; 1];
        let layer = QPointwiseConv2D {
            weights: &weights,
            biases: if kani::any() { Some(&biases) } else { None },
            input_zero_point: kani::any(),
            requantizer: any_requantizer(),
        };
        layer.forward(&input, &mut output);
    }

    #[kani::proof]
    #[kani::unwind(4)]
    fn pointwise_per_channel_never_panics() {
        let weights: [[i8; 2]; 2] = kani::any();
        let biases: [i32; 2] = kani::any();
        let input: [[[i8; 2]; 1]; 1] = kani::any();
        let mut output = [[[0i8; 2]; 1]; 1];
        let requantizers = [any_requantizer(), any_requantizer()];
        let layer = QPointwiseConv2DPerChannel {
            weights: &weights,
            biases: if kani::any() { Some(&biases) } else { None },
            input_zero_point: kani::any(),
            requantizers: &requantizers,
        };
        layer.forward(&input, &mut output);
    }

    #[kani::proof]
    #[kani::unwind(4)]
    fn pools_never_panic() {
        let input: [[[i8; 1]; 3]; 2] = kani::any();
        let mut output = [[[0i8; 1]; 2]; 1];
        QMaxPool2D::<2, 2, 1, 1>.forward(&input, &mut output);
        let avg = QAvgPool2D::<2, 2, 1, 1> {
            qmin: kani::any(),
            qmax: kani::any(),
        };
        avg.forward(&input, &mut output);
        let global = QGlobalAvgPool2D {
            qmin: kani::any(),
            qmax: kani::any(),
        };
        let _ = global.forward(&input);
    }

    /// The prepared requantizer the conv layers use (tinymind (C++)'s
    /// `FastRequantizer`) gives exactly `Requantizer::apply`'s result, for every
    /// accumulator and every requantizer it accepts.
    #[kani::proof]
    fn fast_requantizer_matches_apply() {
        let rq = any_requantizer();
        if let Some(fast) = FastRequantizer::new(&rq) {
            let acc: i32 = kani::any();
            assert!(fast.apply(acc) == rq.apply(acc));
        }
    }

    /// Total for every numerator and denominator.
    #[kani::proof]
    fn rounded_divide_never_panics() {
        let _ = rounded_divide(kani::any(), kani::any());
    }

    /// For a positive denominator and a numerator the bias cannot push past
    /// the `i32` rails, the result is the nearest integer to `n / d`, ties
    /// away from zero.
    #[kani::proof]
    fn rounded_divide_rounds_half_away() {
        let (n, d): (i32, i32) = (kani::any(), kani::any());
        kani::assume(d > 0 && i64::from(n).abs() <= i64::from(i32::MAX - d / 2));
        let r = i64::from(rounded_divide(n, d));
        let (n, d) = (i64::from(n), i64::from(d));
        let error = r * d - n;
        assert!(2 * error.abs() <= d);
        if 2 * error.abs() == d {
            assert!((r * d).abs() > n.abs());
        }
    }
}

/// LSTM network: no panic for any weights, rates, clipping range, inputs and
/// targets, over two steps with training and a reset (one input, one cell, one
/// output in Q8.8; every loop is over the network's const sizes).
mod recurrent {
    use crate::activation::{Sigmoid, Tanh};
    use crate::qformat::Q8_8;
    use crate::recurrent::LstmNetwork;

    fn any_q8_8() -> Q8_8 {
        Q8_8::from_raw(kani::any())
    }

    #[kani::proof]
    #[kani::unwind(6)]
    fn lstm_never_panics() {
        let mut nn = LstmNetwork::<Q8_8, Tanh, Sigmoid, 1, 1, 1>::new(any_q8_8);
        nn.set_learning_rate(any_q8_8());
        nn.set_momentum_rate(any_q8_8());
        nn.set_acceleration_rate(any_q8_8());
        if kani::any() {
            nn.set_gradient_clip(Some((any_q8_8(), any_q8_8())));
        }
        nn.feed_forward(&[any_q8_8()]);
        nn.train(&[any_q8_8()]);
        if kani::any() {
            nn.reset_state();
        }
        nn.feed_forward(&[any_q8_8()]);
        let _ = (nn.calculate_error(&[any_q8_8()]), nn.learned_values());
    }
}
