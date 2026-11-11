//! Int8 recurrent cells, ported from tinymind (C++) `cpp/qlstm.hpp` and
//! `cpp/qgru.hpp`. Re-exported from [`int8`](super).

use super::{
    LUT_SIZE, Requantizer, apply_lut, dot_with_zero_point, multiply_by_quantized_multiplier,
};

// Where the C++ sums would overflow `i32` (outside its contract), these wrap.

/// Zero point of the sigmoid LUTs' output grid, scale 1/256: -128 stands for 0.
const SIGMOID_ZERO_POINT: i32 = -128;

mod sealed {
    pub trait Sealed {}
}

/// The storage of a [`QLstmCell`]'s cell state: `i8` (the default) or `i16`.
/// An `i16` cell state has 256 times the resolution, for long sequences.
/// Sealed: the crate implements it.
pub trait CellStorage: sealed::Sealed + Copy + Into<i32> {
    /// `v` narrowed to this type; the cell calls it only on values already in
    /// range.
    #[doc(hidden)]
    fn narrow(v: i32) -> Self;
}

impl sealed::Sealed for i8 {}
impl CellStorage for i8 {
    fn narrow(v: i32) -> Self {
        v as i8
    }
}

impl sealed::Sealed for i16 {}
impl CellStorage for i16 {
    fn narrow(v: i32) -> Self {
        v as i16
    }
}

/// Saturate to the int8 range (C++ `qLstmSaturateInt8`, `qGruSaturateInt8`).
fn saturate_i8(v: i32) -> i8 {
    v.clamp(-128, 127) as i8
}

/// One gate's pre-activations for every hidden unit:
/// `bias + rescale(W x) + rescale(R h)`, saturated to the LUT input's int8
/// grid. The input and recurrent sums each subtract their tensor's zero point.
#[allow(clippy::too_many_arguments)]
fn gate_preactivations<const IN: usize, const HID: usize>(
    input: &[i8; IN],
    input_zero_point: i8,
    recurrent: &[i8; HID],
    recurrent_zero_point: i8,
    input_weights: &[[i8; IN]; HID],
    recurrent_weights: &[[i8; HID]; HID],
    biases: &[i32; HID],
    input_rescale: (i32, i32),
    recurrent_rescale: (i32, i32),
) -> [i8; HID] {
    let mut pre = [0i8; HID];
    let rows = input_weights.iter().zip(recurrent_weights).zip(biases);
    for (out, ((w_in, w_rec), &bias)) in pre.iter_mut().zip(rows) {
        let acc_in = dot_with_zero_point(input, w_in, input_zero_point);
        let acc_rec = dot_with_zero_point(recurrent, w_rec, recurrent_zero_point);
        let r_in = multiply_by_quantized_multiplier(acc_in, input_rescale.0, input_rescale.1);
        let r_rec =
            multiply_by_quantized_multiplier(acc_rec, recurrent_rescale.0, recurrent_rescale.1);
        *out = saturate_i8(r_in.wrapping_add(r_rec).wrapping_add(bias));
    }
    pre
}

/// One time step of a quantized LSTM cell with `IN` int8 inputs and `HID`
/// hidden units (C++ `QLSTMCell`), in the TFLite / CMSIS-NN quantization model.
///
/// The cell borrows its weights, biases and lookup tables, so they can live in
/// flash, and [`forward`](Self::forward) updates the caller's hidden and cell
/// state in place: the caller runs the time loop. Gates are in TFLite's order:
/// input `i`, forget `f`, cell `g`, output `o`, and the weights are gate-major,
/// `input_weights[gate][unit][input]`, symmetric int8.
///
/// Each step, for each gate and hidden unit, the input and recurrent sums
/// (`i32`, each less its tensor's zero point) are rescaled to the gate LUT's
/// input grid and added to the bias, which is in that grid too; the result,
/// saturated to int8, indexes the gate's LUT. The LUTs' output grids are
/// fixed: sigmoid scale 1/256 with zero point -128, tanh scale 1/128 with zero
/// point 0. Then
///
/// - `c = clamp(rescale(f * (c_prev - c_zp)) + rescale(i * g) + c_zp, cell_qmin, cell_qmax)`
/// - `h = output_requantizer(o * tanh(c))`, with `tanh(c)` from
///   `tanh_cell_lut` after rescaling `c - c_zp` to its input grid.
///
/// Build the multipliers and shifts from float scales with
/// `calibration::build_lstm_params`, the biases with
/// `calibration::quantize_gate_biases` and the LUTs with
/// `calibration::build_sigmoid_lut` and `build_tanh_lut` (`std` feature), or
/// write the integers out as constants.
///
/// ```
/// use tinymind::int8::{QLstmCell, Requantizer};
///
/// // With zero weights and biases every gate sees 0: with these LUTs each
/// // sigmoid gate is 0.5 (0 in its grid) and the cell input g is 0. Each step
/// // the cell state halves (f_times_c is 2^-8: f centred at 128, times c, over
/// // 256) and the hidden state, o * tanh(c), is 0 (the tanh table is all 0).
/// static ZERO_LUT: [i8; 256] = [0; 256];
/// let cell: QLstmCell<'_, 2, 3> = QLstmCell {
///     input_weights: &[[[0; 2]; 3]; 4],
///     recurrent_weights: &[[[0; 3]; 3]; 4],
///     biases: &[[0; 3]; 4],
///     input_zero_point: 0,
///     hidden_zero_point: 0,
///     cell_zero_point: 0,
///     input_to_lut_multiplier: [1 << 30; 4],
///     input_to_lut_shift: [0; 4],
///     recurrent_to_lut_multiplier: [1 << 30; 4],
///     recurrent_to_lut_shift: [0; 4],
///     sigmoid_lut: &ZERO_LUT,
///     tanh_lut: &ZERO_LUT,
///     f_times_c_multiplier: 1 << 30,
///     f_times_c_shift: 7,
///     i_times_g_multiplier: 1 << 30,
///     i_times_g_shift: 7,
///     cell_qmin: -127,
///     cell_qmax: 127,
///     cell_to_tanh_multiplier: 1 << 30,
///     cell_to_tanh_shift: 0,
///     tanh_cell_lut: &ZERO_LUT,
///     output_requantizer: Requantizer { multiplier: 1 << 30, shift: 7, zero_point: 0, qmin: -128, qmax: 127 },
/// };
/// let mut hidden = [5i8, -5, 0];
/// let mut state = [100i8, -60, 7];
/// cell.forward(&[10, -10], &mut hidden, &mut state);
/// assert_eq!(state, [50, -30, 4]); // 3.5 rounds away from zero
/// assert_eq!(hidden, [0, 0, 0]);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct QLstmCell<'a, const IN: usize, const HID: usize, C: CellStorage = i8> {
    /// `input_weights[gate][unit][input]`, gates in the order i, f, g, o.
    pub input_weights: &'a [[[i8; IN]; HID]; 4],
    /// `recurrent_weights[gate][unit][previous unit]`.
    pub recurrent_weights: &'a [[[i8; HID]; HID]; 4],
    /// `biases[gate][unit]`, in the LUT input grid.
    pub biases: &'a [[i32; HID]; 4],
    /// The input tensor's zero point.
    pub input_zero_point: i8,
    /// The hidden state's zero point (also the output requantizer's).
    pub hidden_zero_point: i8,
    /// The cell state's zero point.
    pub cell_zero_point: C,
    /// Per gate, the Q0.31 multiplier rescaling the input sum to the LUT input grid.
    pub input_to_lut_multiplier: [i32; 4],
    /// Per gate, the shift for `input_to_lut_multiplier` (positive divides, as
    /// in [`Requantizer`]).
    pub input_to_lut_shift: [i32; 4],
    /// Per gate, the multiplier rescaling the recurrent sum to the LUT input grid.
    pub recurrent_to_lut_multiplier: [i32; 4],
    /// Per gate, the shift for `recurrent_to_lut_multiplier`.
    pub recurrent_to_lut_shift: [i32; 4],
    /// Sigmoid table for the i, f and o gates (output scale 1/256, zero point -128).
    pub sigmoid_lut: &'a [i8; LUT_SIZE],
    /// Tanh table for the g gate (output scale 1/128, zero point 0).
    pub tanh_lut: &'a [i8; LUT_SIZE],
    /// Multiplier rescaling `f * (c_prev - c_zp)` to the cell grid.
    pub f_times_c_multiplier: i32,
    /// Shift for `f_times_c_multiplier`.
    pub f_times_c_shift: i32,
    /// Multiplier rescaling `i * g` to the cell grid.
    pub i_times_g_multiplier: i32,
    /// Shift for `i_times_g_multiplier`.
    pub i_times_g_shift: i32,
    /// Lowest cell state. If `cell_qmin > cell_qmax` the cell state is `cell_qmax`.
    pub cell_qmin: C,
    /// Highest cell state.
    pub cell_qmax: C,
    /// Multiplier rescaling `c - c_zp` to `tanh_cell_lut`'s input grid.
    pub cell_to_tanh_multiplier: i32,
    /// Shift for `cell_to_tanh_multiplier`.
    pub cell_to_tanh_shift: i32,
    /// Tanh table for the cell state (output scale 1/128, zero point 0).
    pub tanh_cell_lut: &'a [i8; LUT_SIZE],
    /// Rescales `o * tanh(c)` to the hidden state's grid, adds its zero point
    /// and saturates.
    pub output_requantizer: Requantizer,
}

impl<const IN: usize, const HID: usize, C: CellStorage> QLstmCell<'_, IN, HID, C> {
    /// One time step: reads `input`, `hidden` (h at t-1) and `cell` (c at t-1),
    /// and leaves h and c at t in `hidden` and `cell`.
    pub fn forward(&self, input: &[i8; IN], hidden: &mut [i8; HID], cell: &mut [C; HID]) {
        // Every gate reads the whole previous hidden state, so all four are
        // computed before any unit is updated.
        let pre: [[i8; HID]; 4] = core::array::from_fn(|g| {
            gate_preactivations(
                input,
                self.input_zero_point,
                hidden,
                self.hidden_zero_point,
                &self.input_weights[g],
                &self.recurrent_weights[g],
                &self.biases[g],
                (self.input_to_lut_multiplier[g], self.input_to_lut_shift[g]),
                (
                    self.recurrent_to_lut_multiplier[g],
                    self.recurrent_to_lut_shift[g],
                ),
            )
        });
        let [pre_i, pre_f, pre_g, pre_o] = &pre;

        let cell_zp: i32 = self.cell_zero_point.into();
        let cell_lo: i32 = self.cell_qmin.into();
        let cell_hi: i32 = self.cell_qmax.into();
        let units = hidden.iter_mut().zip(cell.iter_mut());
        let gates = pre_i.iter().zip(pre_f).zip(pre_g).zip(pre_o);
        for ((h, c), (((&pi, &pf), &pg), &po)) in units.zip(gates) {
            // Sigmoid outputs less their zero point (0..=255, i.e. 0 to 255/256);
            // tanh's zero point is 0.
            let i = i32::from(apply_lut(pi, self.sigmoid_lut)) - SIGMOID_ZERO_POINT;
            let f = i32::from(apply_lut(pf, self.sigmoid_lut)) - SIGMOID_ZERO_POINT;
            let g = i32::from(apply_lut(pg, self.tanh_lut));
            let o = i32::from(apply_lut(po, self.sigmoid_lut)) - SIGMOID_ZERO_POINT;

            let c_prev: i32 = (*c).into();
            let scaled_fc = multiply_by_quantized_multiplier(
                f * (c_prev - cell_zp),
                self.f_times_c_multiplier,
                self.f_times_c_shift,
            );
            let scaled_ig = multiply_by_quantized_multiplier(
                i * g,
                self.i_times_g_multiplier,
                self.i_times_g_shift,
            );
            let mut c_new = scaled_fc.wrapping_add(scaled_ig).wrapping_add(cell_zp);
            // Two tests, as in the C++: with qmin > qmax the result is qmax.
            if c_new < cell_lo {
                c_new = cell_lo;
            }
            if c_new > cell_hi {
                c_new = cell_hi;
            }
            *c = C::narrow(c_new);

            let tanh_index = saturate_i8(multiply_by_quantized_multiplier(
                c_new - cell_zp,
                self.cell_to_tanh_multiplier,
                self.cell_to_tanh_shift,
            ));
            let tanh_c = i32::from(apply_lut(tanh_index, self.tanh_cell_lut));
            *h = self.output_requantizer.apply(o * tanh_c);
        }
    }
}

/// One time step of a quantized GRU cell with `IN` int8 inputs and `HID`
/// hidden units (C++ `QGRUCell`), the reset-before-multiply variant:
///
/// - `r = sigmoid(W_r x + R_r h + b_r)`, `z = sigmoid(W_z x + R_z h + b_z)`
/// - `n = tanh(W_n x + R_n (r * h) + b_n)`
/// - `h' = (1 - z) * n + z * h`
///
/// This is not PyTorch's `nn.GRU` (nor Keras' default, `reset_after=True`),
/// which computes `n = tanh(W_n x + b_in + r * (R_n h + b_hn))`, so a model
/// trained that way does not run on this cell.
///
/// Gates are in the order reset `r`, update `z`, new `n`, and the weights are
/// gate-major, `input_weights[gate][unit][input]`, symmetric int8. Each gate's
/// sums are rescaled to the LUT input grid and added to the bias as in
/// [`QLstmCell`], with the same fixed LUT output grids. `r * h` is rescaled to
/// the hidden state's grid (and saturated to int8) before `R_n` multiplies it;
/// `h'` is rescaled to that grid too and clamped to `[output_qmin,
/// output_qmax]`.
///
/// The cell borrows its weights, biases and LUTs, and
/// [`forward`](Self::forward) updates the caller's hidden state in place.
/// Build the multipliers and shifts from float scales with
/// `calibration::build_gru_params` (`std` feature), or write them out.
///
/// ```
/// use tinymind::int8::QGruCell;
///
/// // With zero weights and biases, these LUTs make the update gate z 0.5 (0
/// // in the sigmoid grid) and the candidate n 0, so each step halves the
/// // hidden state: z * h, with z centred at 128 and z_times_h 2^-8.
/// static ZERO_LUT: [i8; 256] = [0; 256];
/// let cell: QGruCell<'_, 2, 3> = QGruCell {
///     input_weights: &[[[0; 2]; 3]; 3],
///     recurrent_weights: &[[[0; 3]; 3]; 3],
///     biases: &[[0; 3]; 3],
///     input_zero_point: 0,
///     hidden_zero_point: 0,
///     input_to_lut_multiplier: [1 << 30; 3],
///     input_to_lut_shift: [0; 3],
///     recurrent_to_lut_multiplier: [1 << 30; 3],
///     recurrent_to_lut_shift: [0; 3],
///     sigmoid_lut: &ZERO_LUT,
///     tanh_lut: &ZERO_LUT,
///     r_times_h_multiplier: 1 << 30,
///     r_times_h_shift: 7,
///     one_minus_z_times_n_multiplier: 1 << 30,
///     one_minus_z_times_n_shift: 7,
///     z_times_h_multiplier: 1 << 30,
///     z_times_h_shift: 7,
///     output_qmin: -128,
///     output_qmax: 127,
/// };
/// let mut hidden = [64i8, -40, 7];
/// cell.forward(&[10, -10], &mut hidden);
/// assert_eq!(hidden, [32, -20, 4]); // 3.5 rounds away from zero
/// ```
#[derive(Clone, Copy, Debug)]
pub struct QGruCell<'a, const IN: usize, const HID: usize> {
    /// `input_weights[gate][unit][input]`, gates in the order r, z, n.
    pub input_weights: &'a [[[i8; IN]; HID]; 3],
    /// `recurrent_weights[gate][unit][previous unit]`.
    pub recurrent_weights: &'a [[[i8; HID]; HID]; 3],
    /// `biases[gate][unit]`, in the LUT input grid.
    pub biases: &'a [[i32; HID]; 3],
    /// The input tensor's zero point.
    pub input_zero_point: i8,
    /// The hidden state's zero point.
    pub hidden_zero_point: i8,
    /// Per gate, the Q0.31 multiplier rescaling the input sum to the LUT input grid.
    pub input_to_lut_multiplier: [i32; 3],
    /// Per gate, the shift for `input_to_lut_multiplier` (positive divides, as
    /// in [`Requantizer`]).
    pub input_to_lut_shift: [i32; 3],
    /// Per gate, the multiplier rescaling the recurrent sum to the LUT input grid.
    pub recurrent_to_lut_multiplier: [i32; 3],
    /// Per gate, the shift for `recurrent_to_lut_multiplier`.
    pub recurrent_to_lut_shift: [i32; 3],
    /// Sigmoid table for the r and z gates (output scale 1/256, zero point -128).
    pub sigmoid_lut: &'a [i8; LUT_SIZE],
    /// Tanh table for the n gate (output scale 1/128, zero point 0).
    pub tanh_lut: &'a [i8; LUT_SIZE],
    /// Multiplier rescaling `r * (h - h_zp)` to the hidden grid.
    pub r_times_h_multiplier: i32,
    /// Shift for `r_times_h_multiplier`.
    pub r_times_h_shift: i32,
    /// Multiplier rescaling `(1 - z) * n` to the hidden grid.
    pub one_minus_z_times_n_multiplier: i32,
    /// Shift for `one_minus_z_times_n_multiplier`.
    pub one_minus_z_times_n_shift: i32,
    /// Multiplier rescaling `z * (h - h_zp)` to the hidden grid.
    pub z_times_h_multiplier: i32,
    /// Shift for `z_times_h_multiplier`.
    pub z_times_h_shift: i32,
    /// Lowest hidden state. If `output_qmin > output_qmax` the hidden state is
    /// `output_qmax`.
    pub output_qmin: i8,
    /// Highest hidden state.
    pub output_qmax: i8,
}

impl<const IN: usize, const HID: usize> QGruCell<'_, IN, HID> {
    /// One time step: reads `input` and `hidden` (h at t-1) and leaves h at t
    /// in `hidden`.
    pub fn forward(&self, input: &[i8; IN], hidden: &mut [i8; HID]) {
        let h_zp = i32::from(self.hidden_zero_point);
        let gate = |g: usize, recurrent: &[i8; HID]| {
            gate_preactivations(
                input,
                self.input_zero_point,
                recurrent,
                self.hidden_zero_point,
                &self.input_weights[g],
                &self.recurrent_weights[g],
                &self.biases[g],
                (self.input_to_lut_multiplier[g], self.input_to_lut_shift[g]),
                (
                    self.recurrent_to_lut_multiplier[g],
                    self.recurrent_to_lut_shift[g],
                ),
            )
        };
        let r = gate(0, hidden).map(|p| apply_lut(p, self.sigmoid_lut));
        let z = gate(1, hidden).map(|p| apply_lut(p, self.sigmoid_lut));

        // r * h in the hidden grid, the recurrent input of the n gate.
        let mut rh = [0i8; HID];
        for ((out, &r), &h) in rh.iter_mut().zip(&r).zip(hidden.iter()) {
            let product = (i32::from(r) - SIGMOID_ZERO_POINT) * (i32::from(h) - h_zp);
            let scaled = multiply_by_quantized_multiplier(
                product,
                self.r_times_h_multiplier,
                self.r_times_h_shift,
            );
            *out = saturate_i8(scaled.wrapping_add(h_zp));
        }
        let n = gate(2, &rh).map(|p| apply_lut(p, self.tanh_lut));

        let lo = i32::from(self.output_qmin);
        let hi = i32::from(self.output_qmax);
        for ((h, &z), &n) in hidden.iter_mut().zip(&z).zip(&n) {
            // 1 - z in the sigmoid grid less its zero point: 128 - z (1..=256).
            let one_minus_z = -SIGMOID_ZERO_POINT - i32::from(z);
            let z_centered = i32::from(z) - SIGMOID_ZERO_POINT;
            let scaled_a = multiply_by_quantized_multiplier(
                one_minus_z * i32::from(n),
                self.one_minus_z_times_n_multiplier,
                self.one_minus_z_times_n_shift,
            );
            let scaled_b = multiply_by_quantized_multiplier(
                z_centered * (i32::from(*h) - h_zp),
                self.z_times_h_multiplier,
                self.z_times_h_shift,
            );
            let mut h_new = scaled_a.wrapping_add(scaled_b).wrapping_add(h_zp);
            if h_new < lo {
                h_new = lo;
            }
            if h_new > hi {
                h_new = hi;
            }
            *h = h_new as i8;
        }
    }
}
