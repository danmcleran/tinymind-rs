// The int8 recurrent cells and their calibration against tinymind (C++)'s
// QLSTMCell (int8 and int16 cell state) and QGRUCell, with buildQLSTMParams,
// buildQGRUParams, the bias quantizers and the LUT builders.
// tests/golden/recurrent_int8/stdout.txt is the output of
// scripts/cpp/recurrent_sweep.cpp, which draws the same weights, scales,
// parameters, states and inputs from the same generator, in the same order;
// scripts/compare-with-cpp.sh rebuilds it and checks the golden is current.
// See that program for what the cases cover.

// The calibration functions need the `std` feature.
#![cfg(feature = "std")]

// The sweep itself is in tests/sweeps/recurrent_int8.rs, which bench/ also
// runs on Cortex-M (its direct-parameter trials).

#[path = "sweeps/recurrent_int8.rs"]
mod recurrent_int8;

use recurrent_int8::{GruCalibration, LstmCalibration, Scales};
use tinymind::int8::calibration::{
    GruScales, LstmScales, SIGMOID_LUT_OUTPUT_SCALE, TANH_LUT_OUTPUT_SCALE, build_gru_params,
    build_lstm_params, build_sigmoid_lut, build_tanh_lut, quantize, quantize_gate_biases,
};

/// absmax / 127, or 1/127 for an all-zero gate.
fn absmax_scale<const N: usize, const M: usize>(w: &[[f32; N]; M]) -> f32 {
    let m = w
        .iter()
        .flatten()
        .fold(0f32, |m, v| if v.abs() > m { v.abs() } else { m });
    if m > 0.0 { m / 127.0 } else { 1.0 / 127.0 }
}

fn quantize_weights<const N: usize, const M: usize>(w: &[[f32; N]; M], scale: f32) -> [[i8; N]; M] {
    w.map(|row| row.map(|v| quantize(v, scale, 0, -127, 127)))
}

/// The C++ program's calibration, through `int8::calibration`.
struct Calibrate;

impl recurrent_int8::Calibrate for Calibrate {
    fn lstm<const I: usize, const H: usize>(
        &mut self,
        s: Scales,
        wf_in: &[[[f32; I]; H]; 4],
        wf_rec: &[[[f32; H]; H]; 4],
        bf: &[[f32; H]; 4],
    ) -> Option<LstmCalibration<I, H>> {
        let scales = LstmScales {
            input_scale: s.input,
            hidden_scale: s.hidden,
            cell_scale: s.cell,
            lut_input_scale: s.lut_input,
            input_weight_scales: core::array::from_fn(|g| absmax_scale(&wf_in[g])),
            recurrent_weight_scales: core::array::from_fn(|g| absmax_scale(&wf_rec[g])),
        };
        let p = build_lstm_params(&scales);
        Some(LstmCalibration {
            input_to_lut: (p.input_to_lut_multiplier, p.input_to_lut_shift),
            recurrent_to_lut: (p.recurrent_to_lut_multiplier, p.recurrent_to_lut_shift),
            f_times_c: (p.f_times_c_multiplier, p.f_times_c_shift),
            i_times_g: (p.i_times_g_multiplier, p.i_times_g_shift),
            cell_to_tanh: (p.cell_to_tanh_multiplier, p.cell_to_tanh_shift),
            output: (p.output_multiplier, p.output_shift),
            input_weights: core::array::from_fn(|g| {
                quantize_weights(&wf_in[g], scales.input_weight_scales[g])
            }),
            recurrent_weights: core::array::from_fn(|g| {
                quantize_weights(&wf_rec[g], scales.recurrent_weight_scales[g])
            }),
            biases: quantize_gate_biases(bf, s.lut_input),
            sigmoid_lut: build_sigmoid_lut(s.lut_input, 0, SIGMOID_LUT_OUTPUT_SCALE, -128),
            tanh_lut: build_tanh_lut(s.lut_input, 0, TANH_LUT_OUTPUT_SCALE, 0),
            tanh_cell_lut: build_tanh_lut(s.lut_input, 0, TANH_LUT_OUTPUT_SCALE, 0),
        })
    }

    fn gru<const I: usize, const H: usize>(
        &mut self,
        s: Scales,
        wf_in: &[[[f32; I]; H]; 3],
        wf_rec: &[[[f32; H]; H]; 3],
        bf: &[[f32; H]; 3],
    ) -> Option<GruCalibration<I, H>> {
        let scales = GruScales {
            input_scale: s.input,
            hidden_scale: s.hidden,
            lut_input_scale: s.lut_input,
            input_weight_scales: core::array::from_fn(|g| absmax_scale(&wf_in[g])),
            recurrent_weight_scales: core::array::from_fn(|g| absmax_scale(&wf_rec[g])),
        };
        let p = build_gru_params(&scales);
        Some(GruCalibration {
            input_to_lut: (p.input_to_lut_multiplier, p.input_to_lut_shift),
            recurrent_to_lut: (p.recurrent_to_lut_multiplier, p.recurrent_to_lut_shift),
            r_times_h: (p.r_times_h_multiplier, p.r_times_h_shift),
            one_minus_z_times_n: (
                p.one_minus_z_times_n_multiplier,
                p.one_minus_z_times_n_shift,
            ),
            z_times_h: (p.z_times_h_multiplier, p.z_times_h_shift),
            input_weights: core::array::from_fn(|g| {
                quantize_weights(&wf_in[g], scales.input_weight_scales[g])
            }),
            recurrent_weights: core::array::from_fn(|g| {
                quantize_weights(&wf_rec[g], scales.recurrent_weight_scales[g])
            }),
            biases: quantize_gate_biases(bf, s.lut_input),
            sigmoid_lut: build_sigmoid_lut(s.lut_input, 0, SIGMOID_LUT_OUTPUT_SCALE, -128),
            tanh_lut: build_tanh_lut(s.lut_input, 0, TANH_LUT_OUTPUT_SCALE, 0),
        })
    }
}

#[test]
fn recurrent_cells_match_cpp() {
    let mut out = String::new();
    recurrent_int8::report(&mut out, &mut Calibrate);
    let golden = include_str!("golden/recurrent_int8/stdout.txt");
    for (i, (ours, theirs)) in out.lines().zip(golden.lines()).enumerate() {
        assert_eq!(ours, theirs, "line {}", i + 1);
    }
    assert_eq!(out, golden);
}
