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

use std::fmt::Write;

use tinymind::int8::calibration::{
    GruScales, LstmScales, SIGMOID_LUT_OUTPUT_SCALE, TANH_LUT_OUTPUT_SCALE, build_gru_params,
    build_lstm_params, build_sigmoid_lut, build_tanh_lut, quantize, quantize_gate_biases,
};
use tinymind::int8::{CellStorage, QGruCell, QLstmCell, Requantizer};

const STEPS: usize = 32;
const FNV_START: u32 = 0x811C_9DC5;

struct Lcg(u32);

impl Lcg {
    fn draw(&mut self, n: u32) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.0 >> 8) % n
    }

    fn i8(&mut self) -> i8 {
        (self.draw(256) as i32 - 128) as i8
    }

    fn i16(&mut self) -> i16 {
        (self.draw(65536) as i32 - 32768) as i16
    }

    /// (1 + d) * 2^-(16 + base + e): exact in f32.
    fn scale(&mut self, base: i32) -> f32 {
        let m = (1 + self.draw(1 << 16)) as f32;
        m * 2f32.powi(-16 - base - self.draw(4) as i32)
    }

    /// (2^16 + d) * 2^-(17 + base): exact in f32.
    fn narrow_scale(&mut self, base: i32) -> f32 {
        let m = ((1 << 16) + self.draw(1 << 16)) as f32;
        m * 2f32.powi(-17 - base)
    }

    /// (d - 32768) * 2^-16, in [-0.5, 0.5).
    fn real(&mut self) -> f32 {
        (self.draw(65536) as i32 - 32768) as f32 * 2f32.powi(-16)
    }

    fn multiplier(&mut self) -> i32 {
        ((1u32 << 30) + (self.draw(1 << 24) << 6)) as i32
    }

    fn shift(&mut self, lo: i32, n: u32) -> i32 {
        if self.draw(4) == 0 {
            self.draw(5) as i32 - 3
        } else {
            lo + self.draw(n) as i32
        }
    }

    fn fill(&mut self, x: &mut [i8]) {
        for v in x {
            *v = self.i8();
        }
    }

    fn reals<const N: usize>(&mut self) -> [f32; N] {
        core::array::from_fn(|_| self.real())
    }
}

/// The C++ program's per-cell-type draws and limits.
trait Cell: CellStorage + std::fmt::Display {
    const WIDE: bool;
    const MIN: Self;
    const MIN_PLUS_ONE: Self;
    const MAX: Self;
    fn any(r: &mut Lcg) -> Self;
    fn calibrated_zero_point(r: &mut Lcg) -> Self;
    fn direct_zero_point(r: &mut Lcg) -> Self;
    fn hash(h: u32, x: &[Self]) -> u32;
}

impl Cell for i8 {
    const WIDE: bool = false;
    const MIN: Self = i8::MIN;
    const MIN_PLUS_ONE: Self = i8::MIN + 1;
    const MAX: Self = i8::MAX;
    fn any(r: &mut Lcg) -> Self {
        r.i8()
    }
    fn calibrated_zero_point(r: &mut Lcg) -> Self {
        r.i8() / 8
    }
    fn direct_zero_point(r: &mut Lcg) -> Self {
        r.i8()
    }
    fn hash(h: u32, x: &[Self]) -> u32 {
        fnv(h, x.iter().map(|&v| v as u8))
    }
}

impl Cell for i16 {
    const WIDE: bool = true;
    const MIN: Self = i16::MIN;
    const MIN_PLUS_ONE: Self = i16::MIN + 1;
    const MAX: Self = i16::MAX;
    fn any(r: &mut Lcg) -> Self {
        r.i16()
    }
    fn calibrated_zero_point(r: &mut Lcg) -> Self {
        (r.draw(2001) as i32 - 1000) as i16
    }
    fn direct_zero_point(r: &mut Lcg) -> Self {
        Self::calibrated_zero_point(r)
    }
    fn hash(h: u32, x: &[Self]) -> u32 {
        fnv(h, x.iter().flat_map(|v| v.to_le_bytes()))
    }
}

fn fnv(h: u32, bytes: impl IntoIterator<Item = u8>) -> u32 {
    bytes
        .into_iter()
        .fold(h, |h, b| (h ^ u32::from(b)).wrapping_mul(0x0100_0193))
}

fn fnv_i8(h: u32, x: &[i8]) -> u32 {
    fnv(h, x.iter().map(|&v| v as u8))
}

fn fnv_i32(h: u32, x: &[i32]) -> u32 {
    fnv(h, x.iter().flat_map(|v| v.to_le_bytes()))
}

/// absmax / 127, or 1/127 for an all-zero gate.
fn absmax_scale(w: &[f32]) -> f32 {
    let m = w
        .iter()
        .fold(0f32, |m, v| if v.abs() > m { v.abs() } else { m });
    if m > 0.0 { m / 127.0 } else { 1.0 / 127.0 }
}

fn quantize_weights<const N: usize, const M: usize>(w: &[[f32; N]; M], scale: f32) -> [[i8; N]; M] {
    w.map(|row| row.map(|v| quantize(v, scale, 0, -127, 127)))
}

fn rescales(label: &str, m: &[i32], s: &[i32]) -> String {
    let parts: Vec<String> = m.iter().zip(s).map(|(m, s)| format!("{m},{s}")).collect();
    format!(" {label}={}", parts.join(";"))
}

fn list(h: &[i8]) -> String {
    h.iter().map(i8::to_string).collect::<Vec<_>>().join(",")
}

fn flat<const N: usize, const M: usize, T: Copy>(x: &[[T; N]; M]) -> Vec<T> {
    x.iter().flatten().copied().collect()
}

fn lstm<C: Cell, const I: usize, const H: usize>(
    r: &mut Lcg,
    out: &mut String,
    name: &str,
    calibrated: bool,
    trial: usize,
) {
    let mut w_in = [[[0i8; I]; H]; 4];
    let mut w_rec = [[[0i8; H]; H]; 4];
    let mut bias = [[0i32; H]; 4];
    let mut sig = [0i8; 256];
    let mut tanh = [0i8; 256];
    let mut tanh_c = [0i8; 256];
    let kind = if calibrated { "calibrated" } else { "direct" };
    write!(out, "lstm {name} {kind} {trial}").unwrap();

    let input_zero_point;
    let hidden_zero_point;
    let cell_zero_point;
    let (mut in_m, mut in_s, mut rec_m, mut rec_s) = ([0; 4], [0; 4], [0; 4], [0; 4]);
    let (fc, ig, ct);
    let (cell_qmin, cell_qmax);
    let output_requantizer;
    if calibrated {
        let wf_in: [[[f32; I]; H]; 4] =
            core::array::from_fn(|_| core::array::from_fn(|_| r.reals()));
        let wf_rec: [[[f32; H]; H]; 4] =
            core::array::from_fn(|_| core::array::from_fn(|_| r.reals()));
        let bf: [[f32; H]; 4] = core::array::from_fn(|_| r.reals());
        let input_scale = r.scale(6);
        let hidden_scale = r.narrow_scale(6);
        let cell_scale = r.narrow_scale(if C::WIDE { 12 } else { 4 });
        let lut_input_scale = r.narrow_scale(3);
        let scales = LstmScales {
            input_scale,
            hidden_scale,
            cell_scale,
            lut_input_scale,
            input_weight_scales: core::array::from_fn(|g| absmax_scale(&flat(&wf_in[g]))),
            recurrent_weight_scales: core::array::from_fn(|g| absmax_scale(&flat(&wf_rec[g]))),
        };
        let p = build_lstm_params(&scales);
        bias = quantize_gate_biases(&bf, lut_input_scale);
        for g in 0..4 {
            w_in[g] = quantize_weights(&wf_in[g], scales.input_weight_scales[g]);
            w_rec[g] = quantize_weights(&wf_rec[g], scales.recurrent_weight_scales[g]);
        }
        sig = build_sigmoid_lut(lut_input_scale, 0, SIGMOID_LUT_OUTPUT_SCALE, -128);
        tanh = build_tanh_lut(lut_input_scale, 0, TANH_LUT_OUTPUT_SCALE, 0);
        tanh_c = build_tanh_lut(lut_input_scale, 0, TANH_LUT_OUTPUT_SCALE, 0);

        input_zero_point = r.i8();
        hidden_zero_point = r.i8() / 8;
        cell_zero_point = C::calibrated_zero_point(r);
        (in_m, in_s) = (p.input_to_lut_multiplier, p.input_to_lut_shift);
        (rec_m, rec_s) = (p.recurrent_to_lut_multiplier, p.recurrent_to_lut_shift);
        fc = (p.f_times_c_multiplier, p.f_times_c_shift);
        ig = (p.i_times_g_multiplier, p.i_times_g_shift);
        ct = (p.cell_to_tanh_multiplier, p.cell_to_tanh_shift);
        cell_qmin = if r.draw(2) == 0 {
            C::MIN_PLUS_ONE
        } else {
            C::MIN
        };
        cell_qmax = C::MAX;
        output_requantizer = Requantizer {
            multiplier: p.output_multiplier,
            shift: p.output_shift,
            zero_point: hidden_zero_point,
            qmin: -128,
            qmax: 127,
        };
        write!(
            out,
            " scales=0x{:08X},0x{:08X},0x{:08X},0x{:08X}",
            input_scale.to_bits(),
            hidden_scale.to_bits(),
            cell_scale.to_bits(),
            lut_input_scale.to_bits()
        )
        .unwrap();
    } else {
        for gate in &mut w_in {
            for row in gate {
                r.fill(row);
            }
        }
        for gate in &mut w_rec {
            for row in gate {
                r.fill(row);
            }
        }
        for gate in &mut bias {
            for b in gate {
                *b = r.draw(8193) as i32 - 4096;
            }
        }
        r.fill(&mut sig);
        r.fill(&mut tanh);
        r.fill(&mut tanh_c);
        input_zero_point = r.i8();
        hidden_zero_point = r.i8();
        cell_zero_point = C::direct_zero_point(r);
        for g in 0..4 {
            in_m[g] = r.multiplier();
            in_s[g] = r.shift(6, 8);
            rec_m[g] = r.multiplier();
            rec_s[g] = r.shift(6, 8);
        }
        fc = (r.multiplier(), r.shift(6, 8));
        ig = (r.multiplier(), r.shift(6, 8));
        ct = (r.multiplier(), r.shift(if C::WIDE { 6 } else { 0 }, 10));
        (cell_qmin, cell_qmax) = match r.draw(3) {
            0 => (C::MIN, C::MAX),
            1 => (C::MIN_PLUS_ONE, C::MAX),
            _ => {
                let lo = C::any(r);
                (lo, C::any(r))
            }
        };
        let multiplier = r.multiplier();
        let shift = r.shift(6, 8);
        let zero_point = r.i8();
        let (mut qmin, mut qmax) = (r.i8(), r.i8());
        if r.draw(2) == 0 {
            (qmin, qmax) = (-128, 127);
        }
        output_requantizer = Requantizer {
            multiplier,
            shift,
            zero_point,
            qmin,
            qmax,
        };
    }

    let cell: QLstmCell<'_, I, H, C> = QLstmCell {
        input_weights: &w_in,
        recurrent_weights: &w_rec,
        biases: &bias,
        input_zero_point,
        hidden_zero_point,
        cell_zero_point,
        input_to_lut_multiplier: in_m,
        input_to_lut_shift: in_s,
        recurrent_to_lut_multiplier: rec_m,
        recurrent_to_lut_shift: rec_s,
        sigmoid_lut: &sig,
        tanh_lut: &tanh,
        f_times_c_multiplier: fc.0,
        f_times_c_shift: fc.1,
        i_times_g_multiplier: ig.0,
        i_times_g_shift: ig.1,
        cell_qmin,
        cell_qmax,
        cell_to_tanh_multiplier: ct.0,
        cell_to_tanh_shift: ct.1,
        tanh_cell_lut: &tanh_c,
        output_requantizer,
    };

    out.push_str(&rescales("in", &in_m, &in_s));
    out.push_str(&rescales("rec", &rec_m, &rec_s));
    let rq = &output_requantizer;
    write!(
        out,
        " fc={},{} ig={},{} ct={},{} rq={},{},{},{},{}",
        fc.0,
        fc.1,
        ig.0,
        ig.1,
        ct.0,
        ct.1,
        rq.multiplier,
        rq.shift,
        rq.zero_point,
        rq.qmin,
        rq.qmax
    )
    .unwrap();
    write!(
        out,
        " zp={input_zero_point},{hidden_zero_point},{cell_zero_point} cell={cell_qmin},{cell_qmax}"
    )
    .unwrap();
    let w_hash = w_in
        .iter()
        .flatten()
        .fold(FNV_START, |h, row| fnv_i8(h, row));
    let w_hash = w_rec.iter().flatten().fold(w_hash, |h, row| fnv_i8(h, row));
    let b_hash = fnv_i32(FNV_START, &flat(&bias));
    let lut_hash = fnv_i8(fnv_i8(fnv_i8(FNV_START, &sig), &tanh), &tanh_c);
    write!(
        out,
        " w=0x{w_hash:08X} b=0x{b_hash:08X} lut=0x{lut_hash:08X}"
    )
    .unwrap();

    let mut h = [0i8; H];
    let mut c = [C::MIN; H];
    for k in 0..H {
        if trial == 0 {
            h[k] = hidden_zero_point;
            c[k] = cell_zero_point;
        } else {
            h[k] = r.i8();
            c[k] = C::any(r);
        }
    }
    let mut hash = FNV_START;
    for _ in 0..STEPS {
        let mut x = [0i8; I];
        r.fill(&mut x);
        cell.forward(&x, &mut h, &mut c);
        hash = fnv_i8(hash, &h);
        hash = C::hash(hash, &c);
    }
    writeln!(out, " h={} hash=0x{hash:08X}", list(&h)).unwrap();
}

fn gru<const I: usize, const H: usize>(
    r: &mut Lcg,
    out: &mut String,
    name: &str,
    calibrated: bool,
    trial: usize,
) {
    let mut w_in = [[[0i8; I]; H]; 3];
    let mut w_rec = [[[0i8; H]; H]; 3];
    let mut bias = [[0i32; H]; 3];
    let mut sig = [0i8; 256];
    let mut tanh = [0i8; 256];
    let kind = if calibrated { "calibrated" } else { "direct" };
    write!(out, "gru {name} {kind} {trial}").unwrap();

    let input_zero_point;
    let hidden_zero_point;
    let (mut in_m, mut in_s, mut rec_m, mut rec_s) = ([0; 3], [0; 3], [0; 3], [0; 3]);
    let (rh, zn, zh);
    let (output_qmin, output_qmax);
    if calibrated {
        let wf_in: [[[f32; I]; H]; 3] =
            core::array::from_fn(|_| core::array::from_fn(|_| r.reals()));
        let wf_rec: [[[f32; H]; H]; 3] =
            core::array::from_fn(|_| core::array::from_fn(|_| r.reals()));
        let bf: [[f32; H]; 3] = core::array::from_fn(|_| r.reals());
        let input_scale = r.scale(6);
        let hidden_scale = r.narrow_scale(6);
        let lut_input_scale = r.narrow_scale(3);
        let scales = GruScales {
            input_scale,
            hidden_scale,
            lut_input_scale,
            input_weight_scales: core::array::from_fn(|g| absmax_scale(&flat(&wf_in[g]))),
            recurrent_weight_scales: core::array::from_fn(|g| absmax_scale(&flat(&wf_rec[g]))),
        };
        let p = build_gru_params(&scales);
        bias = quantize_gate_biases(&bf, lut_input_scale);
        for g in 0..3 {
            w_in[g] = quantize_weights(&wf_in[g], scales.input_weight_scales[g]);
            w_rec[g] = quantize_weights(&wf_rec[g], scales.recurrent_weight_scales[g]);
        }
        sig = build_sigmoid_lut(lut_input_scale, 0, SIGMOID_LUT_OUTPUT_SCALE, -128);
        tanh = build_tanh_lut(lut_input_scale, 0, TANH_LUT_OUTPUT_SCALE, 0);

        input_zero_point = r.i8();
        hidden_zero_point = r.i8() / 8;
        (in_m, in_s) = (p.input_to_lut_multiplier, p.input_to_lut_shift);
        (rec_m, rec_s) = (p.recurrent_to_lut_multiplier, p.recurrent_to_lut_shift);
        rh = (p.r_times_h_multiplier, p.r_times_h_shift);
        zn = (
            p.one_minus_z_times_n_multiplier,
            p.one_minus_z_times_n_shift,
        );
        zh = (p.z_times_h_multiplier, p.z_times_h_shift);
        output_qmin = if r.draw(2) == 0 { -127 } else { -128 };
        output_qmax = 127;
        write!(
            out,
            " scales=0x{:08X},0x{:08X},0x{:08X}",
            input_scale.to_bits(),
            hidden_scale.to_bits(),
            lut_input_scale.to_bits()
        )
        .unwrap();
    } else {
        for gate in &mut w_in {
            for row in gate {
                r.fill(row);
            }
        }
        for gate in &mut w_rec {
            for row in gate {
                r.fill(row);
            }
        }
        for gate in &mut bias {
            for b in gate {
                *b = r.draw(8193) as i32 - 4096;
            }
        }
        r.fill(&mut sig);
        r.fill(&mut tanh);
        input_zero_point = r.i8();
        hidden_zero_point = r.i8();
        for g in 0..3 {
            in_m[g] = r.multiplier();
            in_s[g] = r.shift(6, 8);
            rec_m[g] = r.multiplier();
            rec_s[g] = r.shift(6, 8);
        }
        rh = (r.multiplier(), r.shift(6, 8));
        zn = (r.multiplier(), r.shift(6, 8));
        zh = (r.multiplier(), r.shift(6, 8));
        let (lo, hi) = (r.i8(), r.i8());
        (output_qmin, output_qmax) = if r.draw(2) == 0 {
            (-128, 127)
        } else {
            (lo, hi)
        };
    }

    let cell: QGruCell<'_, I, H> = QGruCell {
        input_weights: &w_in,
        recurrent_weights: &w_rec,
        biases: &bias,
        input_zero_point,
        hidden_zero_point,
        input_to_lut_multiplier: in_m,
        input_to_lut_shift: in_s,
        recurrent_to_lut_multiplier: rec_m,
        recurrent_to_lut_shift: rec_s,
        sigmoid_lut: &sig,
        tanh_lut: &tanh,
        r_times_h_multiplier: rh.0,
        r_times_h_shift: rh.1,
        one_minus_z_times_n_multiplier: zn.0,
        one_minus_z_times_n_shift: zn.1,
        z_times_h_multiplier: zh.0,
        z_times_h_shift: zh.1,
        output_qmin,
        output_qmax,
    };

    out.push_str(&rescales("in", &in_m, &in_s));
    out.push_str(&rescales("rec", &rec_m, &rec_s));
    write!(
        out,
        " rh={},{} zn={},{} zh={},{}",
        rh.0, rh.1, zn.0, zn.1, zh.0, zh.1
    )
    .unwrap();
    write!(
        out,
        " zp={input_zero_point},{hidden_zero_point} out={output_qmin},{output_qmax}"
    )
    .unwrap();
    let w_hash = w_in
        .iter()
        .flatten()
        .fold(FNV_START, |h, row| fnv_i8(h, row));
    let w_hash = w_rec.iter().flatten().fold(w_hash, |h, row| fnv_i8(h, row));
    let b_hash = fnv_i32(FNV_START, &flat(&bias));
    let lut_hash = fnv_i8(fnv_i8(FNV_START, &sig), &tanh);
    write!(
        out,
        " w=0x{w_hash:08X} b=0x{b_hash:08X} lut=0x{lut_hash:08X}"
    )
    .unwrap();

    let mut h = [0i8; H];
    for v in &mut h {
        *v = if trial == 0 {
            hidden_zero_point
        } else {
            r.i8()
        };
    }
    let mut hash = FNV_START;
    for _ in 0..STEPS {
        let mut x = [0i8; I];
        r.fill(&mut x);
        cell.forward(&x, &mut h);
        hash = fnv_i8(hash, &h);
    }
    writeln!(out, " h={} hash=0x{hash:08X}", list(&h)).unwrap();
}

fn lstm_shape<C: Cell, const I: usize, const H: usize>(r: &mut Lcg, out: &mut String, name: &str) {
    lstm::<C, I, H>(r, out, name, true, 0);
    lstm::<C, I, H>(r, out, name, true, 1);
    lstm::<C, I, H>(r, out, name, false, 0);
    lstm::<C, I, H>(r, out, name, false, 1);
}

fn gru_shape<const I: usize, const H: usize>(r: &mut Lcg, out: &mut String, name: &str) {
    gru::<I, H>(r, out, name, true, 0);
    gru::<I, H>(r, out, name, true, 1);
    gru::<I, H>(r, out, name, false, 0);
    gru::<I, H>(r, out, name, false, 1);
}

#[test]
fn recurrent_cells_match_cpp() {
    let mut r = Lcg(0x5EED_0005);
    let mut out = String::new();
    lstm_shape::<i8, 3, 4>(&mut r, &mut out, "3x4_i8");
    lstm_shape::<i16, 3, 4>(&mut r, &mut out, "3x4_i16");
    lstm_shape::<i8, 1, 1>(&mut r, &mut out, "1x1_i8");
    lstm_shape::<i16, 2, 3>(&mut r, &mut out, "2x3_i16");
    lstm_shape::<i8, 8, 16>(&mut r, &mut out, "8x16_i8");
    lstm_shape::<i16, 16, 8>(&mut r, &mut out, "16x8_i16");
    lstm_shape::<i8, 5, 7>(&mut r, &mut out, "5x7_i8");

    gru_shape::<3, 4>(&mut r, &mut out, "3x4");
    gru_shape::<1, 1>(&mut r, &mut out, "1x1");
    gru_shape::<2, 3>(&mut r, &mut out, "2x3");
    gru_shape::<8, 16>(&mut r, &mut out, "8x16");
    gru_shape::<16, 8>(&mut r, &mut out, "16x8");
    gru_shape::<5, 7>(&mut r, &mut out, "5x7");

    let golden = include_str!("golden/recurrent_int8/stdout.txt");
    for (i, (ours, theirs)) in out.lines().zip(golden.lines()).enumerate() {
        assert_eq!(ours, theirs, "line {}", i + 1);
    }
    assert_eq!(out, golden);
}
