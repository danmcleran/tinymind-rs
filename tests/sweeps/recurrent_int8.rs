// The int8 recurrent cell sweep (QLstmCell with int8 and int16 cell state,
// QGruCell), shared by tests/recurrent_int8_sweep.rs (on the host, against
// tinymind (C++)'s output, calibrating with int8::calibration) and
// bench/src/sweeps.rs (on Cortex-M: the direct-parameter trials only, see
// `Calibrate`). no_std and allocation-free.

use core::fmt::Write;

use tinymind::int8::{CellStorage, QGruCell, QLstmCell, Requantizer};

/// The scales a calibrated trial draws (`LstmScales` / `GruScales` without the
/// weight scales, which calibration takes from the float weights).
// The device's `Calibrate` ignores them.
#[cfg_attr(target_os = "none", allow(dead_code))]
#[derive(Clone, Copy)]
pub struct Scales {
    pub input: f32,
    pub hidden: f32,
    /// LSTM only (0 for a GRU).
    pub cell: f32,
    pub lut_input: f32,
}

/// What `int8::calibration` makes for a calibrated LSTM trial: the integer
/// parameters, the quantized weights and biases, and the LUTs.
pub struct LstmCalibration<const I: usize, const H: usize> {
    pub input_to_lut: ([i32; 4], [i32; 4]),
    pub recurrent_to_lut: ([i32; 4], [i32; 4]),
    pub f_times_c: (i32, i32),
    pub i_times_g: (i32, i32),
    pub cell_to_tanh: (i32, i32),
    pub output: (i32, i32),
    pub input_weights: [[[i8; I]; H]; 4],
    pub recurrent_weights: [[[i8; H]; H]; 4],
    pub biases: [[i32; H]; 4],
    pub sigmoid_lut: [i8; 256],
    pub tanh_lut: [i8; 256],
    pub tanh_cell_lut: [i8; 256],
}

/// The same for a calibrated GRU trial.
pub struct GruCalibration<const I: usize, const H: usize> {
    pub input_to_lut: ([i32; 3], [i32; 3]),
    pub recurrent_to_lut: ([i32; 3], [i32; 3]),
    pub r_times_h: (i32, i32),
    pub one_minus_z_times_n: (i32, i32),
    pub z_times_h: (i32, i32),
    pub input_weights: [[[i8; I]; H]; 3],
    pub recurrent_weights: [[[i8; H]; H]; 3],
    pub biases: [[i32; H]; 3],
    pub sigmoid_lut: [i8; 256],
    pub tanh_lut: [i8; 256],
}

/// What the sweep needs from `int8::calibration` (std only). The host
/// calibrates. The device cannot: the golden prints the calibrated weights,
/// biases and LUTs only as hashes, and rebuilding them takes float rounding,
/// exp and tanh. So there it returns `None`, the trial runs with zeroed
/// parameters only to keep the generator in step, and its line is not
/// compared (the host compares it).
pub trait Calibrate {
    fn lstm<const I: usize, const H: usize>(
        &mut self,
        scales: Scales,
        input_weights: &[[[f32; I]; H]; 4],
        recurrent_weights: &[[[f32; H]; H]; 4],
        biases: &[[f32; H]; 4],
    ) -> Option<LstmCalibration<I, H>>;

    fn gru<const I: usize, const H: usize>(
        &mut self,
        scales: Scales,
        input_weights: &[[[f32; I]; H]; 3],
        recurrent_weights: &[[[f32; H]; H]; 3],
        biases: &[[f32; H]; 3],
    ) -> Option<GruCalibration<I, H>>;
}

/// 2^-k as an f32 (k in 1..=126), exactly, without `powi` (std only).
fn pow2_neg(k: i32) -> f32 {
    f32::from_bits(((127 - k) as u32) << 23)
}

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
        m * pow2_neg(16 + base + self.draw(4) as i32)
    }

    /// (2^16 + d) * 2^-(17 + base): exact in f32.
    fn narrow_scale(&mut self, base: i32) -> f32 {
        let m = ((1 << 16) + self.draw(1 << 16)) as f32;
        m * pow2_neg(17 + base)
    }

    /// (d - 32768) * 2^-16, in [-0.5, 0.5).
    fn real(&mut self) -> f32 {
        (self.draw(65536) as i32 - 32768) as f32 * pow2_neg(16)
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
trait Cell: CellStorage + core::fmt::Display {
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

/// ` label=m,s;m,s;...`
fn rescales(out: &mut impl Write, label: &str, m: &[i32], s: &[i32]) {
    write!(out, " {label}=").unwrap();
    for (k, (m, s)) in m.iter().zip(s).enumerate() {
        let sep = if k == 0 { "" } else { ";" };
        write!(out, "{sep}{m},{s}").unwrap();
    }
}

/// `v,v,...`
fn list(out: &mut impl Write, h: &[i8]) {
    for (k, v) in h.iter().enumerate() {
        let sep = if k == 0 { "" } else { "," };
        write!(out, "{sep}{v}").unwrap();
    }
}

fn lstm<C: Cell, const I: usize, const H: usize>(
    r: &mut Lcg,
    out: &mut impl Write,
    cal: &mut impl Calibrate,
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
        let scales = Scales {
            input: input_scale,
            hidden: hidden_scale,
            cell: cell_scale,
            lut_input: lut_input_scale,
        };
        let mut output = (0, 0);
        if let Some(p) = cal.lstm(scales, &wf_in, &wf_rec, &bf) {
            (w_in, w_rec, bias) = (p.input_weights, p.recurrent_weights, p.biases);
            (sig, tanh, tanh_c) = (p.sigmoid_lut, p.tanh_lut, p.tanh_cell_lut);
            ((in_m, in_s), (rec_m, rec_s)) = (p.input_to_lut, p.recurrent_to_lut);
            (fc, ig, ct) = (p.f_times_c, p.i_times_g, p.cell_to_tanh);
            output = p.output;
        } else {
            (fc, ig, ct) = ((0, 0), (0, 0), (0, 0));
        }

        input_zero_point = r.i8();
        hidden_zero_point = r.i8() / 8;
        cell_zero_point = C::calibrated_zero_point(r);
        cell_qmin = if r.draw(2) == 0 {
            C::MIN_PLUS_ONE
        } else {
            C::MIN
        };
        cell_qmax = C::MAX;
        output_requantizer = Requantizer {
            multiplier: output.0,
            shift: output.1,
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

    rescales(out, "in", &in_m, &in_s);
    rescales(out, "rec", &rec_m, &rec_s);
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
    let b_hash = bias.iter().fold(FNV_START, |h, gate| fnv_i32(h, gate));
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
    write!(out, " h=").unwrap();
    list(out, &h);
    writeln!(out, " hash=0x{hash:08X}").unwrap();
}

fn gru<const I: usize, const H: usize>(
    r: &mut Lcg,
    out: &mut impl Write,
    cal: &mut impl Calibrate,
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
        let scales = Scales {
            input: input_scale,
            hidden: hidden_scale,
            cell: 0.0,
            lut_input: lut_input_scale,
        };
        if let Some(p) = cal.gru(scales, &wf_in, &wf_rec, &bf) {
            (w_in, w_rec, bias) = (p.input_weights, p.recurrent_weights, p.biases);
            (sig, tanh) = (p.sigmoid_lut, p.tanh_lut);
            ((in_m, in_s), (rec_m, rec_s)) = (p.input_to_lut, p.recurrent_to_lut);
            (rh, zn, zh) = (p.r_times_h, p.one_minus_z_times_n, p.z_times_h);
        } else {
            (rh, zn, zh) = ((0, 0), (0, 0), (0, 0));
        }

        input_zero_point = r.i8();
        hidden_zero_point = r.i8() / 8;
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

    rescales(out, "in", &in_m, &in_s);
    rescales(out, "rec", &rec_m, &rec_s);
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
    let b_hash = bias.iter().fold(FNV_START, |h, gate| fnv_i32(h, gate));
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
    write!(out, " h=").unwrap();
    list(out, &h);
    writeln!(out, " hash=0x{hash:08X}").unwrap();
}

fn lstm_shape<C: Cell, const I: usize, const H: usize>(
    r: &mut Lcg,
    out: &mut impl Write,
    cal: &mut impl Calibrate,
    name: &str,
) {
    lstm::<C, I, H>(r, out, cal, name, true, 0);
    lstm::<C, I, H>(r, out, cal, name, true, 1);
    lstm::<C, I, H>(r, out, cal, name, false, 0);
    lstm::<C, I, H>(r, out, cal, name, false, 1);
}

fn gru_shape<const I: usize, const H: usize>(
    r: &mut Lcg,
    out: &mut impl Write,
    cal: &mut impl Calibrate,
    name: &str,
) {
    gru::<I, H>(r, out, cal, name, true, 0);
    gru::<I, H>(r, out, cal, name, true, 1);
    gru::<I, H>(r, out, cal, name, false, 0);
    gru::<I, H>(r, out, cal, name, false, 1);
}

/// Write the sweep's report (tests/golden/recurrent_int8/stdout.txt).
pub fn report(out: &mut impl Write, cal: &mut impl Calibrate) {
    let mut r = Lcg(0x5EED_0005);
    lstm_shape::<i8, 3, 4>(&mut r, out, cal, "3x4_i8");
    lstm_shape::<i16, 3, 4>(&mut r, out, cal, "3x4_i16");
    lstm_shape::<i8, 1, 1>(&mut r, out, cal, "1x1_i8");
    lstm_shape::<i16, 2, 3>(&mut r, out, cal, "2x3_i16");
    lstm_shape::<i8, 8, 16>(&mut r, out, cal, "8x16_i8");
    lstm_shape::<i16, 16, 8>(&mut r, out, cal, "16x8_i16");
    lstm_shape::<i8, 5, 7>(&mut r, out, cal, "5x7_i8");

    gru_shape::<3, 4>(&mut r, out, cal, "3x4");
    gru_shape::<1, 1>(&mut r, out, cal, "1x1");
    gru_shape::<2, 3>(&mut r, out, cal, "2x3");
    gru_shape::<8, 16>(&mut r, out, cal, "8x16");
    gru_shape::<16, 8>(&mut r, out, cal, "16x8");
    gru_shape::<5, 7>(&mut r, out, cal, "5x7");
}
