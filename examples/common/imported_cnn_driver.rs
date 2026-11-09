//! The imported_cnn example's program, shared by `examples/imported_cnn.rs` and
//! `tests/imported_cnn_golden.rs` (which runs it in-process against the C++
//! output).

use std::io::{self, Write};

use tinymind::int8::calibration::quantize;
use tinymind::int8::{Requantizer, qrelu};

#[path = "imported_cnn_model.rs"]
mod cnn;

#[path = "imported_cnn_flat_model.rs"]
mod cnn_flat;

/// What the C++ check prints, model by model: each layer's integer
/// requantizer (multiplier, shift, zero point: a ReLU is folded into the
/// requantizer's lower bound here and applied after it in the C++, so the
/// bounds differ), the softmax parameters or a hash of the tanh table, then for
/// 16 test inputs a hash of every intermediate tensor and the outputs.
pub fn run(out: &mut dyn Write) -> io::Result<()> {
    run_cnn(out)?;
    run_cnn_flat(out)
}

fn print_rq(out: &mut dyn Write, model: &str, name: &str, rq: &Requantizer) -> io::Result<()> {
    writeln!(
        out,
        "{model} {name} rq={},{},{}",
        rq.multiplier, rq.shift, rq.zero_point
    )
}

/// Test input `s`, element `k` of the flat NHWC tensor, as the C++ computes it.
fn input<const H: usize, const W: usize, const C: usize>(
    s: usize,
    scale: f32,
    zero_point: i8,
) -> [[[i8; C]; W]; H] {
    let mut x = [[[0i8; C]; W]; H];
    for (k, v) in x.iter_mut().flatten().flatten().enumerate() {
        let real = ((s * 23 + k * 29 + s * k * 7) % 37) as f32 / 18.0 - 1.0;
        *v = quantize(real, scale, i32::from(zero_point), -128, 127);
    }
    x
}

/// FNV-1a over every byte of the tensors, continuing from `h`.
fn fnv<'a>(h: u32, tensor: impl IntoIterator<Item = &'a i8>) -> u32 {
    tensor.into_iter().fold(h, |h, &v| {
        (h ^ u32::from(v as u8)).wrapping_mul(0x0100_0193)
    })
}

fn run_cnn(out: &mut dyn Write) -> io::Result<()> {
    print_rq(out, "cnn", "conv1", &cnn::BN1.requantizer)?;
    print_rq(out, "cnn", "conv2", &cnn::BN2_REQUANTIZERS[0])?;
    print_rq(out, "cnn", "conv3", &cnn::CONV3.requantizer)?;
    print_rq(out, "cnn", "fc1", &cnn::FC1.requantizer)?;
    let p = cnn::PROBS;
    writeln!(
        out,
        "cnn softmax={},{},{}",
        p.input_multiplier, p.input_left_shift, p.diff_min
    )?;
    for s in 0..16 {
        let x: cnn::Input = input(s, cnn::INPUT_SCALE, cnn::INPUT_ZERO_POINT);
        // Layer by layer, for the hash of every intermediate tensor.
        let mut a1 = [[[0i8; 4]; 12]; 12];
        cnn::BN1.forward(&x, &mut a1);
        let mut p1 = [[[0i8; 4]; 6]; 6];
        cnn::POOL1.forward(&a1, &mut p1);
        let mut a2 = [[[0i8; 4]; 6]; 6];
        cnn::BN2.forward(&p1, &mut a2);
        let mut a3 = [[[0i8; 8]; 6]; 6];
        cnn::CONV3.forward(&a2, &mut a3);
        let g = cnn::GAP.forward(&a3);
        let mut h = 0x811C_9DC5;
        h = fnv(h, a1.iter().flatten().flatten());
        h = fnv(h, p1.iter().flatten().flatten());
        h = fnv(h, a2.iter().flatten().flatten());
        h = fnv(h, a3.iter().flatten().flatten());
        h = fnv(h, &g);
        // The generated forward for the output.
        let y = cnn::forward(&x);
        let mut best = 0;
        for (k, &v) in y.iter().enumerate() {
            if v > y[best] {
                best = k;
            }
        }
        writeln!(
            out,
            "cnn s={s} layers=0x{h:08X} probs={},{},{} class={best}",
            y[0], y[1], y[2]
        )?;
    }
    Ok(())
}

fn run_cnn_flat(out: &mut dyn Write) -> io::Result<()> {
    print_rq(out, "cnn_flat", "conv1", &cnn_flat::CONV1.requantizer)?;
    print_rq(out, "cnn_flat", "fc1", &cnn_flat::FC1.requantizer)?;
    print_rq(out, "cnn_flat", "fc2", &cnn_flat::FC2.requantizer)?;
    writeln!(
        out,
        "cnn_flat lut fnv=0x{:08X}",
        fnv(0x811C_9DC5, &cnn_flat::OUT_LUT)
    )?;
    for s in 0..16 {
        let x: cnn_flat::Input = input(s, cnn_flat::INPUT_SCALE, cnn_flat::INPUT_ZERO_POINT);
        let mut a1 = [[[0i8; 5]; 10]; 4];
        cnn_flat::CONV1.forward(&x, &mut a1);
        let mut p1 = [[[0i8; 5]; 5]; 2];
        cnn_flat::POOL1.forward(&a1, &mut p1);
        let mut flat = [0i8; 50];
        for (o, &v) in flat.iter_mut().zip(p1.iter().flatten().flatten()) {
            *o = v;
        }
        let h1 = cnn_flat::FC1
            .forward(&flat)
            .map(|v| qrelu(v, cnn_flat::RELU2_ZERO_POINT));
        let lg = cnn_flat::FC2.forward(&h1);
        let mut h = 0x811C_9DC5;
        h = fnv(h, a1.iter().flatten().flatten());
        h = fnv(h, p1.iter().flatten().flatten());
        h = fnv(h, &h1);
        h = fnv(h, &lg);
        let y = cnn_flat::forward(&x);
        writeln!(out, "cnn_flat s={s} layers=0x{h:08X} out={},{}", y[0], y[1])?;
    }
    Ok(())
}
