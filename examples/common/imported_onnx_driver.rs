//! The imported_onnx example's program, shared by `examples/imported_onnx.rs`
//! and `tests/imported_onnx_golden.rs` (which runs it in-process against the
//! C++ output).

use std::io::{self, Write};

#[path = "imported_onnx_cnn_model.rs"]
mod cnn;

#[path = "imported_onnx_cnn_qop_model.rs"]
mod cnn_qop;

#[path = "imported_onnx_mlp_model.rs"]
mod mlp;

#[path = "imported_onnx_resnet_model.rs"]
mod resnet;

#[path = "imported_onnx_resnet_qop_model.rs"]
mod resnet_qop;
// The same models with tinymind (C++)'s code generator's options: int4 weights
// in every dense and convolution layer, or int16 activations (16x8).
#[path = "imported_onnx_cnn_16x8_model.rs"]
mod cnn_16x8;
#[path = "imported_onnx_cnn_int4_model.rs"]
mod cnn_int4;
#[path = "imported_onnx_resnet_int4_model.rs"]
mod resnet_int4;
#[path = "imported_onnx_resnet_qop_16x8_model.rs"]
mod resnet_qop_16x8;

/// Test inputs per model: even-numbered ones span the whole int8 range, odd
/// ones stay within 48 steps of the input's zero point, where the calibrated
/// ranges are.
const SAMPLES: usize = 16;

/// The inputs' generator: a 32-bit linear congruential generator
/// (Numerical Recipes' constants), the same in `scripts/cpp/imported_onnx.cpp`.
struct Lcg(u32);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.0 >> 24
    }
}

/// Fill `x` with sample `s` of a model whose input zero point is `zp`.
fn fill(rng: &mut Lcg, s: usize, zp: i8, x: &mut [i8]) {
    for v in x {
        let r = rng.next();
        *v = if s % 2 == 0 {
            (r as i32 - 128) as i8
        } else {
            (i32::from(zp) + (r % 97) as i32 - 48).clamp(-128, 127) as i8
        };
    }
}

/// Fill `x` with sample `s` of a 16x8 model (input zero point 0): even
/// samples span the whole int16 range, odd ones stay within 2,000 steps of 0.
fn fill16(rng: &mut Lcg, s: usize, x: &mut [i16]) {
    for v in x {
        let r = (rng.next() << 8) | rng.next();
        *v = if s % 2 == 0 {
            (r as i32 - 32768) as i16
        } else {
            ((r % 4001) as i32 - 2000) as i16
        };
    }
}

fn print<T: ToString>(out: &mut dyn Write, model: &str, s: usize, y: &[T]) -> io::Result<()> {
    let v: Vec<String> = y.iter().map(T::to_string).collect();
    writeln!(out, "{model} {s}: {}", v.join(" "))
}

/// For each model: its input grid (the scale's bits), then the int8 output of
/// every test input.
pub fn run(out: &mut dyn Write) -> io::Result<()> {
    let mut rng = Lcg(1);

    writeln!(
        out,
        "onnx_cnn input zero_point={} scale=0x{:08x}",
        cnn::INPUT_ZERO_POINT,
        cnn::INPUT_SCALE.to_bits()
    )?;
    for s in 0..SAMPLES {
        let mut x = cnn::Input::default();
        fill(
            &mut rng,
            s,
            cnn::INPUT_ZERO_POINT,
            x.as_flattened_mut().as_flattened_mut(),
        );
        print(out, "onnx_cnn", s, &cnn::forward(&x))?;
    }

    writeln!(
        out,
        "onnx_cnn_qop input zero_point={} scale=0x{:08x}",
        cnn_qop::INPUT_ZERO_POINT,
        cnn_qop::INPUT_SCALE.to_bits()
    )?;
    for s in 0..SAMPLES {
        let mut x = cnn_qop::Input::default();
        fill(
            &mut rng,
            s,
            cnn_qop::INPUT_ZERO_POINT,
            x.as_flattened_mut().as_flattened_mut(),
        );
        print(out, "onnx_cnn_qop", s, &cnn_qop::forward(&x))?;
    }

    writeln!(
        out,
        "onnx_mlp input zero_point={} scale=0x{:08x}",
        mlp::INPUT_ZERO_POINT,
        mlp::INPUT_SCALE.to_bits()
    )?;
    for s in 0..SAMPLES {
        let mut x = mlp::Input::default();
        fill(&mut rng, s, mlp::INPUT_ZERO_POINT, &mut x);
        print(out, "onnx_mlp", s, &mlp::forward(&x))?;
    }

    writeln!(
        out,
        "onnx_resnet input zero_point={} scale=0x{:08x}",
        resnet::INPUT_ZERO_POINT,
        resnet::INPUT_SCALE.to_bits()
    )?;
    for s in 0..SAMPLES {
        let mut x = resnet::Input::default();
        fill(
            &mut rng,
            s,
            resnet::INPUT_ZERO_POINT,
            x.as_flattened_mut().as_flattened_mut(),
        );
        print(out, "onnx_resnet", s, &resnet::forward(&x))?;
    }

    writeln!(
        out,
        "onnx_resnet_qop input zero_point={} scale=0x{:08x}",
        resnet_qop::INPUT_ZERO_POINT,
        resnet_qop::INPUT_SCALE.to_bits()
    )?;
    for s in 0..SAMPLES {
        let mut x = resnet_qop::Input::default();
        fill(
            &mut rng,
            s,
            resnet_qop::INPUT_ZERO_POINT,
            x.as_flattened_mut().as_flattened_mut(),
        );
        print(out, "onnx_resnet_qop", s, &resnet_qop::forward(&x))?;
    }
    writeln!(
        out,
        "onnx_cnn_int4 input zero_point={} scale=0x{:08x}",
        cnn_int4::INPUT_ZERO_POINT,
        cnn_int4::INPUT_SCALE.to_bits()
    )?;
    for s in 0..SAMPLES {
        let mut x = cnn_int4::Input::default();
        fill(
            &mut rng,
            s,
            cnn_int4::INPUT_ZERO_POINT,
            x.as_flattened_mut().as_flattened_mut(),
        );
        print(out, "onnx_cnn_int4", s, &cnn_int4::forward(&x))?;
    }
    writeln!(
        out,
        "onnx_resnet_int4 input zero_point={} scale=0x{:08x}",
        resnet_int4::INPUT_ZERO_POINT,
        resnet_int4::INPUT_SCALE.to_bits()
    )?;
    for s in 0..SAMPLES {
        let mut x = resnet_int4::Input::default();
        fill(
            &mut rng,
            s,
            resnet_int4::INPUT_ZERO_POINT,
            x.as_flattened_mut().as_flattened_mut(),
        );
        print(out, "onnx_resnet_int4", s, &resnet_int4::forward(&x))?;
    }
    writeln!(
        out,
        "onnx_cnn_16x8 input zero_point={} scale=0x{:08x}",
        cnn_16x8::INPUT_ZERO_POINT,
        cnn_16x8::INPUT_SCALE.to_bits()
    )?;
    for s in 0..SAMPLES {
        let mut x = cnn_16x8::Input::default();
        fill16(&mut rng, s, x.as_flattened_mut().as_flattened_mut());
        print(out, "onnx_cnn_16x8", s, &cnn_16x8::forward(&x))?;
    }
    writeln!(
        out,
        "onnx_resnet_qop_16x8 input zero_point={} scale=0x{:08x}",
        resnet_qop_16x8::INPUT_ZERO_POINT,
        resnet_qop_16x8::INPUT_SCALE.to_bits()
    )?;
    for s in 0..SAMPLES {
        let mut x = resnet_qop_16x8::Input::default();
        fill16(&mut rng, s, x.as_flattened_mut().as_flattened_mut());
        print(
            out,
            "onnx_resnet_qop_16x8",
            s,
            &resnet_qop_16x8::forward(&x),
        )?;
    }
    Ok(())
}
