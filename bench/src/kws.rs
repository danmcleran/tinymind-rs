//! tinymind's KWS-shaped int8 CNN from its framework comparison (conv 3x3x8 per
//! channel, 2x2 max pool, 3x3 depthwise, pointwise 8->16, global average pool,
//! pointwise 16->10), on the same weights and inputs. The output hash must equal
//! tinymind's expected hash, which its C++ and TFLM produce on these boards.
//!
//! The layers take typed tensors, so one buffer cannot be reused for tensors of
//! different shapes as the C++ ping-pongs two byte buffers. Each tensor lives in
//! the narrowest scope instead, and LLVM overlaps the stack slots of tensors
//! whose lifetimes do not overlap: the peak is the conv output plus the pooled
//! tensor, as in the C++.

use core::fmt::Write;
use core::hint::black_box;

use tinymind::int8::{
    QConv2DPerChannel, QDepthwiseConv2D, QGlobalAvgPool2D, QMaxPool2D, QPointwiseConv2DPerChannel,
};

use crate::board::{self, Uart};
use crate::{REPS, median};

#[path = "gen/kws.rs"]
mod model;
use model::*;

type Input = [[[i8; 1]; 20]; 20];

const C1: QConv2DPerChannel<'static, 1, 3, 3, 8> = QConv2DPerChannel {
    weights: &W_C1,
    biases: Some(&B_C1),
    input_zero_point: ZP_C1,
    requantizers: &R_C1,
};
const P1: QMaxPool2D<2, 2> = QMaxPool2D;
const DW: QDepthwiseConv2D<'static, 8, 3, 3> = QDepthwiseConv2D {
    weights: &W_DW,
    biases: Some(&B_DW),
    input_zero_point: ZP_DW,
    requantizers: &R_DW,
};
const PW: QPointwiseConv2DPerChannel<'static, 8, 16> = QPointwiseConv2DPerChannel {
    weights: &W_PW,
    biases: Some(&B_PW),
    input_zero_point: ZP_PW,
    requantizers: &R_PW,
};
const GAP: QGlobalAvgPool2D = QGlobalAvgPool2D {
    qmin: GAP_QMIN,
    qmax: GAP_QMAX,
};
const D: QPointwiseConv2DPerChannel<'static, 16, 10> = QPointwiseConv2DPerChannel {
    weights: &W_D,
    biases: Some(&B_D),
    input_zero_point: ZP_D,
    requantizers: &R_D,
};

#[inline(never)]
fn infer(x: &Input) -> [i8; 10] {
    let pooled = {
        let mut c1 = [[[0i8; 8]; 18]; 18];
        C1.forward(x, &mut c1);
        let mut p1 = [[[0i8; 8]; 9]; 9];
        P1.forward(&c1, &mut p1);
        p1
    };
    let mixed = {
        let mut dw = [[[0i8; 8]; 7]; 7];
        DW.forward(&pooled, &mut dw);
        let mut pw = [[[0i8; 16]; 7]; 7];
        PW.forward(&dw, &mut pw);
        pw
    };
    let mut out = [[[0i8; 10]; 1]; 1];
    D.forward(&[[GAP.forward(&mixed)]], &mut out);
    out[0][0]
}

/// Time `f` on sample `r % N_TEST` for each repetition.
fn time(mut f: impl FnMut(&Input)) -> (u32, u32, u32) {
    let mut t = [0u32; REPS];
    for (r, slot) in t.iter_mut().enumerate() {
        let x = black_box(&X[r % N_TEST]);
        let a = board::cycles();
        f(x);
        *slot = board::cycles().wrapping_sub(a);
    }
    median(t)
}

pub fn run(out: &mut Uart) {
    let _ = writeln!(out, "name=kws-int8-per-channel");
    black_box(infer(black_box(&X[0]))); // warm-up
    let (min, med, max) = time(|x| {
        black_box(infer(x));
    });
    let _ = writeln!(out, "infer_min={min}\ninfer_med={med}\ninfer_max={max}");

    // Per layer, each timed on the activations it actually sees (a separate
    // image, as tinymind's PROFILE build, so the whole-model timing above is
    // of the plain image).
    #[cfg(feature = "kws-profile")]
    profile(out);

    // Every test input: tinymind's FNV-1a hash over every output byte.
    let mut h: u32 = 0x811C_9DC5;
    for x in &X {
        for v in infer(x) {
            h ^= u32::from(v as u8);
            h = h.wrapping_mul(0x0100_0193);
        }
    }
    let _ = writeln!(out, "n_test={N_TEST}");
    let _ = writeln!(
        out,
        "hash=0x{h:08X}\nexpect_hash=0x{EXPECT_HASH:08X}\nhash_match={}",
        u32::from(h == EXPECT_HASH)
    );
}

/// Per-layer cycles, each layer timed on the activations it actually sees. Its
/// own frame, not inlined: with the tensors in `run`'s frame, `infer`'s frame on
/// top of them overflowed the L031's 8 KB of RAM. Two tensors at a time at most,
/// as in `infer`.
#[cfg(feature = "kws-profile")]
#[inline(never)]
fn profile(out: &mut Uart) {
    let (l0, l1, pooled) = {
        let mut c1 = [[[0i8; 8]; 18]; 18];
        let (_, l0, _) = time(|x| C1.forward(x, black_box(&mut c1)));
        C1.forward(&X[0], &mut c1);
        let mut p1 = [[[0i8; 8]; 9]; 9];
        let (_, l1, _) = time(|_| P1.forward(black_box(&c1), black_box(&mut p1)));
        P1.forward(&c1, &mut p1);
        (l0, l1, p1)
    };
    let (l2, l3, mixed) = {
        let mut dw = [[[0i8; 8]; 7]; 7];
        let (_, l2, _) = time(|_| DW.forward(black_box(&pooled), black_box(&mut dw)));
        DW.forward(&pooled, &mut dw);
        let mut pw = [[[0i8; 16]; 7]; 7];
        let (_, l3, _) = time(|_| PW.forward(black_box(&dw), black_box(&mut pw)));
        PW.forward(&dw, &mut pw);
        (l2, l3, pw)
    };
    let (_, l4, _) = time(|_| {
        black_box(GAP.forward(black_box(&mixed)));
    });
    let g = [[GAP.forward(&mixed)]];
    let mut d = [[[0i8; 10]; 1]; 1];
    let (_, l5, _) = time(|_| D.forward(black_box(&g), black_box(&mut d)));
    let _ = writeln!(
        out,
        "layer.0.conv3x3={l0}\nlayer.1.maxpool={l1}\nlayer.2.dwconv3x3={l2}\nlayer.3.pwconv={l3}\nlayer.4.avgpool={l4}\nlayer.5.fc={l5}"
    );
}
