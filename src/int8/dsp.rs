//! EXPERIMENT (branch `exp-smlad-asm`, not for release): the int8 dot products
//! on the Arm DSP extension's SMLAD, written as tinymind (C++)'s
//! `simd_arm_dsp.hpp` writes them. Four int8 values come from one 32-bit
//! load; SXTB16 sign-extends bytes 0 and 2 into two i16 lanes, SXTB16 with
//! ROR #8 bytes 1 and 3; SXTAB16 adds a packed (-zp, -zp) while it extends;
//! SMLAD adds two 16x16 products to an i32. Rust's nightly `core::arch::arm`
//! has `__smlad` but no `__sxtb16`/`__sxtab16`, so this uses stable `asm!`,
//! which needs `unsafe`. Only for cores with the DSP extension (M4, M7, M33):
//! feature `arm-dsp-asm` on any other Arm core gives an illegal instruction.
//!
//! The products are the same as the scalar code's, added in another order;
//! `i32` sums wrap, so every result is the same modulo 2^32.
#![allow(unsafe_code)]

use core::arch::asm;

/// Four int8 values as one word, byte 0 lowest (one LDR on Armv7E-M).
#[inline(always)]
fn word(b: &[i8; 4]) -> u32 {
    u32::from_le_bytes([b[0] as u8, b[1] as u8, b[2] as u8, b[3] as u8])
}

/// Bytes 0 and 2, sign-extended into two i16 lanes.
#[inline(always)]
fn sxtb16(v: u32) -> u32 {
    let r: u32;
    // SAFETY: a register-only DSP instruction; no memory, stack or flags.
    unsafe {
        asm!("sxtb16 {r}, {v}", r = lateout(reg) r, v = in(reg) v,
             options(pure, nomem, nostack, preserves_flags));
    }
    r
}

/// Bytes 1 and 3, sign-extended into two i16 lanes.
#[inline(always)]
fn sxtb16_ror8(v: u32) -> u32 {
    let r: u32;
    // SAFETY: as `sxtb16`.
    unsafe {
        asm!("sxtb16 {r}, {v}, ror #8", r = lateout(reg) r, v = in(reg) v,
             options(pure, nomem, nostack, preserves_flags));
    }
    r
}

/// `a + sxtb16(v)`, lane by lane.
#[inline(always)]
fn sxtab16(a: u32, v: u32) -> u32 {
    let r: u32;
    // SAFETY: as `sxtb16`.
    unsafe {
        asm!("sxtab16 {r}, {a}, {v}", r = lateout(reg) r, a = in(reg) a, v = in(reg) v,
             options(pure, nomem, nostack, preserves_flags));
    }
    r
}

/// `a + sxtb16_ror8(v)`, lane by lane.
#[inline(always)]
fn sxtab16_ror8(a: u32, v: u32) -> u32 {
    let r: u32;
    // SAFETY: as `sxtb16`.
    unsafe {
        asm!("sxtab16 {r}, {a}, {v}, ror #8", r = lateout(reg) r, a = in(reg) a, v = in(reg) v,
             options(pure, nomem, nostack, preserves_flags));
    }
    r
}

/// `acc + a.lo * b.lo + a.hi * b.hi` (i16 lanes), wrapping. SMLAD sets the Q
/// flag on overflow, which nothing reads, so flags are not preserved.
#[inline(always)]
fn smlad(a: u32, b: u32, acc: i32) -> i32 {
    let r: i32;
    // SAFETY: a register-only DSP instruction; it writes only the Q flag.
    unsafe {
        asm!("smlad {r}, {a}, {b}, {c}", r = lateout(reg) r, a = in(reg) a, b = in(reg) b,
             c = in(reg) acc, options(pure, nomem, nostack));
    }
    r
}

/// `(-zp, -zp)` as two i16 lanes.
#[inline(always)]
fn neg_zp_pair(zero_point: i8) -> u32 {
    let n = u32::from((-i16::from(zero_point)) as u16);
    n | (n << 16)
}

/// `sum x * w`, wrapping.
#[inline(always)]
pub(crate) fn dot<const N: usize>(x: &[i8; N], w: &[i8; N]) -> i32 {
    let (xs, xt) = x.as_chunks::<4>();
    let (ws, wt) = w.as_chunks::<4>();
    let mut acc = 0i32;
    for (x, w) in xs.iter().zip(ws) {
        let (x, w) = (word(x), word(w));
        acc = smlad(sxtb16(x), sxtb16(w), acc);
        acc = smlad(sxtb16_ror8(x), sxtb16_ror8(w), acc);
    }
    for (&x, &w) in xt.iter().zip(wt) {
        acc = acc.wrapping_add(i32::from(x) * i32::from(w));
    }
    acc
}

/// `sum w * (x - zero_point)`, wrapping.
#[inline(always)]
pub(crate) fn dot_with_zero_point<const N: usize>(x: &[i8; N], w: &[i8; N], zero_point: i8) -> i32 {
    let nzp = neg_zp_pair(zero_point);
    let (xs, xt) = x.as_chunks::<4>();
    let (ws, wt) = w.as_chunks::<4>();
    let mut acc = 0i32;
    for (x, w) in xs.iter().zip(ws) {
        let (x, w) = (word(x), word(w));
        acc = smlad(sxtb16(w), sxtab16(nzp, x), acc);
        acc = smlad(sxtb16_ror8(w), sxtab16_ror8(nzp, x), acc);
    }
    for (&x, &w) in xt.iter().zip(wt) {
        acc = acc.wrapping_add(i32::from(w) * (i32::from(x) - i32::from(zero_point)));
    }
    acc
}

/// `(sum x0 * w, sum x1 * w)`: two pixels against one filter, each widened
/// weight pair spent on both.
#[inline(always)]
pub(crate) fn dot2_pixels<const N: usize>(x0: &[i8; N], x1: &[i8; N], w: &[i8; N]) -> (i32, i32) {
    let (x0s, x0t) = x0.as_chunks::<4>();
    let (x1s, x1t) = x1.as_chunks::<4>();
    let (ws, wt) = w.as_chunks::<4>();
    let (mut a0, mut a1) = (0i32, 0i32);
    for ((x0, x1), w) in x0s.iter().zip(x1s).zip(ws) {
        let w = word(w);
        let (x0, x1) = (word(x0), word(x1));
        let we = sxtb16(w);
        a0 = smlad(we, sxtb16(x0), a0);
        a1 = smlad(we, sxtb16(x1), a1);
        let wo = sxtb16_ror8(w);
        a0 = smlad(wo, sxtb16_ror8(x0), a0);
        a1 = smlad(wo, sxtb16_ror8(x1), a1);
    }
    for ((&x0, &x1), &w) in x0t.iter().zip(x1t).zip(wt) {
        let w = i32::from(w);
        a0 = a0.wrapping_add(w * i32::from(x0));
        a1 = a1.wrapping_add(w * i32::from(x1));
    }
    (a0, a1)
}

/// `(sum x * w0, sum x * w1)` over the shortest of the three: one window
/// against two filters, each widened input pair spent on both.
#[inline(always)]
pub(crate) fn dot2(x: &[i8], w0: &[i8], w1: &[i8]) -> (i32, i32) {
    let n = x.len().min(w0.len()).min(w1.len());
    let (x, w0, w1) = (&x[..n], &w0[..n], &w1[..n]);
    let (xs, xt) = x.as_chunks::<4>();
    let (w0s, w0t) = w0.as_chunks::<4>();
    let (w1s, w1t) = w1.as_chunks::<4>();
    let (mut a0, mut a1) = (0i32, 0i32);
    for ((x, w0), w1) in xs.iter().zip(w0s).zip(w1s) {
        let x = word(x);
        let (w0, w1) = (word(w0), word(w1));
        let xe = sxtb16(x);
        a0 = smlad(xe, sxtb16(w0), a0);
        a1 = smlad(xe, sxtb16(w1), a1);
        let xo = sxtb16_ror8(x);
        a0 = smlad(xo, sxtb16_ror8(w0), a0);
        a1 = smlad(xo, sxtb16_ror8(w1), a1);
    }
    for ((&x, &w0), &w1) in xt.iter().zip(w0t).zip(w1t) {
        let x = i32::from(x);
        a0 = a0.wrapping_add(x * i32::from(w0));
        a1 = a1.wrapping_add(x * i32::from(w1));
    }
    (a0, a1)
}

/// `acc + sum x * w` over i16 values already widened, two lanes per SMLAD.
#[inline(always)]
pub(crate) fn dot16(mut acc: i32, x: &[i16], w: &[i16]) -> i32 {
    let n = x.len().min(w.len());
    let (x, w) = (&x[..n], &w[..n]);
    let (xs, xt) = x.as_chunks::<2>();
    let (ws, wt) = w.as_chunks::<2>();
    for (x, w) in xs.iter().zip(ws) {
        let x = u32::from(x[0] as u16) | (u32::from(x[1] as u16) << 16);
        let w = u32::from(w[0] as u16) | (u32::from(w[1] as u16) << 16);
        acc = smlad(x, w, acc);
    }
    for (&x, &w) in xt.iter().zip(wt) {
        acc = acc.wrapping_add(i32::from(x) * i32::from(w));
    }
    acc
}
