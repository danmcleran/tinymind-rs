//! The int8 GBDT example's program, shared by `examples/gbdt_tabular_int8.rs`
//! and `tests/gbdt_golden.rs` (which runs it in-process against the C++
//! goldens). Port of tinymind `examples/gbdt_tabular_int8/gbdt_tabular_int8.cpp`.
//!
//! A five-tree gradient-boosted ensemble over two features and three classes,
//! written with float thresholds as a trainer would export it, is quantized onto
//! an int8 feature grid. The program sweeps a 48x48 grid of inputs through both
//! the float trees and the int8 [`QGbdt`], reports how often they agree, and
//! writes the int8 decision regions.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use tinymind::int8::calibration::{affine_params_asymmetric, quantize};
use tinymind::tree::{QGbdt, QTreeNode};

/// Features.
const NF: usize = 2;
/// Classes.
const NC: usize = 3;
/// Grid points per axis.
const G: usize = 48;

/// A node with a float threshold, as a trainer exports it: a split goes left
/// if `x[feature] <= threshold`; `feature < 0` is a leaf.
struct FNode {
    feature: i16,
    threshold: f32,
    left: i16,
    right: i16,
    leaf: i32,
}

const fn s(feature: i16, threshold: f32, left: i16, right: i16) -> FNode {
    FNode {
        feature,
        threshold,
        left,
        right,
        leaf: 0,
    }
}

const fn l(leaf: i32) -> FNode {
    FNode {
        feature: -1,
        threshold: 0.0,
        left: -1,
        right: -1,
        leaf,
    }
}

/// Five trees over two features: vertical bands by f0 (class 0 left, class 1
/// middle, class 2 right), nudged by f1.
///   t0 (class 0): f0 <= -0.15 ? +12 : -4
///   t1 (class 1): f0 <= -0.15 ? -8 : (f0 <= 0.25 ? +12 : -8)
///   t2 (class 2): f0 <=  0.25 ? -4 : +12
///   t3 (class 0): f1 <= -0.10 ? +5 : -3
///   t4 (class 2): f1 <=  0.10 ? -3 : +5
const POOL: usize = 17;
#[rustfmt::skip]
static FLOAT_POOL: [FNode; POOL] = [
    s(0, -0.15, 1, 2), l(12), l(-4),                          // t0 (root 0)
    s(0, -0.15, 4, 5), l(-8), s(0, 0.25, 6, 7), l(12), l(-8), // t1 (root 3)
    s(0, 0.25, 9, 10), l(-4), l(12),                          // t2 (root 8)
    s(1, -0.10, 12, 13), l(5), l(-3),                         // t3 (root 11)
    s(1, 0.10, 15, 16), l(-3), l(5),                          // t4 (root 14)
];
const NT: usize = 5;
static ROOTS: [i16; NT] = [0, 3, 8, 11, 14];
static CLASSES: [i16; NT] = [0, 1, 2, 0, 2];

fn float_walk(root: i16, x: &[f32; NF]) -> i32 {
    let mut n = root as usize;
    while FLOAT_POOL[n].feature >= 0 {
        let node = &FLOAT_POOL[n];
        n = if x[node.feature as usize] <= node.threshold {
            node.left
        } else {
            node.right
        } as usize;
    }
    FLOAT_POOL[n].leaf
}

fn float_predict(x: &[f32; NF]) -> usize {
    let mut logits = [0i32; NC];
    for (&root, &class) in ROOTS.iter().zip(&CLASSES) {
        logits[class as usize] += float_walk(root, x);
    }
    let mut best = 0;
    for c in 1..NC {
        if logits[c] > logits[best] {
            best = c;
        }
    }
    best
}

fn grid(i: usize) -> f32 {
    -1.0 + 2.0 * i as f32 / (G - 1) as f32
}

/// Run the example: the CSV goes to `out_dir`, the report (or with `golden`, the
/// compact golden report) to `out`. Returns whether the agreement check passed.
pub fn run(out_dir: &Path, golden: bool, out: &mut dyn Write) -> io::Result<bool> {
    // Shared feature grid over [-1, 1].
    let (scale, zp) = affine_params_asymmetric(-1.0, 1.0, -128, 127);
    let q = |v: f32| quantize(v, scale, zp, -128, 127);

    // Quantize the thresholds onto the feature grid.
    let qnodes: [QTreeNode<i8>; POOL] = core::array::from_fn(|i| {
        let f = &FLOAT_POOL[i];
        QTreeNode {
            feature: f.feature,
            threshold: if f.feature < 0 { 0 } else { q(f.threshold) },
            left: f.left,
            right: f.right,
            leaf_value: f.leaf,
        }
    });
    let gbdt = QGbdt::<'_, i8, NT, NC, POOL, NF> {
        nodes: &qnodes,
        tree_root: &ROOTS,
        tree_class: &CLASSES,
        base_score: None,
    };

    // Sweep the grid in float and int8; record the int8 regions.
    let mut region = [0usize; G * G];
    let (mut agree, mut total) = (0usize, 0usize);
    let mut hist = [0usize; NC];
    for iy in 0..G {
        for ix in 0..G {
            let (f0, f1) = (grid(ix), grid(iy));
            let c_q = gbdt.predict(&[q(f0), q(f1)]);
            let c_f = float_predict(&[f0, f1]);
            region[iy * G + ix] = c_q;
            hist[c_q] += 1;
            agree += usize::from(c_q == c_f);
            total += 1;
        }
    }

    // CSV: f0, f1, int8 predicted class.
    let mut csv = BufWriter::new(File::create(out_dir.join("gbdt_tabular_int8.csv"))?);
    writeln!(csv, "f0,f1,class")?;
    for iy in 0..G {
        for ix in 0..G {
            let (f0, f1) = (grid(ix), grid(iy));
            writeln!(csv, "{:.5},{:.5},{}", f0, f1, region[iy * G + ix])?;
        }
    }
    csv.flush()?;

    // Fixed probe points.
    let probes: [[f32; NF]; 8] = [
        [-0.8, -0.8],
        [-0.8, 0.8],
        [0.0, -0.8],
        [0.0, 0.8],
        [0.8, -0.8],
        [0.8, 0.8],
        [-0.3, 0.0],
        [0.5, 0.0],
    ];
    let probe = |p: &[f32; NF]| gbdt.predict(&[q(p[0]), q(p[1])]);

    if golden {
        writeln!(out, "# gbdt_tabular_int8 golden output")?;
        writeln!(
            out,
            "# trees={NT} classes={NC} grid={G} agree={agree}/{total}"
        )?;
        write!(out, "probes:")?;
        for p in &probes {
            write!(out, " {}", probe(p))?;
        }
        writeln!(out)?;
        return Ok(true);
    }

    writeln!(
        out,
        "Int8 GBDT ensemble (tabular, {NT} trees, {NC} classes)"
    )?;
    writeln!(
        out,
        "  feature grid: scale={:.5} zp={zp}   pool={POOL} nodes",
        f64::from(scale)
    )?;
    let percent = 100.0 * agree as f64 / total as f64;
    writeln!(
        out,
        "  int8 vs float GBDT agreement over {G}x{G} grid: {agree} / {total} ({percent:.2}%)"
    )?;
    writeln!(
        out,
        "  predicted-class histogram: c0={} c1={} c2={}",
        hist[0], hist[1], hist[2]
    )?;
    write!(out, "  probe predictions:")?;
    for p in &probes {
        write!(
            out,
            " ({:+.1},{:+.1})->{}",
            f64::from(p[0]),
            f64::from(p[1]),
            probe(p)
        )?;
    }
    writeln!(out)?;

    // The int8 trees disagree with the float trees only where a feature lands
    // within one grid step of a threshold.
    let frac = agree as f64 / total as f64;
    if frac < 0.97 {
        writeln!(
            out,
            "FAIL: int8/float agreement {:.2}% below 97%",
            100.0 * frac
        )?;
        return Ok(false);
    }
    writeln!(
        out,
        "PASS (int8 reproduces the float GBDT's regions; flips only at boundary cells)"
    )?;
    Ok(true)
}
