//! The fine-tuning heads on the board: a `DenseHead<Q16_16, 32, 6>` training
//! step and prediction, a `PrototypeHead` update and prediction, and a whole
//! adaptation (30 epochs over 30 labelled samples), on the workload of
//! tests/sweeps/head_bench.rs. Each operation's hash over the 30 samples must
//! equal the one tinymind (C++) printed for the same workload
//! (tests/golden/head/stdout.txt, read by build.rs).

use core::fmt::Write;
use core::hint::black_box;

use crate::board::{self, Uart};
use crate::{REPS, median};

#[allow(dead_code)]
#[path = "../../tests/sweeps/head_bench.rs"]
mod head_bench;
use head_bench::{Data, N, NAMES, adapt, data, hashes, learned};

mod expect {
    include!(concat!(env!("OUT_DIR"), "/head_expect.rs"));
}

/// The embeddings and the shipped head, built at compile time (flash).
static DATA: Data = data();

fn report(out: &mut Uart, name: &str, t: [u32; REPS]) {
    let (min, med, max) = median(t);
    let _ = writeln!(out, "{name}_min={min}\n{name}_med={med}\n{name}_max={max}");
}

pub fn run(out: &mut Uart) {
    let _ = writeln!(out, "name=head\nhead_in=32\nhead_out=6\nsamples={N}");
    let d = &DATA;

    let (got, correct) = hashes(d);
    let mut all = true;
    for ((name, hash), expect) in NAMES.iter().zip(got).zip(expect::EXPECT) {
        all &= hash == expect;
        let _ = writeln!(
            out,
            "{name}_hash=0x{hash:08X}\n{name}_expect=0x{expect:08X}"
        );
    }
    let _ = writeln!(
        out,
        "adapted_correct={correct}\nhash_match={}",
        u32::from(all)
    );

    // One SGD step on a fresh copy of the shipped head.
    let mut t = [0u32; REPS];
    for (r, slot) in t.iter_mut().enumerate() {
        let s = r % N;
        let mut head = black_box(d.shipped);
        let (x, label) = black_box((&d.x[s], d.label[s]));
        let a = board::cycles();
        head.train_class(x, label, head_bench::RATE, None);
        *slot = board::cycles().wrapping_sub(a);
        black_box(&head);
    }
    report(out, "head_step", t);

    // The same step that also returns the gradient for the embedding.
    for (r, slot) in t.iter_mut().enumerate() {
        let s = r % N;
        let mut head = black_box(d.shipped);
        let mut deltas = [tinymind::Q16_16::from_raw(0); head_bench::IN];
        let (x, label) = black_box((&d.x[s], d.label[s]));
        let a = board::cycles();
        head.train_class(x, label, head_bench::RATE, Some(&mut deltas));
        *slot = board::cycles().wrapping_sub(a);
        black_box((&head, &deltas));
    }
    report(out, "head_step_deltas", t);

    for (r, slot) in t.iter_mut().enumerate() {
        let x = black_box(&d.x[r % N]);
        let a = board::cycles();
        let p = d.shipped.predict(x);
        *slot = board::cycles().wrapping_sub(a);
        black_box(p);
    }
    report(out, "head_predict", t);

    let prototypes = learned(d);
    for (r, slot) in t.iter_mut().enumerate() {
        let s = r % N;
        let mut p = black_box(prototypes);
        let (x, label) = black_box((&d.x[s], d.label[s]));
        let a = board::cycles();
        p.learn(x, label);
        *slot = board::cycles().wrapping_sub(a);
        black_box(&p);
    }
    report(out, "proto_learn", t);

    for (r, slot) in t.iter_mut().enumerate() {
        let x = black_box(&d.x[r % N]);
        let a = board::cycles();
        let c = prototypes.predict(x);
        *slot = board::cycles().wrapping_sub(a);
        black_box(c);
    }
    report(out, "proto_predict", t);

    // The whole adaptation: 30 epochs of 30 steps, timed once.
    let mut head = black_box(d.shipped);
    let a = board::cycles();
    adapt(&mut head, black_box(d));
    let total = board::cycles().wrapping_sub(a);
    black_box(&head);
    let _ = writeln!(
        out,
        "head_adapt_steps={}\nhead_adapt_cycles={total}",
        head_bench::EPOCHS * N
    );
}
