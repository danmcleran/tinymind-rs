// ScheduledSampling against tinymind's. tests/golden/sampling/stdout.txt is the
// output of scripts/cpp/scheduled_sampling.cpp, which steps the same samplers
// and trains the same LSTM with the same glibc rand() draws;
// scripts/compare-with-cpp.sh rebuilds it and checks the golden is current. See
// that program for what is covered.

use std::fmt::Write;

use tinymind::Q16_16;
use tinymind::rand::GlibcRand;
use tinymind::recurrent::LstmNetwork;
use tinymind::training::ScheduledSampling;
use tinymind::{Sigmoid, Tanh};

fn sweep(out: &mut String, seed: u32, total: usize) {
    let mut rng = GlibcRand::new(seed);
    let mut sampler = ScheduledSampling::new(total);
    let run = total + 4;
    let quarter = run / 4 + 1;
    let mut truths = 0;
    write!(out, "total={total} seed={seed}").unwrap();
    for s in 0..run {
        if s % quarter == 0 {
            write!(
                out,
                " | step={} ratio={:.6}",
                sampler.current_step(),
                sampler.teacher_forcing_ratio()
            )
            .unwrap();
        }
        truths += sampler.select_input(1, 0, || rng.next_i32() as u32);
        if s % quarter == quarter - 1 {
            write!(out, " truths={truths}").unwrap();
            truths = 0;
        }
        sampler.step();
    }
    writeln!(
        out,
        " truths={truths} end={} next_rand={}",
        sampler.current_step(),
        rng.next_i32()
    )
    .unwrap();
    sampler.reset();
    writeln!(
        out,
        "total={total} reset step={} ratio={:.6}",
        sampler.current_step(),
        sampler.teacher_forcing_ratio()
    )
    .unwrap();
}

const HID: usize = 8;
type Network = LstmNetwork<Q16_16, Tanh, Sigmoid, 1, HID, 1>;

/// One period of the sequence, raw Q16.16 values in [0.1, 0.9].
const SEQUENCE: [i32; 12] = [
    6554, 15073, 26214, 39322, 50463, 57016, 58982, 52429, 41943, 30147, 19661, 11141,
];

fn absorb(hash: &mut u32, v: Q16_16) {
    for b in v.raw().to_le_bytes() {
        *hash ^= u32::from(b);
        *hash = hash.wrapping_mul(0x0100_0193);
    }
}

fn absorb_weights(nn: &Network, hash: &mut u32) {
    for gates in &nn.input_weights()[0] {
        for &w in gates {
            absorb(hash, w);
        }
    }
    for gates in &nn.input_bias_weights() {
        for &w in gates {
            absorb(hash, w);
        }
    }
    for row in &nn.recurrent_weights() {
        for gates in row {
            for &w in gates {
                absorb(hash, w);
            }
        }
    }
    for w in &nn.hidden_weights() {
        absorb(hash, w[0]);
    }
    absorb(hash, nn.hidden_bias_weights()[0]);
}

fn train(out: &mut String, roll_seed: u32, decay_steps: usize, epochs: usize) {
    let mut init = GlibcRand::new(1);
    let quarter = 1 << 14;
    let mut nn = Network::new(|| Q16_16::from_raw(init.next_i32() % (2 * quarter + 1) - quarter));
    let mut rng = GlibcRand::new(roll_seed);
    let mut sampler = ScheduledSampling::new(decay_steps);
    let mut hash = 0x811C_9DC5u32;
    let (mut step, mut truths, mut trained) = (0, 0, 0);
    for _ in 0..epochs {
        nn.reset_state();
        let mut prediction = Q16_16::ZERO;
        for t in 0..24 {
            let truth = Q16_16::from_raw(SEQUENCE[t % 12]);
            let target = [Q16_16::from_raw(SEQUENCE[(t + 1) % 12])];
            let x = if t == 0 {
                truth
            } else {
                let x = sampler.select_input(truth, prediction, || rng.next_i32() as u32);
                truths += i32::from(x == truth);
                x
            };
            nn.feed_forward(&[x]);
            let error = nn.calculate_error(&target);
            if !error.is_within_zero_tolerance() {
                nn.train(&target);
                trained += 1;
            }
            prediction = nn.learned_values()[0];
            sampler.step();
            step += 1;
            absorb(&mut hash, prediction);
            absorb_weights(&nn, &mut hash);
            if step % 600 == 0 {
                writeln!(
                    out,
                    "lstm decay={decay_steps} step={step} truths={truths} trained={trained} error={} hash=0x{hash:08X}",
                    error.raw()
                )
                .unwrap();
            }
        }
    }
}

#[test]
fn scheduled_sampling_matches_cpp_golden() {
    let mut out = String::new();
    sweep(&mut out, 7, 0);
    sweep(&mut out, 7, 1);
    sweep(&mut out, 3, 7);
    sweep(&mut out, 11, 100);
    sweep(&mut out, 5, 999);
    sweep(&mut out, 5, 1000);
    sweep(&mut out, 5, 1001);
    sweep(&mut out, 13, 4321);
    train(&mut out, 7, 3000, 250);
    let golden = include_str!("golden/sampling/stdout.txt");
    for (i, (a, g)) in out.lines().zip(golden.lines()).enumerate() {
        assert_eq!(a, g, "line {}", i + 1);
    }
    assert_eq!(out.lines().count(), golden.lines().count(), "line count");
}
