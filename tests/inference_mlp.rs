// InferencePerceptron against tinymind's inference-only network
// (MultilayerPerceptron with IsTrainable = false). tests/golden/inference_mlp/stdout.txt
// is the output of scripts/cpp/inference_mlp.cpp, which loads the same
// deterministic weights through tinymind's setters and runs the same inputs;
// scripts/compare-with-cpp.sh rebuilds it and checks the golden is current.
// The C++ program also checks its trainable network against the untrainable
// one; here the trainable MultilayerPerceptron, loaded with set_weights, is
// checked too.

use std::fmt::Write;

use tinymind::{
    Activation, InferencePerceptron, MultilayerPerceptron, Q8_8, Q16_16, Relu, Scalar, Sigmoid,
    Tanh,
};

struct Lcg(u32);

impl Lcg {
    /// A raw value within +-2.0 for a format whose 1.0 is `one` raw units.
    fn next_raw(&mut self, one: i32) -> i32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let span = (4 * one + 1) as u32;
        ((self.0 >> 8) % span) as i32 - 2 * one
    }
}

fn run<V, H, O, const IN: usize, const HID: usize, const OUT: usize>(
    name: &str,
    one: i32,
    seed: u32,
    from_raw: impl Fn(i32) -> V,
    to_raw: impl Fn(V) -> i32,
    out: &mut String,
) where
    V: Scalar + PartialEq + core::fmt::Debug,
    H: Activation<V>,
    O: Activation<V>,
{
    let mut rng = Lcg(seed);
    let mut hash: u32 = 0x811C_9DC5;
    let mut agree = true;
    for set in 0..8 {
        // Drawn in the C++ program's order.
        let input_weights: [[V; HID]; IN] =
            core::array::from_fn(|_| core::array::from_fn(|_| from_raw(rng.next_raw(one))));
        let input_bias_weights: [V; HID] = core::array::from_fn(|_| from_raw(rng.next_raw(one)));
        let hidden_weights: [[V; OUT]; HID] =
            core::array::from_fn(|_| core::array::from_fn(|_| from_raw(rng.next_raw(one))));
        let hidden_bias_weights: [V; OUT] = core::array::from_fn(|_| from_raw(rng.next_raw(one)));
        let inference = InferencePerceptron::<V, H, O, IN, HID, OUT>::new(
            input_weights,
            input_bias_weights,
            hidden_weights,
            hidden_bias_weights,
        );
        let mut trainable = MultilayerPerceptron::<V, H, O, IN, HID, OUT>::new(|| V::ZERO);
        trainable.set_weights(&inference);

        for sample in 0..256 {
            let input: [V; IN] = core::array::from_fn(|_| from_raw(rng.next_raw(one)));
            let y = inference.feed_forward(&input);
            trainable.feed_forward(&input);
            agree &= y == trainable.learned_values();
            for (o, &v) in y.iter().enumerate() {
                let raw = to_raw(v);
                for byte in raw.to_le_bytes() {
                    hash ^= u32::from(byte);
                    hash = hash.wrapping_mul(0x0100_0193);
                }
                if set == 0 && sample < 2 {
                    writeln!(out, "{name} set=0 sample={sample} out{o}={raw}").unwrap();
                }
            }
        }
    }
    let agrees = if agree { "yes" } else { "no" };
    writeln!(out, "{name} hash=0x{hash:08X} trainable_agrees={agrees}").unwrap();
}

#[test]
fn inference_mlp_matches_cpp() {
    let q16 = |r: i32| Q16_16::from_raw(r);
    let q16_raw = |v: Q16_16| v.raw();
    let q8 = |r: i32| Q8_8::from_raw(r as i16);
    let q8_raw = |v: Q8_8| i32::from(v.raw());

    let mut out = String::new();
    run::<Q16_16, Tanh, Sigmoid, 2, 4, 1>(
        "q16.16 2-4-1 tanh/sigmoid",
        1 << 16,
        1,
        q16,
        q16_raw,
        &mut out,
    );
    run::<Q8_8, Tanh, Sigmoid, 2, 4, 1>("q8.8 2-4-1 tanh/sigmoid", 1 << 8, 2, q8, q8_raw, &mut out);
    run::<Q16_16, Relu, Sigmoid, 4, 8, 3>(
        "q16.16 4-8-3 relu/sigmoid",
        1 << 16,
        3,
        q16,
        q16_raw,
        &mut out,
    );
    run::<Q8_8, Relu, Sigmoid, 4, 8, 3>("q8.8 4-8-3 relu/sigmoid", 1 << 8, 4, q8, q8_raw, &mut out);

    let golden = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/golden/inference_mlp/stdout.txt"
    ))
    .unwrap();
    assert_eq!(out, golden);
}
