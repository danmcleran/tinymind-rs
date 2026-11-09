// InferenceNetwork against tinymind (C++)'s inference-only NeuralNetwork
// (IsTrainable = false) with one to three hidden layers.
// tests/golden/inference_deep/stdout.txt is the output of
// scripts/cpp/inference_deep.cpp, which loads the same deterministic weights
// through tinymind (C++)'s setters and runs the same inputs;
// scripts/compare-with-cpp.sh rebuilds it and checks the golden is current. The
// C++ program also checks its trainable network against the untrainable one;
// here the trainable NeuralNetwork, loaded with set_weights, is checked too,
// and to_inference must give back the same weights.

use std::fmt::Write;

use tinymind::{
    InferenceDense, InferenceNetwork, Layers, Linear, NeuralNetwork, Q8_8, Q16_16, Relu, Scalar,
    Sigmoid, Tanh,
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

/// One layer's weights in the C++ program's order: by source, then the bias.
fn layer<V: Copy, A, const IN: usize, const OUT: usize>(
    rng: &mut Lcg,
    one: i32,
    from_raw: &impl Fn(i32) -> V,
) -> InferenceDense<V, A, IN, OUT> {
    let mut weights = [[from_raw(0); OUT]; IN];
    for w in weights.iter_mut().flatten() {
        *w = from_raw(rng.next_raw(one));
    }
    let mut bias = [from_raw(0); OUT];
    for w in &mut bias {
        *w = from_raw(rng.next_raw(one));
    }
    InferenceDense::new(weights, bias)
}

fn absorb(hash: &mut u32, raw: i32) {
    for b in (raw as u32).to_le_bytes() {
        *hash = (*hash ^ u32::from(b)).wrapping_mul(0x0100_0193);
    }
}

#[allow(clippy::too_many_arguments)]
fn run<V, T, const IN: usize, const OUT: usize>(
    name: &str,
    one: i32,
    seed: u32,
    from_raw: impl Fn(i32) -> V,
    to_raw: impl Fn(V) -> i32,
    build: impl Fn(&mut Lcg) -> InferenceNetwork<T::Inference>,
    out: &mut String,
) where
    V: Scalar + PartialEq + core::fmt::Debug,
    T: Layers<Value = V, Inputs = [V; IN], Outputs = [V; OUT]> + Default,
{
    let mut rng = Lcg(seed);
    let mut hash = 0x811C_9DC5u32;
    let mut agree = true;
    let mut trainable: NeuralNetwork<T> = NeuralNetwork::new(|| V::ZERO);
    for _ in 0..8 {
        let net = build(&mut rng);
        trainable.set_weights(&net);
        for _ in 0..16 {
            let mut x = [V::ZERO; IN];
            for v in &mut x {
                *v = from_raw(rng.next_raw(one));
            }
            let y = net.feed_forward(&x);
            trainable.feed_forward(&x);
            let z = trainable.learned_values();
            for (&a, &b) in y.iter().zip(&z) {
                absorb(&mut hash, to_raw(a));
                agree &= a == b;
            }
            // The round trip keeps every output.
            assert_eq!(trainable.to_inference().feed_forward(&x), y);
        }
    }
    writeln!(
        out,
        "{name} hash=0x{hash:08X} trainable_agrees={}",
        u8::from(agree)
    )
    .unwrap();
}

fn q16(r: i32) -> Q16_16 {
    Q16_16::from_raw(r)
}
fn q8(r: i32) -> Q8_8 {
    Q8_8::from_raw(r as i16)
}

#[test]
fn inference_network_matches_cpp() {
    let mut out = String::new();
    let (o16, o8) = (1 << 16, 1 << 8);
    let r16 = |v: Q16_16| v.raw();
    let r8 = |v: Q8_8| i32::from(v.raw());

    run::<
        _,
        (
            tinymind::Dense<Q16_16, Tanh, 2, 4>,
            tinymind::Dense<Q16_16, Tanh, 4, 4>,
            tinymind::Dense<Q16_16, Sigmoid, 4, 1>,
        ),
        2,
        1,
    >(
        "q16.16 2-4-4-1 tanh/sigmoid",
        o16,
        1,
        q16,
        r16,
        |g| {
            InferenceNetwork::new((
                layer::<_, Tanh, 2, 4>(g, o16, &q16),
                layer::<_, Tanh, 4, 4>(g, o16, &q16),
                layer::<_, Sigmoid, 4, 1>(g, o16, &q16),
            ))
        },
        &mut out,
    );
    run::<
        _,
        (
            tinymind::Dense<Q8_8, Tanh, 2, 4>,
            tinymind::Dense<Q8_8, Tanh, 4, 4>,
            tinymind::Dense<Q8_8, Sigmoid, 4, 1>,
        ),
        2,
        1,
    >(
        "q8.8 2-4-4-1 tanh/sigmoid",
        o8,
        2,
        q8,
        r8,
        |g| {
            InferenceNetwork::new((
                layer::<_, Tanh, 2, 4>(g, o8, &q8),
                layer::<_, Tanh, 4, 4>(g, o8, &q8),
                layer::<_, Sigmoid, 4, 1>(g, o8, &q8),
            ))
        },
        &mut out,
    );
    run::<
        _,
        (
            tinymind::Dense<Q16_16, Relu, 4, 8>,
            tinymind::Dense<Q16_16, Relu, 8, 6>,
            tinymind::Dense<Q16_16, Relu, 6, 5>,
            tinymind::Dense<Q16_16, Sigmoid, 5, 3>,
        ),
        4,
        3,
    >(
        "q16.16 4-8-6-5-3 relu/sigmoid",
        o16,
        3,
        q16,
        r16,
        |g| {
            InferenceNetwork::new((
                layer::<_, Relu, 4, 8>(g, o16, &q16),
                layer::<_, Relu, 8, 6>(g, o16, &q16),
                layer::<_, Relu, 6, 5>(g, o16, &q16),
                layer::<_, Sigmoid, 5, 3>(g, o16, &q16),
            ))
        },
        &mut out,
    );
    run::<
        _,
        (
            tinymind::Dense<Q8_8, Relu, 4, 8>,
            tinymind::Dense<Q8_8, Relu, 8, 6>,
            tinymind::Dense<Q8_8, Sigmoid, 6, 3>,
        ),
        4,
        3,
    >(
        "q8.8 4-8-6-3 relu/sigmoid",
        o8,
        4,
        q8,
        r8,
        |g| {
            InferenceNetwork::new((
                layer::<_, Relu, 4, 8>(g, o8, &q8),
                layer::<_, Relu, 8, 6>(g, o8, &q8),
                layer::<_, Sigmoid, 6, 3>(g, o8, &q8),
            ))
        },
        &mut out,
    );
    run::<
        _,
        (
            tinymind::Dense<Q16_16, Relu, 3, 6>,
            tinymind::Dense<Q16_16, Linear, 6, 2>,
        ),
        3,
        2,
    >(
        "q16.16 3-6-2 relu/linear",
        o16,
        5,
        q16,
        r16,
        |g| {
            InferenceNetwork::new((
                layer::<_, Relu, 3, 6>(g, o16, &q16),
                layer::<_, Linear, 6, 2>(g, o16, &q16),
            ))
        },
        &mut out,
    );

    assert_eq!(out, include_str!("golden/inference_deep/stdout.txt"));
}
