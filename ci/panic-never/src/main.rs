//! Calls every public function of `tinymind` with inputs the optimiser cannot
//! see (`black_box`), so no check can be folded away by knowing the values.
//! Linking succeeds only if no panic is reachable from any of it.

#![no_std]
#![no_main]

use core::hint::black_box;

use core::mem::size_of;
use tinymind::int8::{self, QDense, QDensePerChannel, QSoftmax, Requantizer};
use tinymind::rand::GlibcRand;

use tinymind::{
    Activation, Dense, Fixed, InferencePerceptron, Linear, MultilayerPerceptron, NeuralNetwork,
    Q8_8, Q16_16, Relu, Sigmoid, Tanh,
};

/// Every `QValue` operation on run-time operands.
fn arithmetic() {
    let (a, b) = (
        Q16_16::from_raw(black_box(1)),
        Q16_16::from_raw(black_box(2)),
    );
    let n = black_box(3);
    black_box((a + b, a - b, a * b, a / b, a.div_int(n), a < b, a == b));
    black_box((
        Q16_16::from_int(n),
        Q16_16::pow2_neg(black_box(40)),
        a.raw(),
    ));
    black_box((
        a.is_within_zero_tolerance(),
        a.to_f64(),
        Q16_16::from_f64_trunc(black_box(0.5)),
    ));
    let mut c = a;
    c += b;
    black_box((
        c,
        <Q16_16 as Fixed>::pow2_neg(black_box(2)),
        Fixed::div_int(a, n),
    ));

    let (a, b) = (Q8_8::from_raw(black_box(1)), Q8_8::from_raw(black_box(2)));
    black_box((a + b, a - b, a * b, a / b, a.div_int(n), a < b, a == b));
    black_box((Q8_8::from_int(n), Q8_8::pow2_neg(black_box(40)), a.raw()));
    black_box((
        a.is_within_zero_tolerance(),
        a.to_f64(),
        Q8_8::from_f64_trunc(black_box(0.5)),
    ));
}

fn activations() {
    let x = Q16_16::from_raw(black_box(5));
    black_box((Relu::activate(x), Relu::derivative(x)));
    black_box((Tanh::activate(x), Tanh::derivative(x)));
    black_box((Sigmoid::activate(x), Sigmoid::derivative(x)));
    let y = Q8_8::from_raw(black_box(5));
    black_box((
        <Relu as Activation<Q8_8>>::activate(y),
        <Relu as Activation<Q8_8>>::derivative(y),
    ));
    black_box((
        <Tanh as Activation<Q8_8>>::activate(y),
        <Tanh as Activation<Q8_8>>::derivative(y),
    ));
    black_box((
        <Sigmoid as Activation<Q8_8>>::activate(y),
        <Sigmoid as Activation<Q8_8>>::derivative(y),
    ));
}

/// Build, train and read a network, the way the examples do.
fn network<V, H, O, const IN: usize, const HID: usize, const OUT: usize>(raw: fn(i32) -> V)
where
    V: Fixed,
    H: Activation<V>,
    O: Activation<V>,
{
    let mut rng = GlibcRand::new(black_box(1));
    let mut nn = MultilayerPerceptron::<V, H, O, IN, HID, OUT>::new(|| raw(rng.next_i32()));
    nn.set_learning_rate(raw(black_box(8192)));
    nn.set_momentum_rate(raw(black_box(4096)));
    nn.set_acceleration_rate(raw(black_box(2048)));
    nn.set_gradient_clip(black_box(Some((raw(-4096), raw(4096)))));
    for _ in 0..black_box(10) {
        let input = [raw(black_box(1)); IN];
        let target = [raw(black_box(2)); OUT];
        nn.feed_forward(&input);
        let error = nn.calculate_error(&target);
        if black_box(error) != V::ZERO {
            nn.train(&target);
        }
    }
    black_box(nn.learned_values());
    black_box((nn.input_weights(), nn.input_bias_weights()));
    black_box((nn.hidden_weights(), nn.hidden_bias_weights()));

    // The inference-only network, and loading weights back.
    let frozen = black_box(nn.to_inference());
    black_box(frozen.feed_forward(&black_box([raw(1); IN])));
    nn.set_weights(&frozen);
    nn.set_input_weights(&black_box([[raw(3); HID]; IN]));
    nn.set_input_bias_weights(&black_box([raw(3); HID]));
    nn.set_hidden_weights(&black_box([[raw(3); OUT]; HID]));
    nn.set_hidden_bias_weights(&black_box([raw(3); OUT]));
    black_box(nn.to_inference());
}

/// A network with three hidden layers, trained and read the same way.
fn deep_network<V: Fixed, H: Activation<V>, O: Activation<V>>(raw: fn(i32) -> V) {
    let mut rng = GlibcRand::new(black_box(1));
    let mut nn = NeuralNetwork::<(
        Dense<V, H, 4, 8>,
        Dense<V, H, 8, 6>,
        Dense<V, H, 6, 5>,
        Dense<V, O, 5, 3>,
    )>::new(|| raw(rng.next_i32()));
    nn.set_learning_rate(raw(black_box(8192)));
    nn.set_momentum_rate(raw(black_box(4096)));
    nn.set_acceleration_rate(raw(black_box(2048)));
    nn.set_gradient_clip(black_box(Some((raw(-4096), raw(4096)))));
    for _ in 0..black_box(10) {
        nn.feed_forward(&black_box([raw(1); 4]));
        let target = black_box([raw(2); 3]);
        if black_box(nn.calculate_error(&target)) != V::ZERO {
            nn.train(&target);
        }
    }
    black_box(nn.learned_values());
    let layers = nn.layers_mut();
    black_box((
        layers.0.weights(),
        layers.0.bias_weights(),
        layers.3.outputs(),
    ));
    layers.1.set_weights(&black_box([[raw(3); 6]; 8]));
    layers.2.set_bias_weights(&black_box([raw(3); 5]));
    black_box(layers.1.weights());
}

// An inference network holds one value per connection and nothing else, on the
// target as on the host (the sizes in docs/xor.md).
const _: () = assert!(size_of::<InferencePerceptron<Q16_16, Tanh, Sigmoid, 2, 4, 1>>() == 68);
const _: () = assert!(size_of::<InferencePerceptron<Q8_8, Tanh, Sigmoid, 2, 3, 1>>() == 26);
const _: () = assert!(size_of::<MultilayerPerceptron<Q16_16, Tanh, Sigmoid, 2, 4, 1>>() == 384);
const _: () = assert!(size_of::<MultilayerPerceptron<Q8_8, Tanh, Sigmoid, 2, 3, 1>>() == 156);

/// A model written by the PyTorch importer (generated code must be panic-free too).
#[path = "../../../examples/common/imported_mlp_model.rs"]
mod imported;

/// The int8 path: primitives, requantizer, a dense layer, activations, tables.
fn int8_inference() {
    let (x, m, s) = (black_box(5), black_box(1 << 30), black_box(3));
    black_box(int8::saturating_rounding_doubling_high_mul(x, m));
    black_box(int8::rounding_divide_by_pot(x, s));
    black_box(int8::multiply_by_quantized_multiplier(x, m, s));
    let rq = Requantizer {
        multiplier: black_box(m),
        shift: black_box(s),
        zero_point: black_box(-1),
        qmin: black_box(-128),
        qmax: black_box(127),
    };
    black_box(rq.apply(black_box(1000)));
    let weights: [[i8; 4]; 3] = black_box([[1; 4]; 3]);
    let biases: [i32; 3] = black_box([0; 3]);
    let layer = QDense {
        weights: &weights,
        biases: if black_box(true) { Some(&biases) } else { None },
        input_zero_point: black_box(-128),
        requantizer: rq,
    };
    let out = layer.forward(&black_box([1i8; 4]));
    let rqs: [Requantizer; 3] = black_box([rq; 3]);
    let per_channel = QDensePerChannel {
        weights: &weights,
        biases: if black_box(true) { Some(&biases) } else { None },
        input_zero_point: black_box(-128),
        requantizers: &rqs,
    };
    black_box(per_channel.forward(&black_box([1i8; 4])));
    let lut: [i8; int8::LUT_SIZE] = black_box([0; int8::LUT_SIZE]);
    for y in out {
        black_box((int8::qrelu(y, -128), int8::qrelu6(y, -128, 100)));
        black_box((int8::lut_index(y), int8::apply_lut(y, &lut)));
    }
    let softmax = QSoftmax {
        input_multiplier: black_box(1 << 30),
        input_left_shift: black_box(23),
        diff_min: black_box(-248),
    };
    black_box(softmax.forward(&black_box([0i8; 1])));
    black_box(softmax.forward(&black_box([0i8; 10])));
}

#[unsafe(no_mangle)]
pub extern "C" fn _start() -> ! {
    arithmetic();
    activations();
    int8_inference();
    black_box(imported::forward(&black_box([0i8; imported::NUM_INPUTS])));
    // The examples' configurations: xor 2-4-1 (both formats) and iris 4-8-3.
    network::<Q16_16, Tanh, Sigmoid, 2, 4, 1>(Q16_16::from_raw);
    network::<Q8_8, Tanh, Sigmoid, 2, 4, 1>(|r| Q8_8::from_raw(r as i16));
    network::<Q16_16, Relu, Sigmoid, 4, 8, 3>(Q16_16::from_raw);
    // Multiple hidden layers, both formats.
    deep_network::<Q16_16, Relu, Sigmoid>(Q16_16::from_raw);
    // A regression head (linear output), as in the energy_efficiency example.
    network::<Q16_16, Relu, Linear, 8, 16, 2>(Q16_16::from_raw);
    deep_network::<Q16_16, Relu, Linear>(Q16_16::from_raw);
    deep_network::<Q8_8, Tanh, Sigmoid>(|r| Q8_8::from_raw(r as i16));
    #[allow(clippy::empty_loop)]
    loop {}
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    unsafe extern "Rust" {
        // Deliberately undefined: a reachable panic makes the link fail here.
        fn tinymind_panic_is_reachable() -> !;
    }
    // SAFETY: never called in a successful build; that is the check.
    unsafe { tinymind_panic_is_reachable() }
}
