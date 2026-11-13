//! Calls every public function of `tinymind` with inputs the optimiser cannot
//! see (`black_box`), so no check can be folded away by knowing the values.
//! Linking succeeds only if no panic is reachable from any of it.

#![no_std]
#![no_main]

use core::hint::black_box;

use core::mem::size_of;
use tinymind::head::{
    DenseHead, ManhattanDistance, PrototypeHead, SoftmaxCrossEntropyLoss, SoftmaxExp,
    SquaredErrorLoss,
};
use tinymind::int8::{
    self, QAdd, QAvgPool2D, QConcat, QConv2D, QConv2DPerChannel, QDense, QDensePerChannel,
    QDepthwiseConv2D, QFft1D, QGlobalAvgPool2D, QGruCell, QLstmCell, QMaxPool2D, QMul, QPad2D,
    QPointwiseConv2D, QPointwiseConv2DPerChannel, QSoftmax, Requantizer,
};
use tinymind::qlearn;
use tinymind::rand::GlibcRand;
use tinymind::recurrent::LstmNetwork;
use tinymind::tree::{self, QDecisionTree, QGbdt, QTreeNode};

use tinymind::{
    Activation, Adam, AdamFloat, Dense, InferenceDense, InferenceNetwork, InferencePerceptron,
    Linear, MultilayerPerceptron, NeuralNetwork, Q8_8, Q16_16, Relu, RmsProp, RmsPropFloat, Scalar,
    Sgd, Sigmoid, Sqrt, Tanh,
};

/// Every `QValue` operation on run-time operands.
fn arithmetic() {
    let (a, b) = (
        Q16_16::from_raw(black_box(1)),
        Q16_16::from_raw(black_box(2)),
    );
    let n = black_box(3);
    black_box((a + b, a - b, a * b, a / b, a.div_int(n), a < b, a == b));
    black_box((a.mul_round_toward_zero(b), a.mul_round_nearest(b)));
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
        <Q16_16 as Scalar>::pow2_neg(black_box(2)),
        Scalar::div_int(a, n),
    ));

    let (a, b) = (Q8_8::from_raw(black_box(1)), Q8_8::from_raw(black_box(2)));
    black_box((a + b, a - b, a * b, a / b, a.div_int(n), a < b, a == b));
    black_box((a.mul_round_toward_zero(b), a.mul_round_nearest(b)));
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
    V: Scalar,
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
fn deep_network<V: Scalar, H: Activation<V>, O: Activation<V>>(raw: fn(i32) -> V) {
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
    // The inference-only form, both ways.
    let inference = black_box(nn.to_inference());
    black_box(inference.feed_forward(&black_box([raw(1); 4])));
    nn.set_weights(&inference);
    let single = InferenceNetwork::new((InferenceDense::<V, O, 2, 3>::new(
        black_box([[raw(5); 3]; 2]),
        black_box([raw(-5); 3]),
    ),));
    black_box(single.feed_forward(&black_box([raw(7); 2])));
}

/// tinymind (C++)'s floating-point Adam and RMSprop on f32 and f64 networks
/// (libm's pow and sqrt).
fn adaptive_float() {
    let mut rng = GlibcRand::new(black_box(3));
    let mut adam =
        NeuralNetwork::<(Dense<f32, Tanh, 2, 4>, Dense<f32, Sigmoid, 4, 1>), _>::with_optimizer(
            || rng.next_i32() as f32 / 2.0e9,
            black_box(AdamFloat::default()),
        );
    let mut rms =
        NeuralNetwork::<(Dense<f64, Relu, 3, 5>, Dense<f64, Linear, 5, 2>), _>::with_optimizer(
            || f64::from(rng.next_i32()) / 2.0e9,
            black_box(RmsPropFloat::default()),
        );
    adam.set_gradient_clip(black_box(Some((-1.0, 1.0))));
    for _ in 0..black_box(10) {
        adam.feed_forward(&black_box([1.0, 0.0]));
        adam.train(&black_box([1.0]));
        rms.feed_forward(&black_box([0.5, -0.5, 0.25]));
        rms.train(&black_box([0.1, -0.2]));
    }
    black_box((
        adam.learned_values(),
        adam.optimizer().timestep(),
        rms.learned_values(),
    ));
}

/// Adam and RMSprop on both network types, in both formats, with clipping on.
fn adaptive<V: Scalar + Sqrt>(raw: fn(i32) -> V)
where
    Sigmoid: Activation<V>,
{
    let adam = Adam::new(raw(black_box(230)), raw(black_box(255)), raw(black_box(1)));
    let rmsprop = RmsProp::new(raw(black_box(230)), raw(black_box(1)));
    let mut rng = GlibcRand::new(black_box(1));
    let mut mlp = MultilayerPerceptron::<V, Relu, Linear, 4, 8, 3, _>::with_optimizer(
        || raw(rng.next_i32()),
        adam,
    );
    let mut deep = NeuralNetwork::<
        (
            Dense<V, Relu, 4, 8>,
            Dense<V, Relu, 8, 6>,
            Dense<V, Sigmoid, 6, 3>,
        ),
        _,
    >::with_optimizer(|| raw(rng.next_i32()), rmsprop);
    mlp.set_gradient_clip(black_box(Some((raw(-4096), raw(4096)))));
    deep.set_gradient_clip(black_box(Some((raw(-4096), raw(4096)))));
    for _ in 0..black_box(10) {
        let x = black_box([raw(1); 4]);
        let t = black_box([raw(2); 3]);
        mlp.feed_forward(&x);
        mlp.train(&t);
        deep.feed_forward(&x);
        deep.train(&t);
    }
    black_box((
        mlp.learned_values(),
        deep.learned_values(),
        mlp.optimizer().timestep(),
    ));
    black_box(raw(black_box(5)).sqrt());
}

/// An LSTM network: sequences with resets, training with clipping, an
/// optimizer, and every getter and setter.
fn lstm<V, H, O, const IN: usize, const HID: usize, const OUT: usize, Opt>(
    raw: fn(i32) -> V,
    opt: Opt,
) where
    V: Scalar,
    H: Activation<V>,
    O: Activation<V>,
    Sigmoid: Activation<V>,
    Tanh: Activation<V>,
    Opt: tinymind::Optimizer<V>,
{
    let mut rng = GlibcRand::new(black_box(1));
    let mut nn =
        LstmNetwork::<V, H, O, IN, HID, OUT, Opt>::with_optimizer(|| raw(rng.next_i32()), opt);
    nn.set_learning_rate(raw(black_box(8192)));
    nn.set_momentum_rate(raw(black_box(4096)));
    nn.set_acceleration_rate(raw(black_box(2048)));
    nn.set_gradient_clip(black_box(Some((raw(-4096), raw(4096)))));
    for step in 0..black_box(10) {
        if black_box(step) == 4 {
            nn.reset_state();
        }
        nn.feed_forward(&black_box([raw(1); IN]));
        let target = black_box([raw(2); OUT]);
        if black_box(nn.calculate_error(&target)) != V::ZERO {
            nn.train(&target);
        }
    }
    black_box((nn.learned_values(), nn.hidden_outputs(), nn.cell_states()));
    black_box((
        nn.input_weights(),
        nn.input_bias_weights(),
        nn.recurrent_weights(),
    ));
    black_box((
        nn.hidden_weights(),
        nn.hidden_bias_weights(),
        nn.optimizer(),
    ));
    nn.set_input_weights(&black_box([[[raw(3); 4]; HID]; IN]));
    nn.set_input_bias_weights(&black_box([[raw(3); 4]; HID]));
    nn.set_recurrent_weights(&black_box([[[raw(3); 4]; HID]; HID]));
    nn.set_hidden_weights(&black_box([[raw(3); OUT]; HID]));
    nn.set_hidden_bias_weights(&black_box([raw(3); OUT]));
    nn.feed_forward(&black_box([raw(1); IN]));
    black_box(nn.learned_values());
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

/// Convolutional models written by the PyTorch importer: padded, depthwise and
/// pointwise convolutions, max, average and global pooling, flatten.
#[path = "../../../examples/common/imported_cnn_model.rs"]
mod imported_cnn;
#[path = "../../../examples/common/imported_cnn_flat_model.rs"]
mod imported_cnn_flat;
/// The voting form (one `QDecisionTree` per tree), on `i16` inputs.
#[path = "../../../examples/common/imported_trees_forest_model.rs"]
mod imported_forest;
/// A model written by the tree importer (generated code must be panic-free too).
#[path = "../../../examples/common/imported_trees_gbdt_model.rs"]
mod imported_trees;

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

/// The quantized FFT: forward, inverse and magnitudes, on tables the compiler
/// cannot see.
fn int8_fft<const N: usize, const HALF: usize>() {
    let cos = black_box([black_box(32767i16); HALF]);
    let sin = black_box([black_box(-32768i16); HALF]);
    let fft: QFft1D<'_, N, HALF> = black_box(QFft1D {
        twiddle_cos: &cos,
        twiddle_sin: &sin,
    });
    let mut real = black_box([black_box(1i16); N]);
    let mut imag = black_box([black_box(-1i16); N]);
    fft.forward(&mut real, &mut imag);
    black_box((&real, &imag));
    fft.inverse(&mut real, &mut imag);
    let mut mag = black_box([0i32; N]);
    fft.magnitude_squared(&real, &imag, &mut mag);
    black_box(&mag);
}

/// Convolutions and pooling: padded and unpadded, strided, per tensor and per
/// channel, with every parameter and tensor opaque.
fn int8_conv() {
    let rq = Requantizer {
        multiplier: black_box(1 << 30),
        shift: black_box(7),
        zero_point: black_box(-3),
        qmin: black_box(-128),
        qmax: black_box(127),
    };
    let rqs: [Requantizer; 4] = black_box([rq; 4]);
    let biases: [i32; 4] = black_box([1; 4]);
    let bias = || if black_box(true) { Some(&biases) } else { None };
    let zp = black_box(-5);
    let input: [[[i8; 2]; 7]; 6] = black_box([[[1; 2]; 7]; 6]);

    let weights: [[[[i8; 2]; 2]; 3]; 4] = black_box([[[[1; 2]; 2]; 3]; 4]);
    // SAME-style padding, stride 2 down, 1 across: 4 x 8 x 4.
    let conv: QConv2D<'_, 2, 3, 2, 4, 2, 1, 1, 2, 1, 1> = QConv2D {
        weights: &weights,
        biases: bias(),
        input_zero_point: zp,
        requantizer: rq,
    };
    let mut out = [[[0i8; 4]; 8]; 4];
    conv.forward(&input, &mut out);
    black_box(&out);
    // VALID: 4 x 6 x 4.
    let conv: QConv2DPerChannel<'_, 2, 3, 2, 4> = QConv2DPerChannel {
        weights: &weights,
        biases: bias(),
        input_zero_point: zp,
        requantizers: &rqs,
    };
    let mut out = [[[0i8; 4]; 6]; 4];
    conv.forward(&input, &mut out);
    black_box(&out);

    // One input channel: the gathered-window path, padded on every side,
    // stride 2, an odd number of filters.
    let mono: [[[i8; 1]; 7]; 6] = black_box([[[1; 1]; 7]; 6]);
    let mono_weights: [[[[i8; 1]; 3]; 3]; 3] = black_box([[[[1; 1]; 3]; 3]; 3]);
    let mono_rqs: [Requantizer; 3] = black_box([rq; 3]);
    let mono_biases: [i32; 3] = black_box([1; 3]);
    let conv: QConv2DPerChannel<'_, 1, 3, 3, 3, 2, 2, 1, 1, 1, 1> = QConv2DPerChannel {
        weights: &mono_weights,
        biases: if black_box(true) {
            Some(&mono_biases)
        } else {
            None
        },
        input_zero_point: zp,
        requantizers: &mono_rqs,
    };
    let mut out = [[[0i8; 3]; 4]; 3];
    conv.forward(&mono, &mut out);
    black_box(&out);

    let dw_weights: [[[i8; 3]; 3]; 2] = black_box([[[1; 3]; 3]; 2]);
    let dw_biases: [i32; 2] = black_box([1; 2]);
    let dw_rqs: [Requantizer; 2] = black_box([rq; 2]);
    let depthwise: QDepthwiseConv2D<'_, 2, 3, 3, 2, 2, 1, 1, 1, 1> = QDepthwiseConv2D {
        weights: &dw_weights,
        biases: if black_box(true) {
            Some(&dw_biases)
        } else {
            None
        },
        input_zero_point: zp,
        requantizers: &dw_rqs,
    };
    let mut out = [[[0i8; 2]; 4]; 3];
    depthwise.forward(&input, &mut out);
    black_box(&out);

    let pw_weights: [[i8; 2]; 4] = black_box([[1; 2]; 4]);
    let pointwise = QPointwiseConv2D {
        weights: &pw_weights,
        biases: bias(),
        input_zero_point: zp,
        requantizer: rq,
    };
    let mut out = [[[0i8; 4]; 7]; 6];
    pointwise.forward(&input, &mut out);
    black_box(&out);
    let pointwise = QPointwiseConv2DPerChannel {
        weights: &pw_weights,
        biases: bias(),
        input_zero_point: zp,
        requantizers: &rqs,
    };
    pointwise.forward(&input, &mut out);
    black_box(&out);

    // 32 channels reach the filter-pair path (PAIR_MIN_CHANNELS); 5 filters
    // and 3 pixels leave an odd last filter and an odd last pixel.
    let wide_input: [[[i8; 32]; 1]; 3] = black_box([[[1; 32]; 1]; 3]);
    let wide_weights: [[i8; 32]; 5] = black_box([[1; 32]; 5]);
    let wide_biases: [i32; 5] = black_box([1; 5]);
    let wide_rqs: [Requantizer; 5] = black_box([rq; 5]);
    let wide_bias = || {
        if black_box(true) {
            Some(&wide_biases)
        } else {
            None
        }
    };
    let mut wide_out = [[[0i8; 5]; 1]; 3];
    QPointwiseConv2D {
        weights: &wide_weights,
        biases: wide_bias(),
        input_zero_point: zp,
        requantizer: rq,
    }
    .forward(&wide_input, &mut wide_out);
    black_box(&wide_out);
    QPointwiseConv2DPerChannel {
        weights: &wide_weights,
        biases: wide_bias(),
        input_zero_point: zp,
        requantizers: &wide_rqs,
    }
    .forward(&wide_input, &mut wide_out);
    black_box(&wide_out);

    let mut out = [[[0i8; 2]; 3]; 3];
    QMaxPool2D::<2, 2>.forward(&input, &mut out);
    black_box(&out);
    let mut out = [[[0i8; 2]; 5]; 2];
    let avg = QAvgPool2D::<3, 3, 2, 1> {
        qmin: black_box(-128),
        qmax: black_box(127),
    };
    avg.forward(&input, &mut out);
    black_box(&out);
    let global = QGlobalAvgPool2D {
        qmin: black_box(-128),
        qmax: black_box(127),
    };
    black_box(global.forward(&input));

    // Elementwise addition over each tensor shape, every parameter opaque.
    let add = QAdd {
        input_a_zero_point: black_box(-3),
        input_b_zero_point: black_box(7),
        left_shift: black_box(20),
        input_a_multiplier: black_box(1 << 30),
        input_a_shift: black_box(1),
        input_b_multiplier: black_box(1 << 30),
        input_b_shift: black_box(0),
        output_requantizer: rq,
    };
    let mut sum = [[[0i8; 2]; 7]; 6];
    add.forward(&input, &black_box(input), &mut sum);
    black_box(&sum);
    let mut row = [[0i8; 2]; 7];
    add.forward(&input[0], &black_box(input[1]), &mut row);
    black_box(&row);
    let mut flat = [0i8; 2];
    add.forward(&input[0][0], &black_box(input[1][1]), &mut flat);
    black_box(&flat);
    // Multiplication, padding and concatenation, every parameter opaque.
    let mul = QMul {
        input_a_zero_point: black_box(-3),
        input_b_zero_point: black_box(7),
        requantizer: rq,
    };
    let mut product = [[[0i8; 2]; 7]; 6];
    mul.forward(&input, &black_box(input), &mut product);
    black_box(&product);
    let mut padded = [[[0i8; 2]; 10]; 8];
    QPad2D::<1, 1, 2, 1> {
        pad_value: black_box(-5),
    }
    .forward(&input, &mut padded);
    black_box(&padded);
    let concat = QConcat {
        input_a_zero_point: black_box(1),
        input_b_zero_point: black_box(-1),
        requantizer_a: rq,
        requantizer_b: black_box(rq),
    };
    let mut joined = [[[0i8; 3]; 7]; 6];
    let narrow: [[[i8; 1]; 7]; 6] = black_box([[[2; 1]; 7]; 6]);
    concat.forward(&input, &narrow, &mut joined);
    black_box(&joined);

    // 256 elements or more: the tabulated path.
    let big: [[[i8; 8]; 8]; 8] = black_box([[[3; 8]; 8]; 8]);
    let mut big_sum = [[[0i8; 8]; 8]; 8];
    add.forward(&big, &black_box(big), &mut big_sum);
    black_box(&big_sum);
}

/// The int8 recurrent cells with every field, input and state opaque.
fn int8_recurrent<C: int8::CellStorage, const IN: usize, const HID: usize>(cell_value: C) {
    let rq = black_box(Requantizer {
        multiplier: 1 << 30,
        shift: 8,
        zero_point: 0,
        qmin: -128,
        qmax: 127,
    });
    let lut = black_box([1i8; 256]);
    let w_in = black_box([[[1i8; IN]; HID]; 4]);
    let w_rec = black_box([[[1i8; HID]; HID]; 4]);
    let biases = black_box([[1i32; HID]; 4]);
    let lstm: QLstmCell<'_, IN, HID, C> = QLstmCell {
        input_weights: &w_in,
        recurrent_weights: &w_rec,
        biases: &biases,
        input_zero_point: black_box(1),
        hidden_zero_point: black_box(-1),
        cell_zero_point: black_box(cell_value),
        input_to_lut_multiplier: black_box([1 << 30; 4]),
        input_to_lut_shift: black_box([8; 4]),
        recurrent_to_lut_multiplier: black_box([1 << 30; 4]),
        recurrent_to_lut_shift: black_box([8; 4]),
        sigmoid_lut: &lut,
        tanh_lut: &lut,
        f_times_c_multiplier: black_box(1 << 30),
        f_times_c_shift: black_box(7),
        i_times_g_multiplier: black_box(1 << 30),
        i_times_g_shift: black_box(7),
        cell_qmin: black_box(cell_value),
        cell_qmax: black_box(cell_value),
        cell_to_tanh_multiplier: black_box(1 << 30),
        cell_to_tanh_shift: black_box(0),
        tanh_cell_lut: &lut,
        output_requantizer: rq,
    };
    let x = black_box([2i8; IN]);
    let mut h = black_box([3i8; HID]);
    let mut c = black_box([cell_value; HID]);
    lstm.forward(&x, &mut h, &mut c);
    black_box((&h, &c));

    let g_in = black_box([[[1i8; IN]; HID]; 3]);
    let g_rec = black_box([[[1i8; HID]; HID]; 3]);
    let g_biases = black_box([[1i32; HID]; 3]);
    let gru: QGruCell<'_, IN, HID> = QGruCell {
        input_weights: &g_in,
        recurrent_weights: &g_rec,
        biases: &g_biases,
        input_zero_point: black_box(1),
        hidden_zero_point: black_box(-1),
        input_to_lut_multiplier: black_box([1 << 30; 3]),
        input_to_lut_shift: black_box([8; 3]),
        recurrent_to_lut_multiplier: black_box([1 << 30; 3]),
        recurrent_to_lut_shift: black_box([8; 3]),
        sigmoid_lut: &lut,
        tanh_lut: &lut,
        r_times_h_multiplier: black_box(1 << 30),
        r_times_h_shift: black_box(7),
        one_minus_z_times_n_multiplier: black_box(1 << 30),
        one_minus_z_times_n_shift: black_box(7),
        z_times_h_multiplier: black_box(1 << 30),
        z_times_h_shift: black_box(7),
        output_qmin: black_box(-128),
        output_qmax: black_box(127),
    };
    gru.forward(&x, &mut h);
    black_box(&h);
}

/// Trees and ensembles with every field opaque, so any index may be out of
/// range: the walk, the ensemble and the validity checks must stay total.
fn trees<T: Copy + Into<i32>>(threshold: T, input: T) {
    let node = QTreeNode {
        feature: black_box(0),
        threshold: black_box(threshold),
        left: black_box(1),
        right: black_box(2),
        leaf_value: black_box(3),
    };
    let nodes: [QTreeNode<T>; 5] = black_box([node; 5]);
    let x: [T; 3] = black_box([input; 3]);
    black_box(tree::walk_tree(&nodes, black_box(0), &x, black_box(5)));
    let t = QDecisionTree::<'_, T, 5, 3>::new(&nodes);
    black_box((t.predict(&x), t.is_valid()));
    let roots: [i16; 4] = black_box([0; 4]);
    let classes: [i16; 4] = black_box([1; 4]);
    let base: [i32; 2] = black_box([0; 2]);
    let g = QGbdt::<'_, T, 4, 2, 5, 3> {
        nodes: &nodes,
        tree_root: &roots,
        tree_class: &classes,
        base_score: if black_box(true) { Some(&base) } else { None },
    };
    black_box((g.forward(&x), g.predict(&x), g.is_valid()));
}

/// An environment whose answers the optimiser cannot see: any number of valid
/// actions (more than `ACTIONS` included) and any random integer.
struct OpaqueEnvironment;

impl qlearn::Environment<4> for OpaqueEnvironment {
    fn valid_actions(&self, _state: usize, actions: &mut [usize; 4]) -> usize {
        *actions = black_box([0; 4]);
        black_box(4)
    }

    fn random_int(&mut self, min: usize, _max: usize) -> usize {
        black_box(min)
    }
}

fn q_learning<V: Scalar>(raw: fn(i32) -> V) {
    let mut q = qlearn::QLearner::<V, _, 6, 4>::new(
        OpaqueEnvironment,
        raw(black_box(13_107)),
        raw(black_box(52_428)),
        black_box(50),
    );
    q.start_new_episode();
    let action = q.take_action(black_box(0)).unwrap_or(black_box(9));
    q.update_from_experience(&qlearn::Experience {
        state: black_box(0),
        action,
        reward: raw(black_box(65_536)),
        new_state: black_box(3),
    });
    q.set_q_value(black_box(7), black_box(1), raw(black_box(1)));
    q.set_learning_rate(raw(black_box(1)));
    q.set_discount_factor(raw(black_box(1)));
    q.set_random_action_decision_point(black_box(101));
    q.set_state(black_box(8));
    black_box((
        q.q_value(black_box(2), black_box(5)),
        q.state(),
        q.q_values(),
    ));
}

/// The deep Q-network: a learner over a 6-8-4 tanh/linear perceptron, its
/// reads and training steps, with states and actions the compiler cannot see.
fn dqn<V: Scalar>(raw: fn(i32) -> V, encode: fn(usize) -> [V; 6])
where
    Tanh: Activation<V>,
{
    let mut rng = GlibcRand::new(black_box(1));
    let net = MultilayerPerceptron::<V, Tanh, Linear, 6, 8, 4>::new(|| {
        raw(rng.next_i32() % 65_537 - 32_768)
    });
    let q_values = qlearn::QNetwork::new(net, encode, black_box(10));
    let mut q = qlearn::QLearner::<V, _, 6, 4, _>::with_q_values(
        OpaqueEnvironment,
        q_values,
        raw(black_box(13_107)),
        raw(black_box(52_428)),
        black_box(50),
    );
    for _ in 0..black_box(3) {
        let action = q.take_action(black_box(0)).unwrap_or(black_box(9));
        q.update_from_experience(&qlearn::Experience {
            state: black_box(0),
            action,
            reward: raw(black_box(65_536)),
            new_state: black_box(3),
        });
    }
    q.set_q_value(black_box(7), black_box(9), raw(black_box(1)));
    let values = q.q_values_mut();
    black_box((
        qlearn::QValues::q_value(values, black_box(2), black_box(5)),
        qlearn::QValues::future_q_value(values, black_box(1), black_box(3)),
        values.target(),
        values.network().learned_values(),
    ));
}

/// The fine-tuning heads: a dense head with each loss (train, the pieces,
/// load and clear) and a prototype head with each distance, with classes and
/// caps the compiler cannot see.
fn heads<V: SoftmaxExp>(raw: fn(i32) -> V) {
    let input: [V; 8] = core::array::from_fn(|i| raw(black_box(i as i32 * 9_000 - 30_000)));
    let rate = raw(black_box(3_277));

    let mut dense = DenseHead::<V, 8, 6>::new();
    let mut rng = GlibcRand::new(black_box(1));
    dense.initialize_weights(|| raw(rng.next_i32() % 65_537 - 32_768));
    let mut deltas = [raw(0); 8];
    black_box(dense.train_class(&input, black_box(2), rate, Some(&mut deltas)));
    black_box(dense.train_class(&input, black_box(9), rate, None));
    let logits = dense.forward(&input);
    let targets = black_box([raw(65_536); 6]);
    let out = dense.output_deltas(&logits, &targets);
    black_box(dense.backward(&out));
    dense.update(&out, &input, rate);
    black_box(SoftmaxCrossEntropyLoss::probabilities(&logits));
    let shipped = dense;
    dense.clear();
    dense.load_weights(&shipped.weights, &shipped.biases);
    black_box((dense.predict(&input), deltas));

    let mut regression = DenseHead::<V, 8, 2, SquaredErrorLoss>::from_weights(
        black_box([[raw(1); 8]; 2]),
        black_box([raw(0); 2]),
    );
    black_box(regression.train(&input, &black_box([raw(1); 2]), rate, None));

    let mut prototypes = PrototypeHead::<V, 8, 6>::new();
    prototypes.learn(&input, black_box(3));
    prototypes.learn(&input, black_box(17));
    prototypes.set_maximum_count(black_box(0));
    prototypes.set_prototype(black_box(1), &input, black_box(9));
    prototypes.forget(black_box(4));
    black_box((
        prototypes.predict(&input),
        prototypes.distance(&input, black_box(5)),
        prototypes.count(black_box(7)),
        prototypes.prototype(black_box(1)).is_some(),
        prototypes.maximum_count(),
    ));
    prototypes.clear();
    let mut manhattan = PrototypeHead::<V, 8, 6, ManhattanDistance>::default();
    manhattan.learn(&input, black_box(0));
    black_box(manhattan.predict(&input));
}

fn one_hot_q16(state: usize) -> [Q16_16; 6] {
    core::array::from_fn(|i| Q16_16::from_int(i32::from(i == state)))
}

#[unsafe(no_mangle)]
pub extern "C" fn _start() -> ! {
    arithmetic();
    activations();
    int8_inference();
    int8_conv();
    int8_fft::<2, 1>();
    int8_fft::<256, 128>();
    // The int8 recurrent cells: both cell-state types, an odd shape and a
    // larger one.
    int8_recurrent::<i8, 3, 16>(black_box(0));
    int8_recurrent::<i16, 3, 16>(black_box(0));
    int8_recurrent::<i8, 5, 7>(black_box(0));
    trees::<i8>(black_box(1), black_box(2));
    trees::<u8>(black_box(1), black_box(2));
    trees::<i16>(black_box(1), black_box(2));
    black_box(imported::forward(&black_box([0i8; imported::NUM_INPUTS])));
    let q = black_box([0i8; imported_trees::NUM_FEATURES]);
    black_box((
        imported_trees::predict(&q),
        imported_trees::predict_inline(&q),
    ));
    black_box(imported_cnn::forward(&black_box([[[0i8; 1]; 12]; 12])));
    black_box(imported_cnn_flat::forward(&black_box([[[0i8; 2]; 11]; 9])));
    let f = black_box([0i16; imported_forest::NUM_FEATURES]);
    black_box((
        imported_forest::predict(&f),
        imported_forest::predict_inline(&f),
    ));
    black_box(imported_trees::quantize_features(&black_box(
        [0.0; imported_trees::NUM_FEATURES],
    )));
    // The examples' configurations: xor 2-4-1 (both formats) and iris 4-8-3.
    network::<Q16_16, Tanh, Sigmoid, 2, 4, 1>(Q16_16::from_raw);
    network::<Q8_8, Tanh, Sigmoid, 2, 4, 1>(|r| Q8_8::from_raw(r as i16));
    network::<Q16_16, Relu, Sigmoid, 4, 8, 3>(Q16_16::from_raw);
    // Multiple hidden layers, both formats.
    deep_network::<Q16_16, Relu, Sigmoid>(Q16_16::from_raw);
    // A regression head (linear output), as in the energy_efficiency example.
    network::<Q16_16, Relu, Linear, 8, 16, 2>(Q16_16::from_raw);
    deep_network::<Q16_16, Relu, Linear>(Q16_16::from_raw);
    // The adaptive optimizers.
    adaptive::<Q16_16>(Q16_16::from_raw);
    adaptive::<Q8_8>(|r| Q8_8::from_raw(r as i16));
    deep_network::<Q8_8, Tanh, Sigmoid>(|r| Q8_8::from_raw(r as i16));
    // Floating point (Tanh and Sigmoid from the libm crate without std).
    network::<f32, Tanh, Sigmoid, 2, 4, 1>(|r| r as f32 / 65536.0);
    network::<f64, Relu, Linear, 8, 16, 2>(|r| f64::from(r) / 65536.0);
    deep_network::<f64, Relu, Sigmoid>(|r| f64::from(r) / 65536.0);
    deep_network::<f32, Tanh, Sigmoid>(|r| r as f32 / 65536.0);
    adaptive_float();
    // Q-learning, both formats and floating point.
    lstm::<Q16_16, Tanh, Sigmoid, 3, 16, 4, _>(Q16_16::from_raw, Sgd);
    lstm::<Q8_8, Relu, Linear, 2, 4, 2, _>(|r| Q8_8::from_raw(r as i16), Sgd);
    lstm::<Q16_16, Tanh, Sigmoid, 2, 4, 2, _>(
        Q16_16::from_raw,
        Adam::new(
            Q16_16::from_raw(black_box(58982)),
            Q16_16::from_raw(black_box(65470)),
            Q16_16::from_raw(black_box(16)),
        ),
    );
    lstm::<f32, Tanh, Sigmoid, 2, 3, 1, _>(|r| r as f32 / 65536.0, Sgd);
    lstm::<f64, Relu, Linear, 2, 3, 2, _>(|r| f64::from(r) / 65536.0, AdamFloat::default());
    q_learning::<Q16_16>(Q16_16::from_raw);
    q_learning::<Q8_8>(|r| Q8_8::from_raw(r as i16));
    q_learning::<f32>(|r| r as f32 / 65536.0);
    dqn::<Q16_16>(Q16_16::from_raw, one_hot_q16);
    // The fine-tuning heads (the float softmax from the libm crate without std).
    heads::<Q16_16>(Q16_16::from_raw);
    heads::<Q8_8>(|r| Q8_8::from_raw(r as i16));
    heads::<f32>(|r| r as f32 / 65536.0);
    heads::<f64>(|r| f64::from(r) / 65536.0);
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
