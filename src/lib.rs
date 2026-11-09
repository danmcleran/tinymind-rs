//! Rust port of the tinymind (C++) neural network library. So far only fixed-point
//! value types are ported; tinymind (C++) also runs its networks with `float` and `double`.
//!
//! Scope: Q16.16 and Q8.8 arithmetic and a fixed-point square root ([`Sqrt`]); ReLU, [`Linear`] and the tanh and sigmoid
//! lookup-table activations; trainable networks with any number of hidden layers
//! ([`NeuralNetwork`] over [`Dense`] layers, trained with SGD and momentum,
//! [`Adam`] or [`RmsProp`], with optional gradient clipping) and their
//! inference-only form ([`InferenceNetwork`] over [`InferenceDense`] layers:
//! the weights only, tinymind (C++)'s `IsTrainable = false`); a one-hidden-layer
//! perceptron that
//! trains on the device ([`MultilayerPerceptron`]) or only runs trained weights
//! ([`InferencePerceptron`], tinymind (C++)'s `IsTrainable = false`), int8
//! affine-quantized inference ([`int8`]: dense layers, 2D convolutions (standard,
//! depthwise and pointwise), pooling, elementwise addition, quantized activations
//! and a TFLite-exact softmax), and quantized decision trees and gradient-boosted
//! ensembles ([`tree`]). Every part produces the same bits as
//! tinymind (C++). [`rand`] re-implements glibc's `rand()`, which tinymind (C++)'s
//! examples use for weight initialisation and sampling, so they can be
//! reproduced bit for bit. The library is `no_std` and allocation-free, like tinymind (C++)'s
//! freestanding build.
//!
//! # Example
//!
//! Train a 2-4-1 network on XOR in Q16.16, as tinymind (C++)'s `examples/xor` does
//! (the `xor` example in this repository is the full port, bit-exact with the C++):
//!
//! ```
//! use tinymind::rand::GlibcRand;
//! use tinymind::{MultilayerPerceptron, Q16_16, Sigmoid, Tanh};
//!
//! // 2 inputs, 4 tanh hidden neurons, 1 sigmoid output.
//! type Net = MultilayerPerceptron<Q16_16, Tanh, Sigmoid, 2, 4, 1>;
//!
//! // Initial weights in [-0.5, 0.5] (raw Q16.16 units), from glibc-compatible rand().
//! let mut init = GlibcRand::new(1);
//! let mut nn = Net::new(|| Q16_16::from_raw(init.next_i32() % 65537 - 32768));
//!
//! let mut rng = GlibcRand::new(7);
//! for _ in 0..20_000 {
//!     let (a, b) = (rng.next_i32() & 1, rng.next_i32() & 1);
//!     let target = [Q16_16::from_int(a ^ b)];
//!     nn.feed_forward(&[Q16_16::from_int(a), Q16_16::from_int(b)]);
//!     if !nn.calculate_error(&target).is_within_zero_tolerance() {
//!         nn.train(&target);
//!     }
//! }
//!
//! for (a, b) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
//!     nn.feed_forward(&[Q16_16::from_int(a), Q16_16::from_int(b)]);
//!     let p = nn.learned_values()[0].to_f64();
//!     assert_eq!(p >= 0.5, a ^ b == 1, "{a} ^ {b}: p = {p}");
//! }
//! ```
#![no_std]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

// Host-side calibration (`int8::calibration`) needs f32::exp and f32::tanh.
#[cfg(feature = "std")]
extern crate std;

pub mod activation;
pub mod int8;
mod lut;
pub mod network;
pub mod neuralnet;
pub mod optimizer;
pub mod qformat;
pub mod rand;
pub mod tree;

#[cfg(kani)]
mod kani_proofs;

pub use activation::{Activation, Linear, Relu, Sigmoid, Tanh};
pub use network::{
    Dense, InferenceDense, InferenceLayers, InferenceNetwork, Layers, NeuralNetwork,
};
pub use neuralnet::{InferencePerceptron, MultilayerPerceptron};
pub use optimizer::{Adam, Optimizer, RmsProp, Sgd};
pub use qformat::{Fixed, Q8_8, Q16_16, QValue, Sqrt};
