//! Minimal Rust port of the tinymind fixed-point neural network library.
//!
//! Scope: enough of tinymind to run its XOR example bit-for-bit: Q16.16
//! arithmetic, the tanh and sigmoid lookup-table activations, and a trainable
//! one-hidden-layer perceptron. The library is `no_std` and allocation-free, like
//! tinymind's freestanding build.
#![no_std]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod activation;
mod lut;
pub mod neuralnet;
pub mod qformat;
pub mod rand;

pub use activation::{Activation, Sigmoid, Tanh};
pub use neuralnet::MultilayerPerceptron;
pub use qformat::{Fixed, Q8_8, Q16_16, QValue};
