//! Head-only fine-tuning: small classifiers that sit on top of a frozen model
//! and learn on the device from a few labelled samples. Ported from
//! tinymind (C++) `cpp/densehead.hpp` and `cpp/prototypehead.hpp`.
//!
//! A model trained on the host meets data on the device that it never saw: a
//! sensor ages, a user moves differently, a new class appears. Retraining the
//! whole network on a microcontroller needs the training state of every layer.
//! Head-only fine-tuning keeps the shipped model frozen and adapts only a head
//! on top of its embedding. Nothing is backpropagated through the frozen model,
//! so there are no stored activations and no gradients for it, and neither
//! head allocates.
//!
//! - [`DenseHead`] is a trainable dense layer, trained by SGD with a softmax
//!   cross-entropy loss ([`SoftmaxCrossEntropyLoss`], the default) or a
//!   squared-error loss ([`SquaredErrorLoss`]). The update is fused into the
//!   backward pass, so its RAM is its parameters: `(IN + 1) x OUT` values.
//! - [`PrototypeHead`] is a nearest-class-mean classifier: one running-mean
//!   embedding per class, no learning rate, no epochs, and a class can be
//!   added at any time. Distance: [`SquaredEuclideanDistance`] (the default)
//!   or [`ManhattanDistance`].
//!
//! The frozen model can be anything that produces an embedding: an
//! [`InferenceNetwork`](crate::InferenceNetwork), an [`int8`](crate::int8)
//! pipeline whose output the application converts to a Q format, or the
//! layers in front of the last one of a trained network. The `gas_sensor_drift` example adapts a Q16.16
//! 128 -> 32 -> 6 classifier to drifted sensor data both ways.
//!
//! Both heads run on [`Q16_16`], [`Q8_8`], `f32` and `f64`. The softmax needs
//! an exponential ([`SoftmaxExp`]): the 96-entry exp lookup table for the Q
//! formats, the math library for the float types (feature `std` or `libm`).
//! [`SquaredErrorLoss`] and [`PrototypeHead`] need only arithmetic.
//!
//! Every operation produces the same bits as tinymind (C++) (floating point:
//! with `std` on a glibc host).
//!
//! ```
//! use tinymind::Q16_16;
//! use tinymind::head::{DenseHead, PrototypeHead};
//!
//! // A 4-value embedding and 2 classes.
//! let a = [1, 1, 0, 0].map(Q16_16::from_int);
//! let b = [0, 0, 1, 1].map(Q16_16::from_int);
//!
//! let mut dense = DenseHead::<Q16_16, 4, 2>::new();
//! let rate = Q16_16::pow2_neg(2); // 0.25
//! for _ in 0..20 {
//!     dense.train_class(&a, 0, rate, None);
//!     dense.train_class(&b, 1, rate, None);
//! }
//! assert_eq!((dense.predict(&a), dense.predict(&b)), (0, 1));
//!
//! let mut prototypes = PrototypeHead::<Q16_16, 4, 2>::new();
//! prototypes.learn(&a, 0);
//! prototypes.learn(&b, 1);
//! assert_eq!((prototypes.predict(&a), prototypes.predict(&b)), (0, 1));
//! ```

use core::marker::PhantomData;

use crate::activation::lookup;
use crate::lut::{EXP_Q8_8, EXP_Q16_16};
use crate::qformat::{Q8_8, Q16_16, Scalar};

/// `exp(x)` for the softmax's shifted logits, `x <= 0` (C++
/// `detail::HeadExp`).
///
/// For the Q formats it is the 96-entry exp lookup table, which spans
/// `[-6, 6]` and clamps below. A class 6 or more behind the largest logit
/// would then keep `e^-6 ~ 0.0025` of the probability, the softmax would never
/// reach a one-hot and the logits of a well separated head would grow until
/// the format wraps; at or below -6 the true value is under 0.0025, so it is
/// taken as 0. For `f32` and `f64` it is the math library's `exp` (feature
/// `std` or `libm`).
pub trait SoftmaxExp: Scalar {
    /// `exp(self)` for `self <= 0`.
    fn softmax_exp(self) -> Self;
}

impl SoftmaxExp for Q16_16 {
    fn softmax_exp(self) -> Self {
        if self <= Q16_16::from_int(-6) {
            Q16_16::ZERO
        } else {
            lookup(self, &EXP_Q16_16)
        }
    }
}

impl SoftmaxExp for Q8_8 {
    fn softmax_exp(self) -> Self {
        if self <= Q8_8::from_int(-6) {
            Q8_8::ZERO
        } else {
            lookup(self, &EXP_Q8_8)
        }
    }
}

#[cfg(feature = "std")]
impl SoftmaxExp for f32 {
    fn softmax_exp(self) -> Self {
        self.exp()
    }
}

#[cfg(feature = "std")]
impl SoftmaxExp for f64 {
    fn softmax_exp(self) -> Self {
        self.exp()
    }
}

#[cfg(all(feature = "libm", not(feature = "std")))]
impl SoftmaxExp for f32 {
    fn softmax_exp(self) -> Self {
        libm::expf(self)
    }
}

#[cfg(all(feature = "libm", not(feature = "std")))]
impl SoftmaxExp for f64 {
    fn softmax_exp(self) -> Self {
        libm::exp(self)
    }
}

/// How a [`DenseHead`] turns its logits and a target into output deltas,
/// following the library's convention `delta = -dL/dz`: a parameter moves by
/// `+learning rate x delta x input`.
pub trait HeadLoss<V> {
    /// `deltas[k]` for `logits[k]` and `targets[k]`.
    fn output_deltas<const N: usize>(logits: &[V; N], targets: &[V; N]) -> [V; N];
}

/// Softmax over the logits with cross-entropy loss, the usual classifier head
/// (C++ `SoftmaxCrossEntropyHeadLoss`): `delta = t - softmax(z)`.
///
/// The largest logit is subtracted before `exp`, so every argument is `<= 0`
/// and every `exp` lies in `[0, 1]`: no format overflows however large the
/// logits grow. See [`SoftmaxExp`] for the classes 6 or more behind.
pub struct SoftmaxCrossEntropyLoss;

impl SoftmaxCrossEntropyLoss {
    /// The softmax of `logits`. In the Q formats the probabilities are
    /// rounded by the format, so they need not sum to exactly 1.
    pub fn probabilities<V: SoftmaxExp, const N: usize>(logits: &[V; N]) -> [V; N] {
        let mut largest = match logits.first() {
            Some(&first) => first,
            None => return *logits,
        };
        for &z in logits.iter().skip(1) {
            if z > largest {
                largest = z;
            }
        }

        let mut probabilities = [V::ZERO; N];
        let mut sum = V::ZERO;
        for (p, &z) in probabilities.iter_mut().zip(logits) {
            *p = (z - largest).softmax_exp();
            sum += *p;
        }

        // exp(0) = 1 is in the sum, so it is never 0.
        if sum != V::ZERO {
            for p in &mut probabilities {
                *p = *p / sum;
            }
        }
        probabilities
    }
}

impl<V: SoftmaxExp> HeadLoss<V> for SoftmaxCrossEntropyLoss {
    fn output_deltas<const N: usize>(logits: &[V; N], targets: &[V; N]) -> [V; N] {
        let mut deltas = Self::probabilities(logits);
        for (d, &t) in deltas.iter_mut().zip(targets) {
            *d = t - *d;
        }
        deltas
    }
}

/// A linear output with squared-error loss `L = 0.5 x sum((t - z)^2)`, for a
/// regression head or a classifier that regresses a one-hot target (C++
/// `SquaredErrorHeadLoss`): `delta = t - z`. Arithmetic only.
pub struct SquaredErrorLoss;

impl<V: Scalar> HeadLoss<V> for SquaredErrorLoss {
    fn output_deltas<const N: usize>(logits: &[V; N], targets: &[V; N]) -> [V; N] {
        let mut deltas = *targets;
        for (d, &z) in deltas.iter_mut().zip(logits) {
            *d = *d - z;
        }
        deltas
    }
}

/// A trainable dense layer to put on top of a frozen model: head-only
/// fine-tuning (C++ `DenseHead`).
///
/// The frozen model turns a sample into an embedding of `IN` values; the head
/// maps it to `OUT` logits and learns from labelled samples. The training
/// state is the head's parameters and one embedding: the update is fused into
/// the backward pass, so there is no gradient buffer, and the RAM is
/// `(IN + 1) x OUT` values.
///
/// The SGD step is plain gradient descent, `learning rate x (delta x input)`,
/// the product with the learning rate rounded to nearest
/// ([`Scalar::mul_round_nearest`]), as the crate's SGD step is.
/// There is no momentum. The Q formats wrap on overflow, so keep the
/// embedding and the logits well inside the format: z-scored inputs and a
/// Q16.16 head are the working recipe, and Q8.8 is too narrow for a wide head.
///
/// To undo an adaptation, reload the shipped parameters with
/// [`load_weights`](Self::load_weights), keeping them in flash rather than
/// spending RAM on a copy.
pub struct DenseHead<V, const IN: usize, const OUT: usize, L = SoftmaxCrossEntropyLoss> {
    /// `weights[o][i]` connects input `i` to output `o`.
    pub weights: [[V; IN]; OUT],
    /// One bias per output.
    pub biases: [V; OUT],
    loss: PhantomData<L>,
}

impl<V: Copy, const IN: usize, const OUT: usize, L> Clone for DenseHead<V, IN, OUT, L> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<V: Copy, const IN: usize, const OUT: usize, L> Copy for DenseHead<V, IN, OUT, L> {}

impl<V: Scalar, const IN: usize, const OUT: usize, L> Default for DenseHead<V, IN, OUT, L> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V, const IN: usize, const OUT: usize, L> DenseHead<V, IN, OUT, L> {
    /// A head with the given parameters, for example the shipped head kept in
    /// flash.
    pub const fn from_weights(weights: [[V; IN]; OUT], biases: [V; OUT]) -> Self {
        Self {
            weights,
            biases,
            loss: PhantomData,
        }
    }
}

impl<V: Scalar, const IN: usize, const OUT: usize, L> DenseHead<V, IN, OUT, L> {
    /// A head with every weight and bias 0. It predicts class 0 for every
    /// input until it is trained.
    pub fn new() -> Self {
        Self::from_weights([[V::ZERO; IN]; OUT], [V::ZERO; OUT])
    }

    /// Set every weight and bias to 0.
    pub fn clear(&mut self) {
        *self = Self::new();
    }

    /// Draw every weight from `random_weight`, output by output, and set the
    /// biases to 0 (C++ `initializeWeights`).
    pub fn initialize_weights(&mut self, mut random_weight: impl FnMut() -> V) {
        for row in &mut self.weights {
            for w in row {
                *w = random_weight();
            }
        }
        self.biases = [V::ZERO; OUT];
    }

    /// Replace every parameter, for example to restore the shipped head.
    pub fn load_weights(&mut self, weights: &[[V; IN]; OUT], biases: &[V; OUT]) {
        self.weights = *weights;
        self.biases = *biases;
    }

    /// `logits[o] = bias[o] + sum_i weight[o][i] x input[i]`.
    pub fn forward(&self, input: &[V; IN]) -> [V; OUT] {
        let mut logits = self.biases;
        for (z, row) in logits.iter_mut().zip(&self.weights) {
            let mut sum = *z;
            for (&w, &x) in row.iter().zip(input) {
                sum += w * x;
            }
            *z = sum;
        }
        logits
    }

    /// The index of the largest logit (the first one on a tie).
    pub fn predict(&self, input: &[V; IN]) -> usize {
        argmax(&self.forward(input))
    }

    /// `input_deltas[i] = sum_o weight[o][i] x output_deltas[o]`, with the
    /// weights as they are now: the gradient with respect to the embedding
    /// (times -1, the same delta convention), for a caller that also adapts
    /// what lies in front of the head. Call it before [`update`](Self::update).
    pub fn backward(&self, output_deltas: &[V; OUT]) -> [V; IN] {
        let mut input_deltas = [V::ZERO; IN];
        for (row, &delta) in self.weights.iter().zip(output_deltas) {
            for (d, &w) in input_deltas.iter_mut().zip(row) {
                *d += w * delta;
            }
        }
        input_deltas
    }

    /// One SGD step: `weight[o][i] += learning rate x (delta[o] x input[i])`
    /// and `bias[o] += learning rate x delta[o]`, each product with the
    /// learning rate rounded to nearest.
    pub fn update(&mut self, output_deltas: &[V; OUT], input: &[V; IN], learning_rate: V) {
        for ((row, b), &delta) in self
            .weights
            .iter_mut()
            .zip(&mut self.biases)
            .zip(output_deltas)
        {
            update_row(row, input, delta, learning_rate);
            *b += learning_rate.mul_round_nearest(delta);
        }
    }
}

impl<V: Scalar, const IN: usize, const OUT: usize, L: HeadLoss<V>> DenseHead<V, IN, OUT, L> {
    /// The output deltas (`t - y` under the loss) for logits from
    /// [`forward`](Self::forward).
    pub fn output_deltas(&self, logits: &[V; OUT], targets: &[V; OUT]) -> [V; OUT] {
        L::output_deltas(logits, targets)
    }

    /// Forward, loss, optional backward and update on one sample.
    /// `input_deltas`, when given, receives [`backward`](Self::backward) of
    /// this sample's deltas, computed before the update. Returns the
    /// prediction made before the update, for a running accuracy.
    pub fn train(
        &mut self,
        input: &[V; IN],
        targets: &[V; OUT],
        learning_rate: V,
        input_deltas: Option<&mut [V; IN]>,
    ) -> usize {
        let logits = self.forward(input);
        let predicted = argmax(&logits);
        let deltas = L::output_deltas(&logits, targets);
        if let Some(input_deltas) = input_deltas {
            *input_deltas = self.backward(&deltas);
        }
        self.update(&deltas, input, learning_rate);
        predicted
    }

    /// [`train`](Self::train) with a one-hot target for `class`. A `class` of
    /// `OUT` or more gives an all-zero target, as in the C++.
    pub fn train_class(
        &mut self,
        input: &[V; IN],
        class: usize,
        learning_rate: V,
        input_deltas: Option<&mut [V; IN]>,
    ) -> usize {
        let targets = core::array::from_fn(|o| if o == class { V::ONE } else { V::ZERO });
        self.train(input, &targets, learning_rate, input_deltas)
    }
}

/// One row of [`DenseHead::update`]. Never inlined: inlined, LLVM unrolls
/// the loop over the rows at `opt-level = 3`, which cost about 1.5 KB of
/// flash on Cortex-M and made the Cortex-M0+ slower, not faster.
#[inline(never)]
fn update_row<V: Scalar, const IN: usize>(
    row: &mut [V; IN],
    input: &[V; IN],
    delta: V,
    learning_rate: V,
) {
    for (w, &x) in row.iter_mut().zip(input) {
        *w += learning_rate.mul_round_nearest(delta * x);
    }
}

/// The first index of the largest value; 0 for an empty slice.
fn argmax<V: PartialOrd>(values: &[V]) -> usize {
    let mut best = 0;
    for (o, v) in values.iter().enumerate().skip(1) {
        if values.get(best).is_some_and(|b| v > b) {
            best = o;
        }
    }
    best
}

/// The distance a [`PrototypeHead`] compares embeddings with.
pub trait Distance<V> {
    /// The distance between `a` and `b`.
    fn distance<const N: usize>(a: &[V; N], b: &[V; N]) -> V;
}

/// `sum_i (a[i] - b[i])^2`, the usual nearest-class-mean distance (C++
/// `SquaredEuclideanDistance`). In a Q format the sum must stay inside the
/// format (it wraps): with Q16.16, `N x (largest difference)^2` below 32767.
pub struct SquaredEuclideanDistance;

impl<V: Scalar> Distance<V> for SquaredEuclideanDistance {
    fn distance<const N: usize>(a: &[V; N], b: &[V; N]) -> V {
        let mut sum = V::ZERO;
        for (&x, &y) in a.iter().zip(b) {
            let difference = x - y;
            sum += difference * difference;
        }
        sum
    }
}

/// `sum_i |a[i] - b[i]|` (C++ `ManhattanDistance`). No multiplies, so it is
/// cheaper on a core without a fast multiplier and has far more headroom
/// before it overflows.
pub struct ManhattanDistance;

impl<V: Scalar> Distance<V> for ManhattanDistance {
    fn distance<const N: usize>(a: &[V; N], b: &[V; N]) -> V {
        let mut sum = V::ZERO;
        for (&x, &y) in a.iter().zip(b) {
            let difference = x - y;
            sum += if difference < V::ZERO {
                V::ZERO - difference
            } else {
                difference
            };
        }
        sum
    }
}

/// A nearest-class-mean (prototype) classifier to put on top of a frozen
/// model: head-only adaptation without gradients (C++ `PrototypeHead`).
///
/// The head keeps one prototype per class, the running mean of the embeddings
/// it was shown for that class, and predicts the class whose prototype is
/// nearest. Learning is one pass with no learning rate and no epochs; a class
/// can be added at any time; a class never shown is never predicted. RAM:
/// `CLASSES x IN` values plus one `u16` count per class.
///
/// Each [`learn`](Self::learn) moves the prototype by `(input - prototype) /
/// count`. Once a class's count reaches the cap set by
/// [`set_maximum_count`](Self::set_maximum_count), the count stops growing and
/// the prototype becomes an exponential moving average with weight `1 / cap`,
/// so it keeps tracking a slowly drifting input instead of freezing on old
/// samples. The default cap, `u16::MAX`, is effectively unlimited.
///
/// Class indices of `CLASSES` or more are ignored by the methods that change
/// the head, and read as an unlearned class.
pub struct PrototypeHead<V, const IN: usize, const CLASSES: usize, D = SquaredEuclideanDistance> {
    prototypes: [[V; IN]; CLASSES],
    counts: [u16; CLASSES],
    maximum_count: u16,
    distance: PhantomData<D>,
}

impl<V: Copy, const IN: usize, const CLASSES: usize, D> Clone for PrototypeHead<V, IN, CLASSES, D> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<V: Copy, const IN: usize, const CLASSES: usize, D> Copy for PrototypeHead<V, IN, CLASSES, D> {}

impl<V: Scalar, const IN: usize, const CLASSES: usize, D> Default
    for PrototypeHead<V, IN, CLASSES, D>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<V: Scalar, const IN: usize, const CLASSES: usize, D> PrototypeHead<V, IN, CLASSES, D> {
    /// A head that knows no class, with the cap at `u16::MAX`.
    pub fn new() -> Self {
        Self {
            prototypes: [[V::ZERO; IN]; CLASSES],
            counts: [0; CLASSES],
            maximum_count: u16::MAX,
            distance: PhantomData,
        }
    }

    /// Forget every class (the cap is kept).
    pub fn clear(&mut self) {
        self.prototypes = [[V::ZERO; IN]; CLASSES];
        self.counts = [0; CLASSES];
    }

    /// Forget one class: it is not predicted again until it is learned.
    pub fn forget(&mut self, class: usize) {
        if let (Some(p), Some(c)) = (self.prototypes.get_mut(class), self.counts.get_mut(class)) {
            *p = [V::ZERO; IN];
            *c = 0;
        }
    }

    /// Fold one labelled embedding into its class's prototype.
    pub fn learn(&mut self, input: &[V; IN], class: usize) {
        let (Some(prototype), Some(count)) =
            (self.prototypes.get_mut(class), self.counts.get_mut(class))
        else {
            return;
        };
        if *count < self.maximum_count {
            *count += 1;
        }
        let n = i32::from(*count);
        for (p, &x) in prototype.iter_mut().zip(input) {
            let difference = x - *p;
            *p += difference.div_int(n);
        }
    }

    /// The cap on a class's count (at least 1; 0 is taken as 1). Counts
    /// above the new cap are lowered to it.
    pub fn set_maximum_count(&mut self, maximum_count: u16) {
        self.maximum_count = maximum_count.max(1);
        for c in &mut self.counts {
            if *c > self.maximum_count {
                *c = self.maximum_count;
            }
        }
    }

    /// The cap on a class's count.
    pub fn maximum_count(&self) -> u16 {
        self.maximum_count
    }

    /// How many samples a class's prototype holds (0: not learned).
    pub fn count(&self, class: usize) -> u16 {
        self.counts.get(class).copied().unwrap_or(0)
    }

    /// A class's prototype, or `None` for a class index out of range.
    pub fn prototype(&self, class: usize) -> Option<&[V; IN]> {
        self.prototypes.get(class)
    }

    /// Load one class's prototype, for example a shipped one, with the weight
    /// it carries against later samples (capped; count 0 forgets the class).
    pub fn set_prototype(&mut self, class: usize, prototype: &[V; IN], count: u16) {
        if let (Some(p), Some(c)) = (self.prototypes.get_mut(class), self.counts.get_mut(class)) {
            *p = *prototype;
            *c = count.min(self.maximum_count);
        }
    }
}

impl<V: Scalar, const IN: usize, const CLASSES: usize, D: Distance<V>>
    PrototypeHead<V, IN, CLASSES, D>
{
    /// The distance from `input` to a class's prototype, or `None` for a
    /// class index out of range.
    pub fn distance(&self, input: &[V; IN], class: usize) -> Option<V> {
        self.prototypes.get(class).map(|p| D::distance(input, p))
    }

    /// The learned class whose prototype is nearest (the first one on a tie),
    /// or `CLASSES` when no class has been learned.
    pub fn predict(&self, input: &[V; IN]) -> usize {
        let mut best = CLASSES;
        let mut best_distance = V::ZERO;
        for (c, (p, &count)) in self.prototypes.iter().zip(&self.counts).enumerate() {
            if count == 0 {
                continue;
            }
            let d = D::distance(input, p);
            if best == CLASSES || d < best_distance {
                best = c;
                best_distance = d;
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_exp_is_zero_beyond_the_table() {
        assert_eq!(Q16_16::from_int(-6).softmax_exp(), Q16_16::ZERO);
        assert_eq!(Q16_16::from_int(-100).softmax_exp(), Q16_16::ZERO);
        assert!(Q16_16::from_raw(-6 * 65536 + 1).softmax_exp() > Q16_16::ZERO);
        assert_eq!(Q16_16::ZERO.softmax_exp(), Q16_16::ONE);
        assert_eq!(Q8_8::from_int(-6).softmax_exp(), Q8_8::ZERO);
        assert_eq!(Q8_8::ZERO.softmax_exp(), Q8_8::ONE);
    }

    #[test]
    fn softmax_of_equal_logits_is_uniform() {
        let p = SoftmaxCrossEntropyLoss::probabilities(&[Q16_16::from_int(3); 4]);
        assert_eq!(p, [Q16_16::pow2_neg(2); 4]);
    }

    #[test]
    fn a_separated_head_stops_moving() {
        // Logits 10 apart: the softmax is one-hot, so the deltas are 0.
        let mut head = DenseHead::<Q16_16, 1, 2>::from_weights(
            [[Q16_16::ZERO], [Q16_16::ZERO]],
            [Q16_16::from_int(10), Q16_16::ZERO],
        );
        let before = head;
        head.train_class(&[Q16_16::ONE], 0, Q16_16::ONE, None);
        assert_eq!(head.biases, before.biases);
    }

    #[test]
    fn train_class_out_of_range_trains_toward_zero() {
        let mut head = DenseHead::<Q16_16, 2, 2, SquaredErrorLoss>::from_weights(
            [[Q16_16::ONE; 2]; 2],
            [Q16_16::ZERO; 2],
        );
        let input = [Q16_16::ONE; 2];
        let mut deltas = [Q16_16::ZERO; 2];
        head.train_class(&input, 7, Q16_16::pow2_neg(3), Some(&mut deltas));
        // Logits were 2, targets 0: delta -2, backward -2 - 2 = -4.
        assert_eq!(deltas, [Q16_16::from_int(-4); 2]);
        assert_eq!(head.biases, [Q16_16::from_raw(-16384); 2]);
    }

    #[test]
    fn prototype_head_running_mean_and_cap() {
        let mut head = PrototypeHead::<Q16_16, 1, 3>::new();
        assert_eq!(head.predict(&[Q16_16::ZERO]), 3);
        for v in [1, 2, 3] {
            head.learn(&[Q16_16::from_int(v)], 1);
        }
        assert_eq!(head.prototype(1), Some(&[Q16_16::from_int(2)]));
        assert_eq!(head.count(1), 3);
        assert_eq!(head.predict(&[Q16_16::from_int(-50)]), 1);

        head.set_maximum_count(2);
        assert_eq!(head.count(1), 2);
        head.learn(&[Q16_16::from_int(4)], 1);
        assert_eq!(head.prototype(1), Some(&[Q16_16::from_int(3)]));

        head.learn(&[Q16_16::from_int(10)], 7); // ignored
        head.forget(1);
        assert_eq!(head.predict(&[Q16_16::ZERO]), 3);
        assert_eq!(head.distance(&[Q16_16::ZERO], 3), None);
    }

    #[test]
    fn manhattan_distance() {
        let a = [Q16_16::from_int(1), Q16_16::from_int(-2)];
        let b = [Q16_16::from_int(-1), Q16_16::from_int(1)];
        assert_eq!(ManhattanDistance::distance(&a, &b), Q16_16::from_int(5));
        assert_eq!(
            SquaredEuclideanDistance::distance(&a, &b),
            Q16_16::from_int(13)
        );
    }
}
