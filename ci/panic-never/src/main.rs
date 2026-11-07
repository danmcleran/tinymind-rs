//! Calls every public function of `tinymind` with inputs the optimiser cannot
//! see (`black_box`), so no check can be folded away by knowing the values.
//! Linking succeeds only if no panic is reachable from any of it.

#![no_std]
#![no_main]

use core::hint::black_box;

use tinymind::rand::GlibcRand;
use tinymind::{Activation, Fixed, MultilayerPerceptron, Q8_8, Q16_16, Relu, Sigmoid, Tanh};

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
}

#[unsafe(no_mangle)]
pub extern "C" fn _start() -> ! {
    arithmetic();
    activations();
    // The examples' configurations: xor 2-4-1 (both formats) and iris 4-8-3.
    network::<Q16_16, Tanh, Sigmoid, 2, 4, 1>(Q16_16::from_raw);
    network::<Q8_8, Tanh, Sigmoid, 2, 4, 1>(|r| Q8_8::from_raw(r as i16));
    network::<Q16_16, Relu, Sigmoid, 4, 8, 3>(Q16_16::from_raw);
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
