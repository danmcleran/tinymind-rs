// Golden test for the kws_cortex_m_int8 example: every tensor (quantized
// input, weights, requantizers and each layer's output) and the logits report
// (int8 logits against the float model's) against
// tests/golden/kws_int8/tensors.txt, which scripts/cpp/kws_int8_dump.cpp
// prints from tinymind (C++)'s example, compiled unchanged.
// scripts/compare-with-cpp.sh rebuilds it and checks the golden is current.
// The rest of the example's stdout (host timings, each build's sizes) is not
// compared.

#[path = "../examples/common/kws_int8_driver.rs"]
mod kws_int8_driver;

#[test]
fn kws_int8_matches_cpp_golden() {
    let mut stdout = Vec::new();
    let tensors = kws_int8_driver::run(&mut stdout, || 0).expect("kws example failed");
    let text = String::from_utf8(stdout).unwrap();
    let report = text.find("\nLogits (int8").expect("no logits report");
    let ours = format!(
        "{}--- stdout from the logits report on\n{}",
        kws_int8_driver::dump(&tensors),
        &text[report + 1..]
    );
    let golden = include_str!("golden/kws_int8/tensors.txt");
    for (n, (a, b)) in ours.lines().zip(golden.lines()).enumerate() {
        assert_eq!(a, b, "line {} differs", n + 1);
    }
    assert_eq!(ours, golden);
}

/// The generator is the standard's mt19937: its 10,000th output from the
/// default seed is 4123659995 ([rand.predef] in the C++ standard).
#[test]
fn mt19937_matches_the_standard() {
    let mut rng = kws_int8_driver::std_random::Mt19937::new(5489);
    let mut v = 0;
    for _ in 0..10_000 {
        v = rng.next_u32();
    }
    assert_eq!(v, 4_123_659_995);
}
