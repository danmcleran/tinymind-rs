// Golden test for the kws_cortex_m_int8 example: every tensor (quantized
// input, weights, requantizers and each layer's output) against
// tests/golden/kws_int8/tensors.txt, which scripts/cpp/kws_int8_dump.cpp
// prints from tinymind (C++)'s example, compiled unchanged.
// scripts/compare-with-cpp.sh rebuilds it and checks the golden is current.
//
// The example's stdout is not compared: it carries host timings and each
// build's own sizes, and its logits are all zero (the synthetic weights shrink
// the signal at every layer), so it would check little.

#[path = "../examples/common/kws_int8_driver.rs"]
mod kws_int8_driver;

#[test]
fn kws_int8_tensors_match_cpp_golden() {
    let mut stdout = Vec::new();
    let tensors = kws_int8_driver::run(&mut stdout, || 0).expect("kws example failed");
    let golden = include_str!("golden/kws_int8/tensors.txt");
    let ours = kws_int8_driver::dump(&tensors);
    for (n, (a, b)) in ours.lines().zip(golden.lines()).enumerate() {
        assert_eq!(a, b, "line {} differs", n + 1);
    }
    assert_eq!(ours, golden);

    // The report's last lines follow from the logits.
    let text = String::from_utf8(stdout).unwrap();
    assert!(text.ends_with(
        "\nLogits (int8): 0 0 0 0 0 0 0 0 0 0 \nPredicted class (untrained random weights): 0\n"
    ));
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
