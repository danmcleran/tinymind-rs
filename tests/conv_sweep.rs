// The int8 convolution and pooling layers against tinymind (C++)'s over random
// tensors. tests/golden/conv/stdout.txt is the output of
// scripts/cpp/conv_sweep.cpp, which draws the same tensors, biases and
// requantizers from the same generator, in the same order, and runs the C++
// layers; scripts/compare-with-cpp.sh rebuilds it and checks the golden is
// current. See that program for what the shapes and parameters cover. The
// sweep itself is in tests/sweeps/conv.rs, which bench/ also runs on Cortex-M.

#[path = "sweeps/conv.rs"]
mod conv;

#[test]
fn conv_and_pool_match_tinymind_cpp() {
    let mut text = String::new();
    conv::run(&mut text);
    assert_eq!(text, include_str!("golden/conv/stdout.txt"));
}
