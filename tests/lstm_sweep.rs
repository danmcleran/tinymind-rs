// LstmNetwork against tinymind's LstmNeuralNetwork. tests/golden/lstm/stdout.txt
// is the output of scripts/cpp/lstm_sweep.cpp, which trains the same networks
// on the same sequences from the same initial weights; scripts/compare-with-cpp.sh
// rebuilds it and checks the golden is current. See that program for what is
// covered.
// The sweep itself is in tests/sweeps/lstm.rs, which bench/ also runs on
// Cortex-M.

#[path = "sweeps/lstm.rs"]
mod lstm;

#[test]
fn lstm_matches_cpp_golden() {
    let mut out = String::new();
    lstm::report(&mut out);
    let golden = include_str!("golden/lstm/stdout.txt");
    for (i, (a, g)) in out.lines().zip(golden.lines()).enumerate() {
        assert_eq!(a, g, "line {}", i + 1);
    }
    assert_eq!(out.lines().count(), golden.lines().count(), "line count");
}
