// QDecisionTree and QGbdt against tinymind's QDecisionTree and QGBDT over
// random ensembles. tests/golden/tree/stdout.txt is the output of
// scripts/cpp/tree_sweep.cpp, which builds the same pools from the same
// generator and runs the same inputs; scripts/compare-with-cpp.sh rebuilds it
// and checks the golden is current. See that program for what is covered.
// The sweep itself is in tests/sweeps/tree.rs, which bench/ also runs on
// Cortex-M.

#[path = "sweeps/tree.rs"]
mod tree;

#[test]
fn trees_match_cpp_sweep() {
    let mut out = String::new();
    tree::report(&mut out);
    let golden = include_str!("golden/tree/stdout.txt");
    for (i, (got, want)) in out.lines().zip(golden.lines()).enumerate() {
        assert_eq!(got, want, "line {}", i + 1);
    }
    assert_eq!(out, golden);
}
