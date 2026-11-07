// Golden test for a model written by the PyTorch importer.
//
// tests/golden/imported_mlp/stdout.txt is the output of scripts/cpp/imported_mlp.cpp:
// the same example model imported with tinymind's own importer and run on its C++
// runtime (scripts/compare-with-cpp.sh rebuilds and diffs it).

#[path = "../examples/common/imported_mlp_driver.rs"]
mod imported_mlp_driver;

#[test]
fn imported_mlp_matches_cpp_golden() {
    let mut out = Vec::new();
    imported_mlp_driver::run(&mut out).unwrap();
    let golden = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/golden/imported_mlp/stdout.txt"
    ))
    .unwrap();
    assert_eq!(
        String::from_utf8_lossy(&out),
        String::from_utf8_lossy(&golden)
    );
}
