// Golden test for the convolutional models written by the PyTorch importer.
//
// tests/golden/imported_cnn/stdout.txt is the output of scripts/cpp/imported_cnn.cpp:
// the same example models imported with tinymind (C++)'s own importer and run on its
// C++ runtime (scripts/compare-with-cpp.sh rebuilds and diffs it).

#[path = "../examples/common/imported_cnn_driver.rs"]
mod imported_cnn_driver;

#[test]
fn imported_cnn_matches_cpp_golden() {
    let mut out = Vec::new();
    imported_cnn_driver::run(&mut out).unwrap();
    let golden = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/golden/imported_cnn/stdout.txt"
    ))
    .unwrap();
    assert_eq!(
        String::from_utf8_lossy(&out),
        String::from_utf8_lossy(&golden)
    );
}
