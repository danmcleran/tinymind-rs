// Golden test for the models written by the ONNX importer.
//
// tests/golden/imported_onnx/stdout.txt is the output of
// scripts/cpp/imported_onnx.cpp: the same ONNX models read by tinymind (C++)'s
// own ONNX importer and run on its int8 layers (scripts/compare-with-cpp.sh
// rebuilds and diffs it).

#[path = "../examples/common/imported_onnx_driver.rs"]
mod imported_onnx_driver;

#[test]
fn imported_onnx_matches_cpp_golden() {
    let mut out = Vec::new();
    imported_onnx_driver::run(&mut out).unwrap();
    let golden = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/golden/imported_onnx/stdout.txt"
    ))
    .unwrap();
    assert_eq!(
        String::from_utf8_lossy(&out),
        String::from_utf8_lossy(&golden)
    );
}
