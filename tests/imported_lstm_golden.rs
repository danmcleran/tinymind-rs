// Golden test for the LSTM models written by the PyTorch importer.
//
// tests/golden/imported_lstm/stdout.txt is the output of
// scripts/cpp/imported_lstm.cpp: the same example models with every integer
// parameter built by tinymind (C++)'s calibration functions and run on its C++
// runtime (scripts/compare-with-cpp.sh rebuilds and diffs it).

#[path = "../examples/common/imported_lstm_driver.rs"]
mod imported_lstm_driver;

#[test]
fn imported_lstm_matches_cpp_golden() {
    let mut out = Vec::new();
    imported_lstm_driver::run(&mut out).unwrap();
    let golden = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/golden/imported_lstm/stdout.txt"
    ))
    .unwrap();
    assert_eq!(
        String::from_utf8_lossy(&out),
        String::from_utf8_lossy(&golden)
    );
}
