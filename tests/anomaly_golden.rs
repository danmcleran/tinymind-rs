// Regression test for the anomaly-detection demo (examples/anomaly.rs).
//
// Unlike the other goldens, tests/golden/anomaly/stdout.txt is this crate's own
// output: the demo has no tinymind counterpart, so there is no C++ run to take it
// from. Its parts are each checked against tinymind elsewhere (multi-layer
// training, Linear, gradient clipping: tests/deep_mlp.rs). This test pins the
// output so that a change to the arithmetic cannot pass unnoticed, and it pins
// the hash the boards must reproduce (bench/src/anomaly.rs, EXPECT_HASH).

#[path = "../examples/common/anomaly_demo.rs"]
mod anomaly_demo;

#[test]
fn anomaly_demo_output_is_unchanged() {
    let mut report = String::new();
    let mut scores = 0;
    anomaly_demo::run(&mut report, &mut |_, _| scores += 1).unwrap();
    let golden = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/golden/anomaly/stdout.txt"
    ))
    .unwrap();
    assert_eq!(report, golden);
    assert_eq!(scores, 256 + 4 * 64);
    assert!(
        report.contains("hash=0x7C074DAD"),
        "bench/src/anomaly.rs EXPECT_HASH must match"
    );
}
