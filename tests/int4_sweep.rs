// The packed int4 layers against tinymind (C++) (cpp/qint4.hpp): QDenseInt4,
// QConv2DInt4 (unpacked and packed paths, odd windows that start filters
// mid-byte), QDepthwiseConv2DInt4, pack_int4 / unpack_int4 and
// calibration::quantize_int4_per_channel, over random tensors with the
// requantizers and draws of conv_sweep. Each layer is also compared with the
// int8 layer on the unpacked weights (same=1). tests/golden/int4/stdout.txt is
// the output of scripts/cpp/int4_sweep.cpp (scripts/compare-with-cpp.sh
// rebuilds it and checks the golden is current).

// The sweep itself is in tests/sweeps/int4.rs, which bench/ also runs on
// Cortex-M.

#[path = "sweeps/int4.rs"]
mod int4;

use tinymind::int8::calibration::quantize_int4_per_channel;

struct Host;

impl int4::Quantize for Host {
    fn quantize(&mut self, w: &[[f32; 9]; 4]) -> Option<([[i8; 9]; 4], [f32; 4])> {
        Some(quantize_int4_per_channel(w))
    }
}

#[test]
fn int4_sweep_matches_cpp() {
    let mut out = String::new();
    int4::run(&mut out, &mut Host);
    let golden = include_str!("golden/int4/stdout.txt");
    for (n, (got, want)) in out.lines().zip(golden.lines()).enumerate() {
        assert_eq!(got, want, "line {}", n + 1);
    }
    assert_eq!(out.lines().count(), golden.lines().count());
}
