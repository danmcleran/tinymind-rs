// QSoftmax against the TFLite Micro reference int8 softmax: the golden rows,
// parameters and sweep hashes of tinymind's unit test
// (unit_test/quantization/quantization_unit_test.cpp, qsoftmax_fixedpoint_*),
// which come from tflite::micro::reference_ops::Softmax with the parameters
// TFLite Micro computes. Extracted from that file by script, not retyped.
// scripts/compare-with-cpp.sh also sweeps tinymind's C++ kernel against this one.

use tinymind::int8::QSoftmax;

struct Golden {
    #[cfg_attr(not(feature = "std"), allow(dead_code))]
    input_scale: f64,
    #[cfg_attr(not(feature = "std"), allow(dead_code))]
    beta: f64,
    params: QSoftmax,
    /// Four rows of 12: random, near-equal, one dominant class, random.
    input: [i8; 48],
    output: [i8; 48],
}

/// The last case's multiplier is capped (input scale 40), which leaves
/// diff_min at 0.
const GOLDENS: [Golden; 6] = [
    Golden {
        input_scale: 0.1,
        beta: 1.0,
        params: QSoftmax {
            input_multiplier: 1717986918,
            input_left_shift: 23,
            diff_min: -248,
        },
        input: [
            -109, -70, 71, -47, -16, 122, 57, -12, 122, -50, 9, -61, 90, 91, 92, 90, 91, 92, 90,
            91, 92, 90, 91, 92, -128, -128, -128, -128, -128, 127, -128, -128, -128, -128, -128,
            -128, 110, -61, -122, 64, 25, -63, 115, 89, -70, -84, 12, 74,
        ],
        output: [
            -128, -128, -127, -128, -128, 0, -128, -128, 0, -128, -128, -128, -109, -107, -105,
            -109, -107, -105, -109, -107, -105, -109, -107, -105, -128, -128, -128, -128, -128,
            127, -128, -128, -128, -128, -128, -128, -37, -128, -128, -127, -128, -128, 22, -117,
            -128, -128, -128, -126,
        ],
    },
    Golden {
        input_scale: 0.0625,
        beta: 1.0,
        params: QSoftmax {
            input_multiplier: 1073741824,
            input_left_shift: 23,
            diff_min: -248,
        },
        input: [
            104, 112, -94, -14, 5, -30, 64, -37, 43, -79, -9, -103, 90, 91, 92, 90, 91, 92, 90, 91,
            92, 90, 91, 92, -128, -128, -128, -128, -128, 127, -128, -128, -128, -128, -128, -128,
            40, -56, -34, 107, -11, -18, 56, -21, -23, 124, 104, 75,
        ],
        output: [
            -35, 25, -128, -128, -128, -128, -120, -128, -126, -128, -128, -128, -108, -107, -105,
            -108, -107, -105, -108, -107, -105, -108, -107, -105, -128, -128, -128, -128, -128,
            127, -128, -128, -128, -128, -128, -128, -127, -128, -128, -76, -128, -128, -126, -128,
            -128, 23, -85, -121,
        ],
    },
    Golden {
        input_scale: 0.5,
        beta: 1.0,
        params: QSoftmax {
            input_multiplier: 1073741824,
            input_left_shift: 26,
            diff_min: -31,
        },
        input: [
            -82, -41, 61, -52, -20, 1, -19, 31, 34, 40, 5, -62, 90, 91, 92, 90, 91, 92, 90, 91, 92,
            90, 91, 92, -128, -128, -128, -128, -128, 127, -128, -128, -128, -128, -128, -128, 119,
            -92, 113, -14, 89, 40, -8, 41, 87, 86, -95, 8,
        ],
        output: [
            -128, -128, 127, -128, -128, -128, -128, -128, -128, -128, -128, -128, -116, -108, -96,
            -116, -108, -96, -116, -108, -96, -116, -108, -96, -128, -128, -128, -128, -128, 127,
            -128, -128, -128, -128, -128, -128, 116, -128, -116, -128, -128, -128, -128, -128,
            -128, -128, -128, -128,
        ],
    },
    Golden {
        input_scale: 0.00390625,
        beta: 1.0,
        params: QSoftmax {
            input_multiplier: 1073741824,
            input_left_shift: 19,
            diff_min: -3968,
        },
        input: [
            -49, -118, -10, -104, 61, 76, -4, -25, -93, 46, -41, 0, 90, 91, 92, 90, 91, 92, 90, 91,
            92, 90, 91, 92, -128, -128, -128, -128, -128, 127, -128, -128, -128, -128, -128, -128,
            76, -60, 112, -68, -118, -33, 96, -30, -58, -69, -7, -84,
        ],
        output: [
            -109, -114, -106, -113, -99, -98, -106, -108, -112, -101, -109, -105, -107, -107, -107,
            -107, -107, -107, -107, -107, -107, -107, -107, -107, -109, -109, -109, -109, -109,
            -77, -109, -109, -109, -109, -109, -109, -98, -110, -94, -111, -114, -109, -96, -108,
            -110, -111, -106, -112,
        ],
    },
    Golden {
        input_scale: 0.05,
        beta: 2.5,
        params: QSoftmax {
            input_multiplier: 1073741824,
            input_left_shift: 24,
            diff_min: -124,
        },
        input: [
            75, -26, 55, 87, -91, 73, 40, 36, -111, -2, -37, -7, 90, 91, 92, 90, 91, 92, 90, 91,
            92, 90, 91, 92, -128, -128, -128, -128, -128, 127, -128, -128, -128, -128, -128, -128,
            84, 22, 113, -81, -96, 79, -70, 20, 40, -14, -95, 76,
        ],
        output: [
            -88, -128, -125, 52, -128, -97, -127, -128, -128, -128, -128, -128, -109, -107, -104,
            -109, -107, -104, -109, -107, -104, -109, -107, -104, -128, -128, -128, -128, -128,
            127, -128, -128, -128, -128, -128, -128, -122, -128, 116, -128, -128, -125, -128, -128,
            -128, -128, -128, -126,
        ],
    },
    Golden {
        input_scale: 40.0,
        beta: 1.0,
        params: QSoftmax {
            input_multiplier: 2147483647,
            input_left_shift: 31,
            diff_min: 0,
        },
        input: [
            -71, 11, 19, -21, -85, -124, 72, 47, 91, -4, -120, -72, 90, 91, 92, 90, 91, 92, 90, 91,
            92, 90, 91, 92, -128, -128, -128, -128, -128, 127, -128, -128, -128, -128, -128, -128,
            42, -28, 78, 113, 12, -41, -86, 52, -119, -10, -56, -29,
        ],
        output: [
            -128, -128, -128, -128, -128, -128, -128, -128, 127, -128, -128, -128, -128, -128, -64,
            -128, -128, -64, -128, -128, -64, -128, -128, -64, -128, -128, -128, -128, -128, 127,
            -128, -128, -128, -128, -128, -128, -128, -128, -128, 127, -128, -128, -128, -128,
            -128, -128, -128, -128,
        ],
    },
];

#[cfg(feature = "std")]
#[test]
fn params_match_tflite_micro() {
    for g in &GOLDENS {
        let p = tinymind::int8::calibration::build_softmax(g.input_scale, g.beta);
        assert_eq!(p, g.params, "input_scale {} beta {}", g.input_scale, g.beta);
    }
}

#[test]
fn rows_match_tflite_micro() {
    for g in &GOLDENS {
        for r in 0..4 {
            let row: [i8; 12] = core::array::from_fn(|i| g.input[r * 12 + i]);
            let want: [i8; 12] = core::array::from_fn(|i| g.output[r * 12 + i]);
            assert_eq!(g.params.forward(&row), want, "{:?} row {r}", g.params);
        }
    }
}

/// 4096 rows of 12 per golden case from a fixed LCG, every fourth row in a
/// narrow band so logits tie or sit one step apart; FNV-1a of the outputs.
/// The goldens above are too few to catch a defect that changes 1 row in 200
/// (two Newton steps instead of three); this is not.
#[test]
fn sweep_matches_tflite_micro() {
    const EXPECTED: [u32; 6] = [
        0x1723FB42, 0x89B83D62, 0x5B984723, 0xF5D0854C, 0x5FFB453A, 0xAFE86A52,
    ];
    for (c, g) in GOLDENS.iter().enumerate() {
        let mut state: u32 = 0x1234 + c as u32;
        let mut hash: u32 = 0x811C_9DC5;
        for r in 0..4096 {
            let input: [i8; 12] = core::array::from_fn(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let v = state >> 24;
                if r % 4 == 3 {
                    (96 + (v & 7)) as i8
                } else {
                    (v as i32 - 128) as i8
                }
            });
            for y in g.params.forward(&input) {
                hash ^= u32::from(y as u8);
                hash = hash.wrapping_mul(0x0100_0193);
            }
        }
        assert_eq!(hash, EXPECTED[c], "case {c}");
    }
}

/// Four equal logits: exactly 1/4 each, 64/256 on the int8 grid.
#[cfg(feature = "std")]
#[test]
fn equal_inputs_split_evenly() {
    let softmax = tinymind::int8::calibration::build_softmax(0.1, 1.0);
    assert_eq!(softmax.forward(&[17i8; 4]), [-64; 4]); // 64 - 128
}
