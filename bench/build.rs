//! Put the selected board's memory.x where cortex-m-rt's link.x finds it, and
//! write the FFT workload's twiddle tables (tinymind (C++)'s, read from the FFT
//! sweep's golden, with the hash it expects) to fft_twiddles.rs, and the fine-tuning
//! heads' expected hashes (read from the head sweep's golden) to head_expect.rs,
//! and the features workload's tables and hashes (tinymind (C++)'s
//! gen_tables.cpp output, tests/golden/features/tables.h) to features_tables.rs.
use std::{env, fs, path::PathBuf};

fn main() {
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let boards = [
        ("CARGO_FEATURE_F411", include_str!("memory-f411.x")),
        ("CARGO_FEATURE_L031", include_str!("memory-l031.x")),
        ("CARGO_FEATURE_QEMU_M4", include_str!("memory-qemu-m4.x")),
        ("CARGO_FEATURE_QEMU_M0", include_str!("memory-qemu-m0.x")),
    ];
    let selected: Vec<&str> = boards
        .iter()
        .filter(|(feature, _)| env::var_os(feature).is_some())
        .map(|(_, memory)| *memory)
        .collect();
    let [memory] = selected[..] else {
        panic!("select exactly one board feature: f411, l031, qemu-m4 or qemu-m0");
    };
    fs::write(out.join("memory.x"), memory).unwrap();
    println!("cargo:rustc-link-search={}", out.display());
    println!("cargo:rerun-if-changed=memory-f411.x");
    println!("cargo:rerun-if-changed=memory-l031.x");
    println!("cargo:rerun-if-changed=memory-qemu-m4.x");
    println!("cargo:rerun-if-changed=memory-qemu-m0.x");

    fs::write(out.join("fft_twiddles.rs"), fft_twiddles(256)).unwrap();
    println!("cargo:rerun-if-changed=../tests/golden/fft/stdout.txt");

    fs::write(out.join("head_expect.rs"), head_expect()).unwrap();
    println!("cargo:rerun-if-changed=../tests/golden/head/stdout.txt");

    fs::write(out.join("features_tables.rs"), features_tables()).unwrap();
    println!("cargo:rerun-if-changed=../tests/golden/features/tables.h");
}

/// The features workload's tables as Rust statics, shaped for the layers,
/// and its expected hashes, from gen_tables.cpp's header.
fn features_tables() -> String {
    let header = fs::read_to_string("../tests/golden/features/tables.h").unwrap();
    let values = |name: &str| -> Vec<String> {
        let line = header
            .lines()
            .find(|l| l.contains(&format!(" {name}[")))
            .unwrap_or_else(|| panic!("{name} in the features golden"));
        let body = line.split('{').nth(1).unwrap().split('}').next().unwrap();
        body.split(',').map(|v| v.trim().to_string()).collect()
    };
    // (Rust name, C++ name, element type, row length or 0 for a flat array)
    let tables = [
        ("WINDOW", "kWindow", "i16", 0),
        ("TW_COS", "kTwCos", "i16", 0),
        ("TW_SIN", "kTwSin", "i16", 0),
        ("BAND_START", "kBandStart", "u16", 0),
        ("BAND_LENGTH", "kBandLength", "u16", 0),
        ("MEL_WEIGHTS", "kMelWeights", "u16", 0),
        ("DCT", "kDct", "i16", 32),
        ("CENTROIDS", "kCentroids", "i8", 32),
        ("INV_RADIUS", "kInvRadius", "u32", 0),
        ("FEATURE_WEIGHT", "kFeatureWeight", "u16", 0),
        ("GAUSS_MEANS", "kGaussMeans", "i8", 32),
        ("GAUSS_PRECISION", "kGaussPrecision", "u32", 32),
        ("GAUSS_BIAS", "kGaussBias", "i32", 0),
    ];
    let mut out = String::new();
    for (rust, cpp, ty, row) in tables {
        let v = values(cpp);
        if row == 0 {
            out += &format!(
                "pub static {rust}: [{ty}; {}] = [{}];\n",
                v.len(),
                v.join(", ")
            );
        } else {
            assert_eq!(v.len() % row, 0, "{cpp} rows of {row}");
            let rows: Vec<String> = v
                .chunks(row)
                .map(|r| format!("[{}]", r.join(", ")))
                .collect();
            out += &format!(
                "pub static {rust}: [[{ty}; {row}]; {}] = [{}];\n",
                rows.len(),
                rows.join(", ")
            );
        }
    }
    for name in [
        "HASH_MEL",
        "HASH_MFCC",
        "HASH_KMEANS",
        "HASH_GAUSSIAN",
        "HASH_LEARN",
    ] {
        let line = header
            .lines()
            .find(|l| l.starts_with(&format!("#define {name} ")))
            .unwrap_or_else(|| panic!("{name} in the features golden"));
        let hex = line.rsplit(' ').next().unwrap().trim_end_matches('u');
        out += &format!("pub const {name}: u32 = {hex};\n");
    }
    out
}

/// `EXPECT`, the five `bench <name> <hex>` hashes tinymind (C++) printed for
/// the fine-tuning heads' workload, in the golden's order.
fn head_expect() -> String {
    let golden = fs::read_to_string("../tests/golden/head/stdout.txt").unwrap();
    let hashes: Vec<String> = golden
        .lines()
        .filter_map(|l| l.strip_prefix("bench "))
        .map(|l| format!("0x{}", l.split(' ').nth(1).expect("bench <name> <hex>")))
        .collect();
    assert_eq!(hashes.len(), 5, "five bench lines in the head golden");
    format!("pub const EXPECT: [u32; 5] = [{}];\n", hashes.join(", "))
}

/// `static COS` and `static SIN` for length `n`, from the golden's `twiddles
/// n=<n> k=<k> cos=<list> sin=<list>` lines, and `FORWARD_HASH`, the hash
/// tinymind (C++) printed for the forward transform of the random signal, and
/// `RFORWARD_HASH`, the one it printed for the real FFT of its real part.
fn fft_twiddles(n: usize) -> String {
    let golden = fs::read_to_string("../tests/golden/fft/stdout.txt").unwrap();
    let prefix = format!("twiddles n={n} ");
    let (mut cos, mut sin) = (Vec::new(), Vec::new());
    for line in golden.lines().filter(|l| l.starts_with(&prefix)) {
        for field in line.split(' ') {
            if let Some(v) = field.strip_prefix("cos=") {
                cos.extend(v.split(',').map(|x| x.parse::<i16>().unwrap()));
            } else if let Some(v) = field.strip_prefix("sin=") {
                sin.extend(v.split(',').map(|x| x.parse::<i16>().unwrap()));
            }
        }
    }
    assert!(
        cos.len() == n / 2 && sin.len() == n / 2,
        "twiddles for n={n} in the golden"
    );
    let list = |v: &[i16]| v.iter().map(i16::to_string).collect::<Vec<_>>().join(", ");
    // The forward transform's hash on the sweep's random signal.
    let random = format!("n={n} built signal=random ");
    let forward = golden
        .lines()
        .find(|l| l.starts_with(&random))
        .and_then(|l| l.split(' ').find_map(|f| f.strip_prefix("forward=")))
        .expect("the random signal's forward hash in the golden");
    // The real FFT's hash on the same signal's real part.
    let rrandom = format!("n={n} built rfft signal=random ");
    let rforward = golden
        .lines()
        .find(|l| l.starts_with(&rrandom))
        .and_then(|l| l.split(' ').find_map(|f| f.strip_prefix("forward=")))
        .expect("the random signal's real FFT hash in the golden");
    format!(
        "pub static COS: [i16; {h}] = [{}];\npub static SIN: [i16; {h}] = [{}];\n\
         pub const FORWARD_HASH: u32 = {forward};\npub const RFORWARD_HASH: u32 = {rforward};\n",
        list(&cos),
        list(&sin),
        h = n / 2
    )
}
