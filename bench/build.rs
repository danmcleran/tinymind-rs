//! Put the selected board's memory.x where cortex-m-rt's link.x finds it, and
//! write the FFT workload's twiddle tables (tinymind (C++)'s, read from the FFT
//! sweep's golden, with the hash it expects) to fft_twiddles.rs, and the fine-tuning
//! heads' expected hashes (read from the head sweep's golden) to head_expect.rs.
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
/// tinymind (C++) printed for the forward transform of the random signal.
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
    format!(
        "pub static COS: [i16; {h}] = [{}];\npub static SIN: [i16; {h}] = [{}];\n\
         pub const FORWARD_HASH: u32 = {forward};\n",
        list(&cos),
        list(&sin),
        h = n / 2
    )
}
