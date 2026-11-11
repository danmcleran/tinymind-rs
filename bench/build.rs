//! Put the selected board's memory.x where cortex-m-rt's link.x finds it.
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
}
