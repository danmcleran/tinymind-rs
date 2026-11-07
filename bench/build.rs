//! Put the selected board's memory.x where cortex-m-rt's link.x finds it.
use std::{env, fs, path::PathBuf};

fn main() {
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let memory = match (
        env::var_os("CARGO_FEATURE_F411").is_some(),
        env::var_os("CARGO_FEATURE_L031").is_some(),
    ) {
        (true, false) => include_str!("memory-f411.x"),
        (false, true) => include_str!("memory-l031.x"),
        _ => panic!("select exactly one board feature: f411 or l031"),
    };
    fs::write(out.join("memory.x"), memory).unwrap();
    println!("cargo:rustc-link-search={}", out.display());
    println!("cargo:rerun-if-changed=memory-f411.x");
    println!("cargo:rerun-if-changed=memory-l031.x");
}
