// `GlibcRand` against the real glibc `rand()`. Only on Linux with glibc; the
// golden tests cover other hosts for the seeds the examples use.
//
// This calls C through FFI, so it needs `unsafe`. That is fine in a test crate:
// the library itself stays `#![forbid(unsafe_code)]`.
#![cfg(all(target_os = "linux", target_env = "gnu"))]

use tinymind::rand::GlibcRand;

unsafe extern "C" {
    fn srand(seed: u32);
    fn rand() -> i32;
}

/// Seeds that exercise the corner cases (0 maps to 1, and seeds of 2^31 and up
/// are not valid i32 values), plus a fixed-seed spread over the rest.
fn seeds() -> Vec<u32> {
    let mut seeds = vec![0, 1, 2, 7, 42, 127_773, 2_147_483_646, 2_147_483_647];
    seeds.extend([2_147_483_648, 3_000_000_000, 4_294_967_294, 4_294_967_295]);
    let mut x: u32 = 0x1234_5678;
    for _ in 0..200 {
        // xorshift32: any fixed spread will do.
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        seeds.push(x);
    }
    seeds
}

// One test function only: srand/rand share global state, and the test harness
// runs test functions in parallel threads.
#[test]
fn matches_glibc_rand() {
    for seed in seeds() {
        let mut ours = GlibcRand::new(seed);
        // SAFETY: srand and rand have no preconditions; nothing else in this
        // test binary uses the generator concurrently.
        unsafe { srand(seed) };
        for i in 0..1000 {
            // SAFETY: as above.
            let expected = unsafe { rand() };
            assert_eq!(ours.next_i32(), expected, "seed {seed}, draw {i}");
        }
    }
}
