//! Helpers shared by the golden tests. (`tests/common/mod.rs` rather than
//! `tests/common.rs`, so Cargo does not run it as a test crate of its own.)

/// Byte-compare `actual` with `expected`; on a mismatch, panic naming the first
/// differing line.
pub fn assert_same(name: &str, actual: &[u8], expected: &[u8]) {
    if actual == expected {
        return;
    }
    let a = String::from_utf8_lossy(actual);
    let e = String::from_utf8_lossy(expected);
    let mut al = a.split('\n');
    let mut el = e.split('\n');
    let mut n = 1;
    loop {
        match (al.next(), el.next()) {
            (Some(x), Some(y)) if x == y => n += 1,
            (x, y) => panic!("{name} differs at line {n}:\n  actual:   {x:?}\n  expected: {y:?}"),
        }
    }
}
