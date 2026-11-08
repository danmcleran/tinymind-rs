//! A bit-exact re-implementation of glibc's `srand()`/`rand()`.
//!
//! tinymind's examples (xor, iris, energy efficiency) draw their initial weights
//! from `rand()` seeded with 1, then re-seed it with 7 for the shuffle and the
//! training samples. Reproducing a C++ run bit-for-bit therefore needs the
//! identical random sequences. glibc's default generator is the TYPE_3
//! additive feedback generator: `r[i] = r[i-3] + r[i-31]` (mod 2^32), seeded by a
//! Park-Miller LCG, with the first 310 outputs discarded and the low bit dropped.
//!
//! This is not part of tinymind itself; it exists so the examples can be compared
//! against the C++ builds on Linux/glibc.

/// glibc `rand()` state. `next()` returns values in `0..=0x7FFF_FFFF`, like `rand()`.
#[derive(Clone, Debug)]
pub struct GlibcRand {
    /// Ring buffer holding the last 34 words of the `r[]` sequence.
    ring: [u32; 34],
    /// Index of the next word to produce, counted from the start of `r[]`.
    pos: usize,
}

impl GlibcRand {
    /// Equivalent to `srand(seed)`.
    pub fn new(seed: u32) -> Self {
        let mut r = [0i32; 34];
        // glibc maps seed 0 to 1.
        r[0] = if seed == 0 { 1 } else { seed as i32 };
        for i in 1..31 {
            // Park-Miller "minimal standard" LCG, 16807 * r mod (2^31 - 1), using
            // Schrage's method so the product never overflows 32 bits.
            let hi = r[i - 1] / 127_773;
            let lo = r[i - 1] % 127_773;
            let mut word = 16_807 * lo - 2_836 * hi;
            if word < 0 {
                word += 2_147_483_647;
            }
            r[i] = word;
        }
        for i in 31..34 {
            r[i] = r[i - 31];
        }

        let mut rng = Self {
            ring: r.map(|w| w as u32),
            pos: 34,
        };
        // glibc discards the first 310 outputs (10 * the degree, 31).
        for _ in 0..310 {
            rng.step();
        }
        rng
    }

    /// Equivalent to `rand()`.
    pub fn next_i32(&mut self) -> i32 {
        (self.step() >> 1) as i32
    }

    fn step(&mut self) -> u32 {
        let i = self.pos;
        let word = self.ring[(i - 31) % 34].wrapping_add(self.ring[(i - 3) % 34]);
        self.ring[i % 34] = word;
        self.pos += 1;
        word
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_glibc_seed_1() {
        // The well-known first outputs of glibc rand() with the default seed.
        let mut rng = GlibcRand::new(1);
        let expected = [
            1_804_289_383,
            846_930_886,
            1_681_692_777,
            1_714_636_915,
            1_957_747_793,
        ];
        for e in expected {
            assert_eq!(rng.next_i32(), e);
        }
    }
}
