//! C++ `std::mt19937` and libstdc++'s `std::uniform_real_distribution<float>`,
//! for examples whose C++ originals draw from them. Not part of the crate.
//!
//! `mt19937` is fully specified by the C++ standard. The distribution is not:
//! this is libstdc++'s (GCC 13), whose `generate_canonical<float, 24>` takes one
//! 32-bit draw, converts it to `float` (rounding to nearest), divides by 2^32,
//! and steps a result of 1.0 down to the largest `float` below it; the
//! distribution then returns `u * (b - a) + a` in `float`.

/// The 32-bit Mersenne Twister with the standard's parameters.
pub struct Mt19937 {
    state: [u32; 624],
    index: usize,
}

impl Mt19937 {
    pub fn new(seed: u32) -> Self {
        let mut state = [0u32; 624];
        state[0] = seed;
        for i in 1..624 {
            let prev = state[i - 1];
            state[i] = 1_812_433_253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        Mt19937 { state, index: 624 }
    }

    fn twist(&mut self) {
        for i in 0..624 {
            let y = (self.state[i] & 0x8000_0000) | (self.state[(i + 1) % 624] & 0x7FFF_FFFF);
            let mut next = self.state[(i + 397) % 624] ^ (y >> 1);
            if y & 1 != 0 {
                next ^= 0x9908_B0DF;
            }
            self.state[i] = next;
        }
        self.index = 0;
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            self.twist();
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9D2C_5680;
        y ^= (y << 15) & 0xEFC6_0000;
        y ^ (y >> 18)
    }
}

/// libstdc++'s `std::uniform_real_distribution<float>(a, b)(rng)`.
pub fn uniform_f32(rng: &mut Mt19937, a: f32, b: f32) -> f32 {
    let mut u = rng.next_u32() as f32 / 4_294_967_296.0f32;
    if u >= 1.0 {
        // std::nextafter(1.0f, 0.0f)
        u = f32::from_bits(1.0f32.to_bits() - 1);
    }
    u * (b - a) + a
}
