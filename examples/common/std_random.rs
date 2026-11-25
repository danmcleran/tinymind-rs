//! C++ `std::mt19937`, libstdc++'s `std::uniform_real_distribution<float>` and
//! `std::normal_distribution<double>`,
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

/// libstdc++'s `std::generate_canonical<double, 53>(rng)` for `mt19937`: two
/// 32-bit draws, `(lo + hi * 2^32) / 2^64` in `double` (the sum rounded to 53
/// bits), with a result of 1.0 stepped down to the largest `double` below it.
#[allow(dead_code)]
pub fn canonical_f64(rng: &mut Mt19937) -> f64 {
    let lo = f64::from(rng.next_u32());
    let hi = f64::from(rng.next_u32());
    let u = (lo + hi * 4_294_967_296.0) / 18_446_744_073_709_551_616.0;
    if u >= 1.0 {
        // std::nextafter(1.0, 0.0)
        f64::from_bits(1.0f64.to_bits() - 1)
    } else {
        u
    }
}

/// libstdc++'s `std::normal_distribution<double>(mean, stddev)`: Marsaglia's
/// polar method on [`canonical_f64`] draws. Each accepted pair gives two
/// values; the second is kept for the next call, as libstdc++ keeps it.
#[allow(dead_code)]
pub struct NormalF64 {
    mean: f64,
    stddev: f64,
    saved: Option<f64>,
}

#[allow(dead_code)]
impl NormalF64 {
    pub fn new(mean: f64, stddev: f64) -> Self {
        NormalF64 {
            mean,
            stddev,
            saved: None,
        }
    }

    pub fn sample(&mut self, rng: &mut Mt19937) -> f64 {
        let value = match self.saved.take() {
            Some(v) => v,
            None => {
                let (x, y, r2) = loop {
                    let x = 2.0 * canonical_f64(rng) - 1.0;
                    let y = 2.0 * canonical_f64(rng) - 1.0;
                    let r2 = x * x + y * y;
                    if r2 <= 1.0 && r2 != 0.0 {
                        break (x, y, r2);
                    }
                };
                let mult = (-2.0 * r2.ln() / r2).sqrt();
                self.saved = Some(x * mult);
                y * mult
            }
        };
        value * self.stddev + self.mean
    }
}
