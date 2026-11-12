// The quantized FFT sweep, shared by tests/fft_sweep.rs (on the host, against
// tinymind (C++)'s output, with build_fft_twiddles' tables) and
// bench/src/sweeps.rs (on Cortex-M, against the same golden, with the C++'s
// tables read from it). no_std and allocation-free. The C++ is
// scripts/cpp/fft_sweep.cpp.

use core::fmt::Write;

use tinymind::int8::QFft1D;

/// What the sweep needs from `int8::calibration` (std only): on the host the
/// real `build_fft_twiddles`, on the device the C++'s tables read from the
/// golden.
pub trait Twiddles {
    /// `build_fft_twiddles::<N, HALF>()`.
    fn twiddles<const N: usize, const HALF: usize>(&mut self) -> ([i16; HALF], [i16; HALF]);
}

struct Samples(u32);

impl Samples {
    fn next(&mut self) -> i16 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.0 >> 16) as u16 as i16
    }
}

const FNV_OFFSET: u32 = 0x811C_9DC5;

fn fnv(hash: u32, byte: u32) -> u32 {
    (hash ^ byte).wrapping_mul(0x0100_0193)
}

fn hash16(x: &[i16], mut hash: u32) -> u32 {
    for &v in x {
        let v = u32::from(v as u16);
        hash = fnv(fnv(hash, v & 0xFF), v >> 8);
    }
    hash
}

fn hash32(x: &[i32], mut hash: u32) -> u32 {
    for &v in x {
        for b in v.to_le_bytes() {
            hash = fnv(hash, u32::from(b));
        }
    }
    hash
}

/// `cos(-2πj/N)` (or the sine) for any `j`, from a half table: the second half
/// is the negated first.
fn full(table: &[i16], n: usize, j: usize) -> i32 {
    let m = j % n;
    if m < n / 2 {
        i32::from(table[m])
    } else {
        -i32::from(table[m - n / 2])
    }
}

const SIGNALS: [&str; 9] = [
    "impulse",
    "offset-impulse",
    "dc",
    "tone",
    "two-tones",
    "random",
    "rails",
    "min",
    "max-min",
];

fn make_signal<const N: usize>(
    s: usize,
    c: &[i16],
    sn: &[i16],
    rng: &mut Samples,
) -> ([i16; N], [i16; N]) {
    let mut re = [0i16; N];
    let mut im = [0i16; N];
    match s {
        0 => re[0] = 32767,
        1 => {
            re[3 % N] = -20000;
            im[3 % N] = 12345;
        }
        2 => {
            re = [1000; N];
            im = [-500; N];
        }
        3 => {
            for (i, r) in re.iter_mut().enumerate() {
                *r = ((full(c, N, i) * 20000) >> 15) as i16;
            }
        }
        4 => {
            for i in 0..N {
                let j = i * (3 * N / 8 + 1);
                re[i] = (((full(c, N, i) * 12000) >> 15) + ((full(c, N, j) * 9000) >> 15)) as i16;
                im[i] = ((full(sn, N, i) * 12000) >> 15) as i16;
            }
        }
        5 => {
            for i in 0..N {
                re[i] = rng.next();
                im[i] = rng.next();
            }
        }
        6 => {
            let rail = |v: i16| if v & 1 != 0 { 32767 } else { -32768 };
            for i in 0..N {
                re[i] = rail(rng.next());
                im[i] = rail(rng.next());
            }
        }
        7 => {
            re = [-32768; N];
            im = [-32768; N];
        }
        _ => {
            re = [32767; N];
            im = [-32768; N];
        }
    }
    (re, im)
}

fn run_signal<const N: usize, const HALF: usize>(
    out: &mut impl Write,
    fft: &QFft1D<'_, N, HALF>,
    label: &str,
    s: usize,
    (c, sn): (&[i16], &[i16]),
    rng: &mut Samples,
) {
    let (x, y) = make_signal::<N>(s, c, sn, rng);
    let h_input = hash16(&y, hash16(&x, FNV_OFFSET));
    let mut mag = [0i32; N];
    fft.magnitude_squared(&x, &y, &mut mag);
    let h_mag_in = hash32(&mag, FNV_OFFSET);

    let (mut re, mut im) = (x, y);
    fft.forward(&mut re, &mut im);
    let h_forward = hash16(&im, hash16(&re, FNV_OFFSET));
    fft.magnitude_squared(&re, &im, &mut mag);
    let h_mag = hash32(&mag, FNV_OFFSET);
    fft.inverse(&mut re, &mut im);
    let h_round_trip = hash16(&im, hash16(&re, FNV_OFFSET));

    let (mut re, mut im) = (x, y);
    fft.inverse(&mut re, &mut im);
    let h_inverse = hash16(&im, hash16(&re, FNV_OFFSET));

    let _ = writeln!(
        out,
        "n={N} {label} signal={} input=0x{h_input:08X} mag_in=0x{h_mag_in:08X} \
         forward=0x{h_forward:08X} mag=0x{h_mag:08X} roundtrip=0x{h_round_trip:08X} \
         inverse=0x{h_inverse:08X}",
        SIGNALS[s]
    );
}

fn write_list(out: &mut impl Write, values: &[i16]) {
    for (i, v) in values.iter().enumerate() {
        let _ = if i == 0 {
            write!(out, "{v}")
        } else {
            write!(out, ",{v}")
        };
    }
}

fn sweep<const N: usize, const HALF: usize>(out: &mut impl Write, tw: &mut impl Twiddles) {
    let (c, sn) = tw.twiddles::<N, HALF>();
    for (chunk, (cc, sc)) in c.chunks(16).zip(sn.chunks(16)).enumerate() {
        let k = chunk * 16;
        let _ = write!(out, "twiddles n={N} k={k} cos=");
        write_list(out, cc);
        let _ = write!(out, " sin=");
        write_list(out, sc);
        let _ = writeln!(out);
    }

    let mut rng = Samples(N as u32);
    let fft: QFft1D<'_, N, HALF> = QFft1D {
        twiddle_cos: &c,
        twiddle_sin: &sn,
    };
    for s in 0..SIGNALS.len() {
        run_signal(out, &fft, "built", s, (&c, &sn), &mut rng);
    }

    // Random twiddles, -32768 included; the test signals still use the built
    // tables.
    let mut rc = [0i16; HALF];
    let mut rs = [0i16; HALF];
    for (a, b) in rc.iter_mut().zip(rs.iter_mut()) {
        *a = rng.next();
        *b = rng.next();
    }
    rc[0] = -32768;
    rs[0] = -32768;
    let random: QFft1D<'_, N, HALF> = QFft1D {
        twiddle_cos: &rc,
        twiddle_sin: &rs,
    };
    for s in 0..SIGNALS.len() {
        run_signal(out, &random, "random-twiddles", s, (&c, &sn), &mut rng);
    }
}

/// Write the sweep's report (tests/golden/fft/stdout.txt).
pub fn report(out: &mut impl Write, tw: &mut impl Twiddles) {
    sweep::<2, 1>(out, tw);
    sweep::<4, 2>(out, tw);
    sweep::<8, 4>(out, tw);
    sweep::<16, 8>(out, tw);
    sweep::<32, 16>(out, tw);
    sweep::<64, 32>(out, tw);
    sweep::<128, 64>(out, tw);
    sweep::<256, 128>(out, tw);
    sweep::<512, 256>(out, tw);
    sweep::<1024, 512>(out, tw);
}
