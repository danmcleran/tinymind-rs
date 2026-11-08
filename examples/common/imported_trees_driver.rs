//! The imported-trees example's program, shared by `examples/imported_trees.rs`
//! and `tests/imported_trees_golden.rs` (which runs it in-process against the
//! C++ golden). Two scikit-learn classifiers for the UCI Wine data, written by
//! `tools/import_trees` (`imported_trees_gbdt_model.rs`,
//! `imported_trees_forest_model.rs`), run on the held-out samples and on random
//! inputs. `scripts/cpp/imported_trees.cpp` runs the same models, imported by
//! tinymind's own importer, on tinymind's C++ runtime and prints the same.

use std::io::{self, Write};

#[path = "imported_trees_data.rs"]
mod data;
#[path = "imported_trees_forest_model.rs"]
mod forest;
#[path = "imported_trees_gbdt_model.rs"]
mod gbdt;

struct Lcg(u32);

impl Lcg {
    fn draw(&mut self, n: u32) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.0 >> 8) % n
    }
}

fn fnv(hash: &mut u32, v: i32) {
    for b in v.to_le_bytes() {
        *hash ^= u32::from(b);
        *hash = hash.wrapping_mul(0x0100_0193);
    }
}

fn list<T: std::fmt::Display>(v: &[T]) -> String {
    v.iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Run one generated module: both of its forms must agree everywhere.
macro_rules! run_model {
    ($out:expr, $name:literal, $m:ident, $ty:literal) => {{
        let out: &mut dyn Write = $out;
        writeln!(
            out,
            "model {}: {} trees, {} {} inputs, {} outputs",
            $name,
            $m::NUM_TREES,
            $m::NUM_FEATURES,
            $ty,
            $m::NUM_OUTPUTS
        )?;
        let mut correct = 0;
        for (s, (x, &label)) in data::TEST_X.iter().zip(&data::TEST_Y).enumerate() {
            let q = $m::quantize_features(x);
            let raw = $m::forward(&q);
            let class = $m::predict(&q);
            assert_eq!(
                raw,
                $m::forward_inline(&q),
                "{} sample {s}: forms differ",
                $name
            );
            assert_eq!(
                class,
                $m::predict_inline(&q),
                "{} sample {s}: forms differ",
                $name
            );
            correct += usize::from(class == label);
            writeln!(
                out,
                "sample {s:2}: q=[{}] raw=[{}] class={class} label={label}",
                list(&q),
                list(&raw)
            )?;
        }
        writeln!(out, "held-out accuracy: {correct}/{}", data::NUM_SAMPLES)?;
        // Random inputs over the whole input range.
        let mut rng = Lcg(1);
        let (lo, span) = ($m::InputType::MIN as i32, 1u32 << $m::InputType::BITS);
        let mut hash = 0x811C_9DC5u32;
        for _ in 0..256 {
            let mut q = [0; $m::NUM_FEATURES];
            for v in &mut q {
                *v = (lo + rng.draw(span) as i32) as $m::InputType;
            }
            let raw = $m::forward(&q);
            assert_eq!(raw, $m::forward_inline(&q), "{}: forms differ", $name);
            raw.iter().for_each(|&v| fnv(&mut hash, v));
            fnv(&mut hash, $m::predict(&q) as i32);
        }
        writeln!(out, "random inputs: 256, hash=0x{hash:08X}")?;
    }};
}

/// Run both models; returns whether every sample was handled.
pub fn run(out: &mut dyn Write) -> io::Result<()> {
    run_model!(out, "gbdt", gbdt, "int8");
    run_model!(out, "forest", forest, "int16");
    Ok(())
}
