//! The imported tree models (examples/common/imported_trees_*_model.rs) on the
//! board: both classify the held-out UCI Wine samples, with the hash of every
//! raw output and class checked against the host run, and both forms are timed.
//!
//! The node pools are immutable `static`s, which cortex-m-rt's linker script
//! places in `.rodata`, in flash: the walk reads them where they are. To show
//! what that costs, the gradient-boosted pool is also copied into SRAM and timed
//! there (F411 only: the 7.5 KB pool does not fit beside the stack in the
//! L031's 8 KB of SRAM).

use core::fmt::Write;
use core::hint::black_box;

use crate::board::{self, Uart};
use crate::{REPS, median};

#[path = "../../examples/common/imported_trees_data.rs"]
mod data;
#[path = "../../examples/common/imported_trees_forest_model.rs"]
mod forest;
#[path = "../../examples/common/imported_trees_gbdt_model.rs"]
mod gbdt;

/// The same loop on the host (x86-64, same modules): FNV-1a over every raw
/// output and class of both models on all 54 held-out samples.
const EXPECT_HASH: u32 = 0x4719_E9A0;

fn fnv(hash: &mut u32, v: i32) {
    for b in v.to_le_bytes() {
        *hash ^= u32::from(b);
        *hash = hash.wrapping_mul(0x0100_0193);
    }
}

/// Median cycles of `f` over `REPS` runs.
fn time(mut f: impl FnMut()) -> (u32, u32, u32) {
    let mut t = [0u32; REPS];
    for slot in t.iter_mut() {
        let a = board::cycles();
        f();
        *slot = board::cycles().wrapping_sub(a);
    }
    median(t)
}

fn report(out: &mut Uart, name: &str, (min, med, max): (u32, u32, u32)) {
    let _ = writeln!(out, "{name}_min={min}\n{name}_med={med}\n{name}_max={max}");
}

pub fn run(out: &mut Uart) {
    let _ = writeln!(out, "name=trees");

    // Every held-out sample through both models, both forms.
    let mut hash = 0x811C_9DC5u32;
    let (mut gbdt_ok, mut forest_ok, mut forms_agree) = (0, 0, true);
    for (x, &label) in data::TEST_X.iter().zip(&data::TEST_Y) {
        let q = gbdt::quantize_features(x);
        let raw = gbdt::forward(&q);
        forms_agree &= raw == gbdt::forward_inline(&q);
        raw.iter().for_each(|&v| fnv(&mut hash, v));
        let class = gbdt::predict(&q);
        fnv(&mut hash, class as i32);
        gbdt_ok += u32::from(class == label);

        let q = forest::quantize_features(x);
        let votes = forest::forward(&q);
        forms_agree &= votes == forest::forward_inline(&q);
        votes.iter().for_each(|&v| fnv(&mut hash, v));
        let class = forest::predict(&q);
        fnv(&mut hash, class as i32);
        forest_ok += u32::from(class == label);
    }
    let n = data::NUM_SAMPLES;
    let _ = writeln!(
        out,
        "gbdt_correct={gbdt_ok}/{n}\nforest_correct={forest_ok}/{n}"
    );
    let _ = writeln!(out, "forms_agree={}", u32::from(forms_agree));
    let _ = writeln!(
        out,
        "hash=0x{hash:08X}\nexpect_hash=0x{EXPECT_HASH:08X}\nhash_match={}",
        u32::from(hash == EXPECT_HASH)
    );

    // Where the pool is: flash is 0x0800_0000.., SRAM 0x2000_0000...
    let _ = writeln!(
        out,
        "gbdt_pool_addr=0x{:08X}",
        gbdt::MODEL.nodes.as_ptr() as u32
    );
    let _ = writeln!(
        out,
        "gbdt_pool_bytes={}",
        core::mem::size_of_val(gbdt::MODEL.nodes)
    );

    // One prediction, on the first held-out sample.
    let gq = black_box(gbdt::quantize_features(&data::TEST_X[0]));
    let fq = black_box(forest::quantize_features(&data::TEST_X[0]));
    report(
        out,
        "gbdt_pool_flash",
        time(|| {
            black_box(gbdt::predict(black_box(&gq)));
        }),
    );
    report(
        out,
        "gbdt_inline",
        time(|| {
            black_box(gbdt::predict_inline(black_box(&gq)));
        }),
    );
    report(
        out,
        "forest_pool_flash",
        time(|| {
            black_box(forest::predict(black_box(&fq)));
        }),
    );
    report(
        out,
        "forest_inline",
        time(|| {
            black_box(forest::predict_inline(black_box(&fq)));
        }),
    );
    report(
        out,
        "quantize",
        time(|| {
            black_box(gbdt::quantize_features(black_box(&data::TEST_X[0])));
        }),
    );

    // Both boards: the trees that lie in the first 256 nodes (3 KB, which fits
    // in the L031's SRAM), walked from flash and from an SRAM copy.
    const SUBSET: usize = 256;
    let roots = gbdt::MODEL.tree_root;
    let trees = roots
        .iter()
        .zip(
            roots
                .iter()
                .skip(1)
                .chain(core::iter::once(&(gbdt::MODEL.nodes.len() as i16))),
        )
        .take_while(|&(_, &end)| end as usize <= SUBSET)
        .count();
    let flash: &[tinymind::tree::QTreeNode<i8>] = &gbdt::MODEL.nodes[..SUBSET];
    let mut copy = [flash[0]; SUBSET];
    copy.copy_from_slice(flash);
    let sram: &[tinymind::tree::QTreeNode<i8>] = black_box(&copy);
    let walk = |pool: &[tinymind::tree::QTreeNode<i8>]| {
        roots[..trees].iter().fold(0i32, |acc, &r| {
            acc.wrapping_add(tinymind::tree::walk_tree(pool, r, &gq, SUBSET))
        })
    };
    let _ = writeln!(
        out,
        "subset_trees={trees}
subset_nodes={SUBSET}"
    );
    let _ = writeln!(
        out,
        "subset_matches={}",
        u32::from(walk(flash) == walk(sram))
    );
    let _ = writeln!(out, "subset_sram_addr=0x{:08X}", sram.as_ptr() as u32);
    report(
        out,
        "subset_flash",
        time(|| {
            black_box(walk(black_box(flash)));
        }),
    );
    report(
        out,
        "subset_sram",
        time(|| {
            black_box(walk(black_box(sram)));
        }),
    );

    #[cfg(feature = "f411")]
    {
        // The same ensemble over a copy of its arrays on the stack, in SRAM.
        let nodes = black_box(*gbdt::MODEL.nodes);
        let roots = black_box(*gbdt::MODEL.tree_root);
        let classes = black_box(*gbdt::MODEL.tree_class);
        let base = black_box(gbdt::BASE_SCORE);
        let ram = tinymind::tree::QGbdt {
            nodes: &nodes,
            tree_root: &roots,
            tree_class: &classes,
            base_score: Some(&base),
        };
        let _ = writeln!(out, "gbdt_pool_ram_addr=0x{:08X}", nodes.as_ptr() as u32);
        let _ = writeln!(
            out,
            "ram_matches={}",
            u32::from(ram.forward(&gq) == gbdt::forward(&gq))
        );
        report(
            out,
            "gbdt_pool_ram",
            time(|| {
                black_box(ram.predict(black_box(&gq)));
            }),
        );
    }
}
