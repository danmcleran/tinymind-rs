// QDecisionTree and QGbdt against tinymind's QDecisionTree and QGBDT over
// random ensembles. tests/golden/tree/stdout.txt is the output of
// scripts/cpp/tree_sweep.cpp, which builds the same pools from the same
// generator and runs the same inputs; scripts/compare-with-cpp.sh rebuilds it
// and checks the golden is current. See that program for what is covered.

use std::fmt::Write;

use tinymind::tree::{QDecisionTree, QGbdt, QTreeNode};

const NODES: usize = 512;
const TREES: usize = 8;
const CLASSES: usize = 3;
const FEATURES: usize = 6;
const INPUTS: usize = 64;
const MODELS: usize = 24;
const MAX_DEPTH: u32 = 5;

struct Lcg(u32);

impl Lcg {
    fn draw(&mut self, n: u32) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (self.0 >> 8) % n
    }
}

/// A threshold and input type with the C++ program's range.
trait Value: Copy + Into<i32> {
    const LO: i32;
    const SPAN: u32;
    fn from_i32(v: i32) -> Self;
}

macro_rules! value {
    ($t:ty) => {
        impl Value for $t {
            const LO: i32 = <$t>::MIN as i32;
            const SPAN: u32 = 1 << <$t>::BITS;
            fn from_i32(v: i32) -> Self {
                v as $t
            }
        }
    };
}
value!(i8);
value!(u8);
value!(i16);

fn rand_value<T: Value>(r: &mut Lcg) -> T {
    T::from_i32(r.draw(T::SPAN) as i32 + T::LO)
}

fn build<T: Value>(r: &mut Lcg, pool: &mut [QTreeNode<T>], count: &mut usize, depth: u32) -> i16 {
    let idx = *count;
    *count += 1;
    // `||` short-circuits as in the C++: no draw at the maximum depth.
    if depth == MAX_DEPTH || r.draw(3) == 0 {
        let leaf_value = if r.draw(4) == 0 {
            r.draw(2_000_001) as i32 - 1_000_000
        } else {
            r.draw(21) as i32 - 10
        };
        pool[idx] = QTreeNode {
            feature: -1,
            threshold: T::from_i32(0),
            left: -1,
            right: -1,
            leaf_value,
        };
        return idx as i16;
    }
    let feature = r.draw(FEATURES as u32) as i16;
    let threshold = rand_value::<T>(r);
    let left = build(r, pool, count, depth + 1);
    let right = build(r, pool, count, depth + 1);
    pool[idx] = QTreeNode {
        feature,
        threshold,
        left,
        right,
        leaf_value: 0,
    };
    idx as i16
}

struct Fnv(u32);

impl Fnv {
    fn add(&mut self, v: i32) {
        for b in v.to_le_bytes() {
            self.0 ^= u32::from(b);
            self.0 = self.0.wrapping_mul(0x0100_0193);
        }
    }
}

fn run<T: Value>(name: &str, seed: u32, out: &mut String) {
    let mut r = Lcg(seed);
    let leaf0 = QTreeNode {
        feature: -1,
        threshold: T::from_i32(0),
        left: -1,
        right: -1,
        leaf_value: 0,
    };
    for m in 0..MODELS {
        let mut pool = [leaf0; NODES];
        let mut roots = [0i16; TREES];
        let mut classes = [0i16; TREES];
        let mut count = 0;
        for t in 0..TREES {
            roots[t] = build(&mut r, &mut pool, &mut count, 0);
            classes[t] = r.draw(CLASSES as u32) as i16;
        }
        let has_base = r.draw(2) == 0;
        let base: [i32; CLASSES] = core::array::from_fn(|_| r.draw(41) as i32 - 20);

        let gbdt = QGbdt::<'_, T, TREES, CLASSES, NODES, FEATURES> {
            nodes: &pool,
            tree_root: &roots,
            tree_class: &classes,
            base_score: has_base.then_some(&base),
        };
        assert!(gbdt.is_valid());
        let tree = QDecisionTree::<'_, T, NODES, FEATURES>::new(&pool);

        let mut hash = Fnv(0x811C_9DC5);
        let hi = T::LO + T::SPAN as i32 - 1;
        for _ in 0..INPUTS {
            let mut x = [T::from_i32(0); FEATURES];
            for v in &mut x {
                *v = if r.draw(2) == 0 {
                    let n = pool[r.draw(count as u32) as usize];
                    let near = if n.feature >= 0 {
                        n.threshold.into()
                    } else {
                        rand_value::<T>(&mut r).into()
                    } + r.draw(3) as i32
                        - 1;
                    T::from_i32(near.clamp(T::LO, hi))
                } else {
                    rand_value(&mut r)
                };
            }
            for s in gbdt.forward(&x) {
                hash.add(s);
            }
            hash.add(gbdt.predict(&x) as i32);
            hash.add(tree.predict(&x));
        }
        writeln!(
            out,
            "{name} model {m:2}: nodes={count:3} base={} hash=0x{:08X}",
            u8::from(has_base),
            hash.0
        )
        .unwrap();
    }
}

#[test]
fn trees_match_cpp_sweep() {
    let mut out = String::new();
    run::<i8>("i8", 1, &mut out);
    run::<u8>("u8", 2, &mut out);
    run::<i16>("i16", 3, &mut out);
    let golden = include_str!("golden/tree/stdout.txt");
    for (i, (got, want)) in out.lines().zip(golden.lines()).enumerate() {
        assert_eq!(got, want, "line {}", i + 1);
    }
    assert_eq!(out, golden);
}
