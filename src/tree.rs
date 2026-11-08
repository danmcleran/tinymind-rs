//! Quantized decision trees and gradient-boosted tree ensembles (C++
//! `cpp/qtree.hpp`).
//!
//! A tree walk is a handful of integer compares and branches: no
//! multiply-accumulates, no activation tables. Each split compares one input
//! feature against a threshold on the same integer grid as the feature. An affine
//! map is monotone, so `x_q <= t_q` makes the same decision as `x <= t` in the
//! real domain, except for an input within one grid step of the threshold. Leaf
//! values are `i32`, so an ensemble can add them up into per-class scores.
//!
//! - [`QTreeNode`]: one node of a flat node array (a *pool*), with integer
//!   indices to its children.
//! - [`QDecisionTree`]: one tree, rooted at node 0; returns its leaf's value.
//! - [`QGbdt`]: an additive ensemble whose trees share one pool. Each tree adds
//!   its leaf value to one class's score; the prediction is the class with the
//!   largest score. A single regression output is the case of one class.
//!
//! Both borrow their arrays, so a model can be `static` or `const` data in
//! flash, and both are generic over the threshold type `T` and the input type
//! `I` (`i8`, `u8`, `i16`, anything that converts losslessly to `i32`); the
//! compare is made in `i32`, as in the C++.
//!
//! # Malformed models
//!
//! The C++ reads out of bounds for a child, root or feature index outside the
//! model (undefined behaviour), and checks indices at compile time only for its
//! compile-time models. Here every function is total instead: a walk that meets
//! an index outside the pool or the input stops and contributes 0, and a tree
//! whose class index is out of range contributes nothing. A cyclic pool ends
//! after as many steps as the pool has nodes, as in the C++. Well-formed models
//! give the C++ results exactly; check a model with [`QDecisionTree::is_valid`]
//! or [`QGbdt::is_valid`], which are `const fn`, so a model in a `const` can be
//! checked at compile time:
//!
//! ```
//! use tinymind::tree::{QDecisionTree, QTreeNode};
//!
//! const fn split(feature: i16, threshold: i8, left: i16, right: i16) -> QTreeNode<i8> {
//!     QTreeNode { feature, threshold, left, right, leaf_value: 0 }
//! }
//! const fn leaf(value: i32) -> QTreeNode<i8> {
//!     QTreeNode { feature: -1, threshold: 0, left: -1, right: -1, leaf_value: value }
//! }
//!
//! // x[0] <= 10 ? 1 : (x[1] <= -5 ? 2 : 3)
//! static NODES: [QTreeNode<i8>; 5] = [split(0, 10, 1, 2), leaf(1), split(1, -5, 3, 4), leaf(2), leaf(3)];
//! const TREE: QDecisionTree<'static, i8, 5, 2> = QDecisionTree::new(&NODES);
//! const _: () = assert!(TREE.is_valid());
//!
//! assert_eq!(TREE.predict(&[10i8, 0]), 1);
//! assert_eq!(TREE.predict(&[11i8, -5]), 2);
//! assert_eq!(TREE.predict(&[11i8, -4]), 3);
//! ```
//!
//! Not ported: the C++ compile-time models (`QStaticDecisionTree`,
//! `QStaticGBDT`) and their evaluation policies. Rust cannot run a pool through
//! the type system the way C++ templates do. The repository's tree importer
//! (`tools/import_trees`, for scikit-learn, XGBoost and LightGBM models) writes
//! a module with both forms instead: a pool walked by these types, and the same
//! trees as nested `if`s (the inline policy).

/// One node of a quantized decision tree (C++ `QTreeNode`).
///
/// `feature >= 0` marks a split: go to `left` if `x[feature] <= threshold`,
/// else to `right` (indices into the same node array). `feature < 0` marks a
/// leaf, whose prediction is `leaf_value`; a split's `leaf_value` and a leaf's
/// other fields are not read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QTreeNode<T> {
    /// Input feature this node splits on; negative for a leaf.
    pub feature: i16,
    /// Split threshold, on the feature's integer grid.
    pub threshold: T,
    /// Child for `x[feature] <= threshold`.
    pub left: i16,
    /// Child for `x[feature] > threshold`.
    pub right: i16,
    /// The leaf's value.
    pub leaf_value: i32,
}

/// The node at index `n`, or `None` outside the pool.
fn node<T>(nodes: &[QTreeNode<T>], n: i16) -> Option<&QTreeNode<T>> {
    usize::try_from(n).ok().and_then(|i| nodes.get(i))
}

/// Walk the tree rooted at `root` in the pool `nodes` to a leaf and return its
/// value (C++ `walkQTree`). At most `max_nodes` splits are followed, so a cyclic
/// pool ends; the node reached then gives its `leaf_value`, as in the C++. An
/// index outside `nodes` or `x` ends the walk with 0 (see the module docs).
pub fn walk_tree<I, T>(nodes: &[QTreeNode<T>], root: i16, x: &[I], max_nodes: usize) -> i32
where
    I: Copy + Into<i32>,
    T: Copy + Into<i32>,
{
    let mut n = root;
    for _ in 0..max_nodes {
        let Some(node) = node(nodes, n) else {
            return 0;
        };
        if node.feature < 0 {
            return node.leaf_value;
        }
        let Some(&xv) = x.get(node.feature as usize) else {
            return 0;
        };
        n = if xv.into() <= node.threshold.into() {
            node.left
        } else {
            node.right
        };
    }
    node(nodes, n).map_or(0, |node| node.leaf_value)
}

/// Whether every split in `nodes` reads a feature below `features` and points
/// at children inside the pool (C++ `qtreePoolIsValid`).
const fn pool_is_valid<T>(nodes: &[QTreeNode<T>], features: usize) -> bool {
    let len = nodes.len();
    let mut i = 0;
    while i < len {
        let n = &nodes[i];
        if n.feature >= 0
            && (n.feature as usize >= features
                || n.left < 0
                || n.left as usize >= len
                || n.right < 0
                || n.right as usize >= len)
        {
            return false;
        }
        i += 1;
    }
    true
}

/// A single quantized decision tree over `FEATURES` inputs, with `NODES` nodes;
/// `nodes[0]` is the root (C++ `QDecisionTree`).
#[derive(Clone, Copy, Debug)]
pub struct QDecisionTree<'a, T, const NODES: usize, const FEATURES: usize> {
    /// The node pool.
    pub nodes: &'a [QTreeNode<T>; NODES],
}

impl<'a, T, const NODES: usize, const FEATURES: usize> QDecisionTree<'a, T, NODES, FEATURES> {
    /// A tree over the pool `nodes`.
    pub const fn new(nodes: &'a [QTreeNode<T>; NODES]) -> Self {
        Self { nodes }
    }

    /// Whether every split reads a feature below `FEATURES` and points at
    /// children inside the pool. A cycle is not detected; it ends as described
    /// in the module docs.
    pub const fn is_valid(&self) -> bool {
        pool_is_valid(self.nodes, FEATURES)
    }

    /// The value of the leaf that `x` reaches.
    pub fn predict<I: Copy + Into<i32>>(&self, x: &[I; FEATURES]) -> i32
    where
        T: Copy + Into<i32>,
    {
        const { assert!(NODES > 0 && FEATURES > 0, "a tree needs nodes and features") };
        walk_tree(self.nodes, 0, x, NODES)
    }
}

/// A gradient-boosted ensemble of `TREES` trees sharing one pool of `NODES`
/// nodes, scoring `CLASSES` classes from `FEATURES` inputs (C++ `QGBDT`).
///
/// Tree `t` starts at `nodes[tree_root[t]]` and adds its leaf value to the
/// score of class `tree_class[t]`. Scores start at `base_score` (zero when
/// `None`).
#[derive(Clone, Copy, Debug)]
pub struct QGbdt<
    'a,
    T,
    const TREES: usize,
    const CLASSES: usize,
    const NODES: usize,
    const FEATURES: usize,
> {
    /// The shared node pool.
    pub nodes: &'a [QTreeNode<T>; NODES],
    /// Each tree's root index into `nodes`.
    pub tree_root: &'a [i16; TREES],
    /// The class each tree scores, in `0..CLASSES` (0 for a regression).
    pub tree_class: &'a [i16; TREES],
    /// Each class's starting score, or `None` for zeros.
    pub base_score: Option<&'a [i32; CLASSES]>,
}

impl<T, const TREES: usize, const CLASSES: usize, const NODES: usize, const FEATURES: usize>
    QGbdt<'_, T, TREES, CLASSES, NODES, FEATURES>
{
    /// Whether every split reads a feature below `FEATURES` and points at
    /// children inside the pool, every root is inside the pool, and every class
    /// index is below `CLASSES`. A cycle is not detected; it ends as described
    /// in the module docs.
    pub const fn is_valid(&self) -> bool {
        let mut t = 0;
        while t < TREES {
            let root = self.tree_root[t];
            let class = self.tree_class[t];
            if root < 0 || root as usize >= NODES || class < 0 || class as usize >= CLASSES {
                return false;
            }
            t += 1;
        }
        pool_is_valid(self.nodes, FEATURES)
    }

    /// Every class's score: its base score plus the leaf values of its trees,
    /// added in tree order (wrapping on `i32` overflow, which the C++ leaves
    /// undefined).
    pub fn forward<I: Copy + Into<i32>>(&self, x: &[I; FEATURES]) -> [i32; CLASSES]
    where
        T: Copy + Into<i32>,
    {
        const {
            assert!(
                TREES > 0 && CLASSES > 0 && NODES > 0 && FEATURES > 0,
                "an ensemble needs trees, classes, nodes and features"
            )
        };
        let mut scores = self.base_score.copied().unwrap_or([0; CLASSES]);
        for (&root, &class) in self.tree_root.iter().zip(self.tree_class) {
            let leaf = walk_tree(self.nodes, root, x, NODES);
            if let Some(s) = usize::try_from(class).ok().and_then(|c| scores.get_mut(c)) {
                *s = s.wrapping_add(leaf);
            }
        }
        scores
    }

    /// The class with the largest score; the lowest index on a tie.
    pub fn predict<I: Copy + Into<i32>>(&self, x: &[I; FEATURES]) -> usize
    where
        T: Copy + Into<i32>,
    {
        let scores = self.forward(x);
        // The running best is carried by value, not re-read by index: panic
        // freedom must not depend on the compiler proving `best` in range.
        let mut best = (0, scores.first().copied().unwrap_or(0));
        for (c, &s) in scores.iter().enumerate().skip(1) {
            if s > best.1 {
                best = (c, s);
            }
        }
        best.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const fn split(feature: i16, threshold: i8, left: i16, right: i16) -> QTreeNode<i8> {
        QTreeNode {
            feature,
            threshold,
            left,
            right,
            leaf_value: 0,
        }
    }

    const fn leaf(value: i32) -> QTreeNode<i8> {
        QTreeNode {
            feature: -1,
            threshold: 0,
            left: -1,
            right: -1,
            leaf_value: value,
        }
    }

    #[test]
    fn walk_follows_splits_inclusively() {
        let nodes = [split(0, 0, 1, 2), leaf(-7), leaf(9)];
        assert_eq!(walk_tree(&nodes, 0, &[0i8], 3), -7);
        assert_eq!(walk_tree(&nodes, 0, &[1i8], 3), 9);
        assert_eq!(walk_tree(&nodes, 0, &[-128i8], 3), -7);
        // A root that is a leaf.
        assert_eq!(walk_tree(&nodes, 2, &[0i8], 3), 9);
        // Mixed input and threshold types compare in i32.
        assert_eq!(walk_tree(&nodes, 0, &[200u8], 3), 9);
    }

    #[test]
    fn malformed_pools_are_total() {
        // Child, root and feature out of range give 0.
        let bad_child = [split(0, 0, 5, -1), leaf(1)];
        assert_eq!(walk_tree(&bad_child, 0, &[0i8], 2), 0);
        assert_eq!(walk_tree(&bad_child, 0, &[1i8], 2), 0);
        assert_eq!(walk_tree(&bad_child, -1, &[0i8], 2), 0);
        assert_eq!(walk_tree(&bad_child, 9, &[0i8], 2), 0);
        let bad_feature = [split(3, 0, 1, 1), leaf(1)];
        assert_eq!(walk_tree(&bad_feature, 0, &[0i8], 2), 0);
        // A cycle stops after max_nodes steps and reads that node's leaf_value
        // field, as the C++ does.
        let mut cyclic = [split(0, 0, 1, 1), split(0, 0, 0, 0)];
        cyclic[0].leaf_value = 4;
        cyclic[1].leaf_value = 5;
        assert_eq!(walk_tree(&cyclic, 0, &[0i8], 2), 4);
        assert_eq!(walk_tree(&cyclic, 0, &[0i8], 3), 5);
        assert_eq!(walk_tree(&cyclic, 0, &[0i8], 0), 4);
    }

    static ENSEMBLE: [QTreeNode<i8>; 6] = [
        split(0, 0, 1, 2),
        leaf(10),
        leaf(-10),
        split(1, 5, 4, 5),
        leaf(3),
        leaf(30),
    ];
    const GBDT: QGbdt<'static, i8, 2, 2, 6, 2> = QGbdt {
        nodes: &ENSEMBLE,
        tree_root: &[0, 3],
        tree_class: &[0, 1],
        base_score: Some(&[1, -1]),
    };
    const _: () = assert!(GBDT.is_valid());

    #[test]
    fn ensemble_adds_leaves_per_class() {
        assert_eq!(GBDT.forward(&[0i8, 5]), [11, 2]);
        assert_eq!(GBDT.predict(&[0i8, 5]), 0);
        assert_eq!(GBDT.forward(&[1i8, 6]), [-9, 29]);
        assert_eq!(GBDT.predict(&[1i8, 6]), 1);
        // A tie goes to the lowest class.
        let tie = QGbdt::<'_, i8, 2, 2, 6, 2> {
            base_score: Some(&[0, 7]),
            ..GBDT
        };
        assert_eq!(tie.forward(&[0i8, 5]), [10, 10]);
        assert_eq!(tie.predict(&[0i8, 5]), 0);
        // No base score starts at zero; sums wrap.
        let wrap = QGbdt::<'_, i8, 2, 2, 6, 2> {
            base_score: Some(&[i32::MAX, 0]),
            ..GBDT
        };
        assert_eq!(wrap.forward(&[0i8, 5])[0], i32::MIN + 9);
        let none = QGbdt::<'_, i8, 2, 2, 6, 2> {
            base_score: None,
            ..GBDT
        };
        assert_eq!(none.forward(&[0i8, 5]), [10, 3]);
    }

    #[test]
    fn validity() {
        let nodes = [split(1, 0, 1, 2), leaf(0), leaf(0)];
        assert!(QDecisionTree::<'_, i8, 3, 2>::new(&nodes).is_valid());
        assert!(!QDecisionTree::<'_, i8, 3, 1>::new(&nodes).is_valid());
        let child = [split(0, 0, 1, 3), leaf(0), leaf(0)];
        assert!(!QDecisionTree::<'_, i8, 3, 1>::new(&child).is_valid());
        let negative = [split(0, 0, -1, 2), leaf(0), leaf(0)];
        assert!(!QDecisionTree::<'_, i8, 3, 1>::new(&negative).is_valid());
        let bad_class = QGbdt::<'_, i8, 2, 2, 6, 2> {
            tree_class: &[0, 2],
            ..GBDT
        };
        assert!(!bad_class.is_valid());
        // An invalid class index is skipped.
        assert_eq!(bad_class.forward(&[0i8, 5]), [11, -1]);
        let bad_root = QGbdt::<'_, i8, 2, 2, 6, 2> {
            tree_root: &[0, 6],
            ..GBDT
        };
        assert!(!bad_root.is_valid());
        assert_eq!(bad_root.forward(&[0i8, 5]), [11, -1]);
    }
}
