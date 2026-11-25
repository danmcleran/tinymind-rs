//! Anomaly scores for int8 feature vectors, ported from tinymind (C++)
//! `cpp/qanomaly.hpp`: how far a sample sits from the "normal" data a model
//! was fitted to.
//!
//! - [`QKMeansAnomaly`]: the distance to the nearest of `K` centroids, in units
//!   of that cluster's mean squared radius.
//! - [`QKMeansAnomalyTrainer`]: learns those centroids, the feature weights and
//!   the radii on the device, one sample at a time, with no float and no
//!   buffered data.
//! - [`QDiagGaussianAnomaly`]: the negative log-likelihood under a mixture of
//!   `K` diagonal Gaussians (hard assignment: the best component), fitted on
//!   the host and built by
//!   [`calibration::build_diag_gaussian_anomaly`](super::calibration::build_diag_gaussian_anomaly)
//!   (`std` feature).
//!
//! The features are any int8 vector on a fixed affine grid: log-mel bands
//! from [`QLogMelSpectrogram`](super::QLogMelSpectrogram) requantized to int8,
//! an int8 network's embedding, or quantized sensor readings. A higher score
//! is more anomalous; a deployment compares it with a threshold chosen on
//! normal data (for example the largest score seen on held-out normal samples
//! times a margin). Integer arithmetic only.

/// Checks the C++ `static_assert`s: at least one feature and one cluster, and
/// few enough features that a weighted distance fits 64 bits.
const fn check<const F: usize, const K: usize>() {
    assert!(F > 0, "at least one feature");
    assert!(K > 0, "at least one cluster or component");
    assert!(F <= 65536, "too many features for a 64-bit distance");
}

/// `sum_f w_f (a_f - b_f)^2`, or the unweighted sum when `w` is `None`.
fn weighted_squared_distance<const F: usize>(
    a: &[i8; F],
    b: &[i8; F],
    w: Option<&[u16; F]>,
) -> u64 {
    match w {
        None => {
            // At most 65,536 features of 255^2: below 2^32.
            let mut plain = 0u32;
            for (&x, &c) in a.iter().zip(b) {
                let d = i32::from(x) - i32::from(c);
                plain = plain.wrapping_add((d * d) as u32);
            }
            u64::from(plain)
        }
        Some(w) => {
            let mut sum = 0u64;
            for ((&x, &c), &wf) in a.iter().zip(b).zip(w) {
                let d = i32::from(x) - i32::from(c);
                sum = sum.wrapping_add(u64::from((d * d) as u32) * u64::from(wf));
            }
            sum
        }
    }
}

/// `(d2 * inv) >> 24` without overflowing 64 bits: exact while `d2` fits 32
/// bits, and to 16 bits of `d2` beyond that, where the score is huge.
pub(crate) fn normalize_distance(d2: u64, inv: u32) -> u64 {
    if d2 <= u64::from(u32::MAX) {
        (d2 * u64::from(inv)) >> 24
    } else {
        ((d2 >> 16).wrapping_mul(u64::from(inv))) >> 8
    }
}

/// `round(a / b)` for `b > 0`, ties away from zero. A divisor that is not
/// positive (never passed) gives 0 instead of panicking.
pub(crate) fn rounded_divide(a: i32, b: i32) -> i32 {
    if b <= 0 {
        return 0;
    }
    if a >= 0 {
        a.saturating_add(b / 2) / b
    } else {
        -(a.saturating_neg().saturating_add(b / 2) / b)
    }
}

/// K-means anomaly score over `F` int8 features and `K` clusters (C++
/// `QKMeansAnomaly`).
///
/// ```text
/// score(x) = min over clusters k of |x - c_k|_w^2 / r_k^2, in Q8
/// ```
///
/// `|v|_w^2 = sum_f w_f v_f^2` is a squared distance with a weight per feature
/// (all weights 1 when `feature_weight` is `None`). The trainer sets `w_f`
/// inversely to how much feature `f` varies within the clusters of normal
/// data, which makes the distance a diagonal Mahalanobis distance: a feature
/// that is normally steady counts for more when it moves than one that
/// normally swings. `r_k^2` is cluster `k`'s mean squared radius, so a typical
/// member of a cluster scores about 256 and a sample twice as far from every
/// centroid as that cluster's typical member scores about 1024. Normalizing by
/// the radius lets a tight cluster and a loose one share one threshold.
///
/// [`score`](Self::score) returns `i32::MAX` if no cluster is in use. The
/// tables can live in flash, or in a [`QKMeansAnomalyTrainer`] that learned
/// them ([`QKMeansAnomalyTrainer::model`]).
///
/// ```
/// use tinymind::int8::QKMeansAnomaly;
///
/// // Two clusters over two features, each with mean squared radius 4 LSB^2.
/// static CENTROIDS: [[i8; 2]; 2] = [[0, 0], [100, 100]];
/// static INV_RADIUS_SQ: [u32; 2] = [1 << 30, 1 << 30]; // 2^32 / 4
/// let model: QKMeansAnomaly<'_, 2, 2> =
///     QKMeansAnomaly { centroids: &CENTROIDS, inv_radius_sq: &INV_RADIUS_SQ, feature_weight: None };
/// assert_eq!(model.score(&[2, 0]), 256); // a typical member: 4 / 4 in Q8
/// assert_eq!(model.score(&[98, 96]), 5 * 256); // 20 / 4
/// ```
#[derive(Clone, Copy, Debug)]
pub struct QKMeansAnomaly<'a, const F: usize, const K: usize> {
    /// `K` centroids on the features' int8 grid.
    pub centroids: &'a [[i8; F]; K],
    /// `floor(2^32 / r_k^2)` per cluster, with `r_k^2` in the units of the
    /// weighted distance (LSB^2 times the weights); 0 marks a cluster that is
    /// not in use.
    pub inv_radius_sq: &'a [u32; K],
    /// One weight per feature (any fixed point; the trainer uses Q8), or
    /// `None` for an unweighted distance.
    pub feature_weight: Option<&'a [u16; F]>,
}

impl<const F: usize, const K: usize> QKMeansAnomaly<'_, F, K> {
    /// The anomaly score of `x` in Q8 (see the type's documentation),
    /// saturated to `i32::MAX`.
    pub fn score(&self, x: &[i8; F]) -> i32 {
        const { check::<F, K>() };
        let mut best = u64::MAX;
        for (c, &inv) in self.centroids.iter().zip(self.inv_radius_sq) {
            if inv == 0 {
                continue;
            }
            let d2 = weighted_squared_distance(x, c, self.feature_weight);
            best = best.min(normalize_distance(d2, inv));
        }
        i32::try_from(best).unwrap_or(i32::MAX)
    }
}

/// Learns a [`QKMeansAnomaly`] from normal samples on the device, one sample
/// at a time, in two phases (C++ `QKMeansAnomalyTrainer`):
///
/// - [`learn`](Self::learn): online (MacQueen) k-means. The first `K` samples
///   seed the centroids; after that each sample moves its nearest centroid by
///   `(x - c) / n`, `n` the number of samples that centroid has absorbed. Once
///   `n` reaches the cap set by [`set_maximum_count`](Self::set_maximum_count),
///   the step stays `1 / cap`: the centroid becomes a moving average that
///   follows slow drift. The centroids are kept in Q8 (`i16`), so steps far
///   below one LSB still accumulate. `learn` also measures how much each
///   feature varies around its centroid (from samples whose centroid has
///   absorbed at least 8), which sets the feature weights.
/// - [`calibrate`](Self::calibrate): the first call rounds the centroids to
///   int8, sets the feature weights and freezes both; every call adds the
///   weighted `|x - c|^2` to the nearest cluster's radius estimate. Calibrate
///   on normal samples not used in `learn` for an honest radius, or on the
///   same ones.
///
/// Feature weights: `w_f = 256 * mean_g(v_g) / v_f` in Q8, clamped to
/// `1..=65535`, where `v_f` is feature `f`'s mean squared deviation from its
/// centroid during learning.
/// [`set_feature_weighting(false)`](Self::set_feature_weighting) before
/// calibrating gives the plain Euclidean distance instead.
///
/// [`model`](Self::model) then gives the [`QKMeansAnomaly`], borrowing this
/// object. A cluster that no calibration sample reached is left out of the
/// score. Passing the learning data several times through `learn` brings
/// online k-means closer to batch k-means; one pass is enough for a
/// stationary signal.
///
/// RAM: `3 * K * F` bytes, plus 18 bytes per cluster and 10 per feature.
///
/// ```
/// use tinymind::int8::QKMeansAnomalyTrainer;
///
/// let mut trainer = QKMeansAnomalyTrainer::<2, 1>::new();
/// for i in 0..64i8 {
///     trainer.learn(&[i % 4, -(i % 4)]); // normal: a small wobble around (1.5, -1.5)
/// }
/// for i in 0..16i8 {
///     trainer.calibrate(&[i % 4, -(i % 4)]);
/// }
/// let model = trainer.model();
/// assert!(model.score(&[1, -1]) < 512);
/// assert!(model.score(&[40, -40]) > 100 * 256);
/// ```
#[derive(Clone, Debug)]
pub struct QKMeansAnomalyTrainer<const F: usize, const K: usize> {
    residual_sum: [u64; F],
    residual_count: u32,
    weight: [u16; F],
    centroid_q8: [[i16; F]; K],
    centroid: [[i8; F]; K],
    sum_squared_distance: [u64; K],
    calibration_count: [u32; K],
    inverse_radius_squared: [u32; K],
    count: [u16; K],
    maximum_count: u16,
    seeded: usize,
    frozen: bool,
    weighting: bool,
}

#[cfg(kani)]
impl<const F: usize, const K: usize> QKMeansAnomalyTrainer<F, K> {
    /// A trainer in any state the public methods can leave it in: every field
    /// symbolic, the cap at least 1, at most `K` clusters seeded, counts
    /// within the cap.
    pub(crate) fn any_state() -> Self {
        let t = QKMeansAnomalyTrainer {
            residual_sum: kani::any(),
            residual_count: kani::any(),
            weight: kani::any(),
            centroid_q8: kani::any(),
            centroid: kani::any(),
            sum_squared_distance: kani::any(),
            calibration_count: kani::any(),
            inverse_radius_squared: kani::any(),
            count: kani::any(),
            maximum_count: kani::any(),
            seeded: kani::any(),
            frozen: kani::any(),
            weighting: kani::any(),
        };
        kani::assume(t.maximum_count >= 1 && t.seeded <= K);
        kani::assume(t.count.iter().all(|&c| c <= t.maximum_count));
        t
    }
}

impl<const F: usize, const K: usize> Default for QKMeansAnomalyTrainer<F, K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const F: usize, const K: usize> QKMeansAnomalyTrainer<F, K> {
    /// A trainer with nothing learned, feature weighting on and the step cap
    /// at `u16::MAX`.
    pub const fn new() -> Self {
        const { check::<F, K>() };
        QKMeansAnomalyTrainer {
            residual_sum: [0; F],
            residual_count: 0,
            weight: [256; F],
            centroid_q8: [[0; F]; K],
            centroid: [[0; F]; K],
            sum_squared_distance: [0; K],
            calibration_count: [0; K],
            inverse_radius_squared: [0; K],
            count: [0; K],
            maximum_count: u16::MAX,
            seeded: 0,
            frozen: false,
            weighting: true,
        }
    }

    /// Forget everything learned and calibrated. The step cap and the
    /// weighting setting are kept.
    pub fn reset(&mut self) {
        *self = QKMeansAnomalyTrainer {
            maximum_count: self.maximum_count,
            weighting: self.weighting,
            ..Self::new()
        };
    }

    /// Weight the features (the default) or use the plain Euclidean distance.
    /// Takes effect in [`calibrate`](Self::calibrate) and [`model`](Self::model).
    pub fn set_feature_weighting(&mut self, enabled: bool) {
        self.weighting = enabled;
    }

    /// Cap the number of samples a centroid counts (0 is taken as 1): once a
    /// centroid has absorbed `count`, each new sample moves it by
    /// `(x - c) / count`, a moving average.
    pub fn set_maximum_count(&mut self, count: u16) {
        self.maximum_count = count.max(1);
    }

    /// One online k-means step on a normal sample (see the type's
    /// documentation). Does nothing once [`calibrate`](Self::calibrate) has
    /// been called.
    pub fn learn(&mut self, x: &[i8; F]) {
        if self.frozen {
            return;
        }
        if let Some(seed) = self.centroid_q8.get_mut(self.seeded) {
            for (c, &v) in seed.iter_mut().zip(x) {
                *c = i16::from(v) * 256;
            }
            if let Some(n) = self.count.get_mut(self.seeded) {
                *n = 1;
            }
            self.seeded += 1;
            return;
        }

        let mut nearest = 0;
        let mut best = u64::MAX;
        for (k, c) in self.centroid_q8.iter().enumerate() {
            let mut d2 = 0u64;
            for (&v, &cf) in x.iter().zip(c) {
                // |d| <= 65,280.
                let d = i64::from(v) * 256 - i64::from(cf);
                d2 = d2.wrapping_add((d * d) as u64);
            }
            if d2 < best {
                best = d2;
                nearest = k;
            }
        }

        let (Some(count), Some(centroid)) = (
            self.count.get_mut(nearest),
            self.centroid_q8.get_mut(nearest),
        ) else {
            return;
        };
        let settled = *count >= 8;
        if *count < self.maximum_count {
            *count += 1;
        }
        let n = i32::from(*count);
        for ((c, &v), sum) in centroid.iter_mut().zip(x).zip(self.residual_sum.iter_mut()) {
            let residual = i32::from(v) * 256 - i32::from(*c);
            if settled {
                // Q16 LSB^2.
                *sum = sum.wrapping_add((i64::from(residual) * i64::from(residual)) as u64);
            }
            // Between c and x * 256, so within i16.
            *c = (i32::from(*c) + rounded_divide(residual, n)) as i16;
        }
        self.residual_count = self.residual_count.wrapping_add(u32::from(settled));
    }

    /// Add a normal sample to the radius estimate of its nearest cluster. The
    /// first call freezes the centroids and the feature weights (see the
    /// type's documentation).
    pub fn calibrate(&mut self, x: &[i8; F]) {
        if !self.frozen {
            self.freeze();
        }
        let weights = if self.weighting {
            Some(&self.weight)
        } else {
            None
        };
        let mut nearest = None;
        let mut best = u64::MAX;
        for (k, c) in self.centroid.iter().enumerate().take(self.seeded) {
            let d2 = weighted_squared_distance(x, c, weights);
            if d2 < best || nearest.is_none() {
                best = d2;
                nearest = Some(k);
            }
        }
        let Some(k) = nearest else {
            return;
        };
        let (Some(sum), Some(count), Some(inverse)) = (
            self.sum_squared_distance.get_mut(k),
            self.calibration_count.get_mut(k),
            self.inverse_radius_squared.get_mut(k),
        ) else {
            return;
        };
        *sum = sum.wrapping_add(best);
        *count = count.wrapping_add(1);
        // r^2 = ceil(mean |x - c|^2), at least 1 LSB^2. A count that wrapped
        // to 0 (after 2^32 calls) leaves the radius as it was.
        let n = u64::from(*count);
        let Some(radius_squared) = sum.wrapping_add(n.wrapping_sub(1)).checked_div(n) else {
            return;
        };
        let inv = (1u64 << 32) / radius_squared.max(1);
        *inverse = u32::try_from(inv).unwrap_or(u32::MAX);
    }

    /// The learned model, borrowing this trainer's tables. Before the first
    /// [`calibrate`](Self::calibrate) every cluster is out of use and the
    /// score is `i32::MAX`.
    pub fn model(&self) -> QKMeansAnomaly<'_, F, K> {
        QKMeansAnomaly {
            centroids: &self.centroid,
            inv_radius_sq: &self.inverse_radius_squared,
            feature_weight: if self.weighting {
                Some(&self.weight)
            } else {
                None
            },
        }
    }

    /// How many clusters have been seeded (at most `K`).
    pub fn seeded_clusters(&self) -> usize {
        self.seeded
    }

    /// How many learning samples each centroid has absorbed (capped by
    /// [`set_maximum_count`](Self::set_maximum_count)).
    pub fn learn_counts(&self) -> &[u16; K] {
        &self.count
    }

    /// How many calibration samples reached each cluster.
    pub fn calibration_counts(&self) -> &[u32; K] {
        &self.calibration_count
    }

    /// The int8 centroids (all zero before the first
    /// [`calibrate`](Self::calibrate)).
    pub fn centroids(&self) -> &[[i8; F]; K] {
        &self.centroid
    }

    /// The feature weights in Q8 (all 256 before the first
    /// [`calibrate`](Self::calibrate), or if no sample settled).
    pub fn feature_weights(&self) -> &[u16; F] {
        &self.weight
    }

    fn freeze(&mut self) {
        for (c, q8) in self.centroid.iter_mut().zip(&self.centroid_q8) {
            for (v, &q) in c.iter_mut().zip(q8) {
                *v = rounded_divide(i32::from(q), 256).clamp(-128, 127) as i8;
            }
        }
        if self.residual_count > 0 {
            let mut total = 0u64;
            for &s in &self.residual_sum {
                total = total.wrapping_add(s);
            }
            // w_f = 256 * (total / F) / sum_f, the residual count cancelling.
            let mean_variance = total / F as u64;
            for (w, &sum) in self.weight.iter_mut().zip(&self.residual_sum) {
                // Scale numerator and denominator down together until the
                // numerator times 256 fits 64 bits.
                let mut numerator = mean_variance;
                let mut denominator = sum;
                while numerator > (u64::MAX >> 9) {
                    numerator >>= 1;
                    denominator >>= 1;
                }
                // numerator * 256 < 2^63; the sum wraps as the C++'s
                // unsigned sum would (only if the residual sums wrapped).
                let q = match (numerator * 256)
                    .wrapping_add(denominator / 2)
                    .checked_div(denominator)
                {
                    Some(q) => q.clamp(1, 65535),
                    None => 65535,
                };
                *w = q as u16;
            }
        }
        self.frozen = true;
    }
}

/// Mixture of `K` diagonal Gaussians over `F` int8 features, scored by hard
/// assignment (C++ `QDiagGaussianAnomaly`):
///
/// ```text
/// score(x) = min over k of sum_f (x_f - mu_kf)^2 / (2 sigma_kf^2) + b_k
/// b_k      = -ln(pi_k) + sum_f ln(sqrt(2 pi) sigma_kf)
/// ```
///
/// which is the negative log-likelihood of `x` under its most likely component
/// (an upper bound on the mixture's negative log-likelihood), in nats, Q8
/// (1/256 nat), saturated to the `i32` range. The variances are of the
/// real-valued features, so `b_k` is the real density's normalizer; the
/// quadratic term is the same in any units.
/// [`calibration::build_diag_gaussian_anomaly`](super::calibration::build_diag_gaussian_anomaly)
/// makes all three tables from a float fit (for example scikit-learn's
/// `GaussianMixture` with `covariance_type="diag"`).
///
/// ```
/// use tinymind::int8::QDiagGaussianAnomaly;
///
/// // One component, one feature: mean 0, sigma 1 LSB (precision 2^24 / 2),
/// // bias 0.
/// let model: QDiagGaussianAnomaly<'_, 1, 1> =
///     QDiagGaussianAnomaly { means: &[[0]], precision: &[[1 << 23]], bias: &[0] };
/// assert_eq!(model.score(&[0]), 0);
/// assert_eq!(model.score(&[2]), 2 * 256); // 2^2 / 2 nats
/// ```
#[derive(Clone, Copy, Debug)]
pub struct QDiagGaussianAnomaly<'a, const F: usize, const K: usize> {
    /// Component means on the features' int8 grid.
    pub means: &'a [[i8; F]; K],
    /// `2^24 / (2 sigma^2)` per component and feature, sigma in LSBs (Q24 of
    /// `1 / (2 sigma^2)`; a variance of 1/512 LSB^2 is the smallest it can
    /// hold).
    pub precision: &'a [[u32; F]; K],
    /// `b_k` per component in Q8.
    pub bias: &'a [i32; K],
}

impl<const F: usize, const K: usize> QDiagGaussianAnomaly<'_, F, K> {
    /// The score of `x` in Q8 nats (see the type's documentation).
    pub fn score(&self, x: &[i8; F]) -> i32 {
        const { check::<F, K>() };
        let mut best = i64::MAX;
        for ((mu, p), &b) in self.means.iter().zip(self.precision).zip(self.bias) {
            // Q24; each term is below 2^48, so F <= 65,536 terms fit.
            let mut quadratic = 0u64;
            for ((&v, &m), &pf) in x.iter().zip(mu).zip(p) {
                let d = i32::from(v) - i32::from(m);
                quadratic = quadratic.wrapping_add(u64::from((d * d) as u32) * u64::from(pf));
            }
            // To Q8, rounded; at most 2^48 after the shift.
            let value = (quadratic.wrapping_add(32768) >> 16) as i64 + i64::from(b);
            best = best.min(value);
        }
        best.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounded_divide_ties_away_from_zero() {
        assert_eq!(rounded_divide(5, 2), 3);
        assert_eq!(rounded_divide(-5, 2), -3);
        assert_eq!(rounded_divide(4, 3), 1);
        assert_eq!(rounded_divide(-4, 3), -1);
        assert_eq!(rounded_divide(-128 * 256, 256), -128);
        assert_eq!(rounded_divide(7, 0), 0);
    }

    #[test]
    fn normalize_distance_switches_form_above_32_bits() {
        assert_eq!(normalize_distance(1 << 20, 1 << 24), 1 << 20);
        let big = (1u64 << 40) + 12345;
        assert_eq!(normalize_distance(big, 1 << 24), (big >> 16) << 16);
    }

    #[test]
    fn unused_clusters_are_skipped() {
        let model: QKMeansAnomaly<'_, 1, 2> = QKMeansAnomaly {
            centroids: &[[0], [10]],
            inv_radius_sq: &[0, 1 << 24],
            feature_weight: None,
        };
        // Cluster 0 is closer but out of use; cluster 1's r^2 is 256.
        assert_eq!(model.score(&[0]), 100);
        let none: QKMeansAnomaly<'_, 1, 1> = QKMeansAnomaly {
            centroids: &[[0]],
            inv_radius_sq: &[0],
            feature_weight: None,
        };
        assert_eq!(none.score(&[0]), i32::MAX);
    }

    #[test]
    fn trainer_before_calibration_scores_max() {
        let mut t = QKMeansAnomalyTrainer::<3, 2>::new();
        t.learn(&[1, 2, 3]);
        assert_eq!(t.seeded_clusters(), 1);
        assert_eq!(t.model().score(&[1, 2, 3]), i32::MAX);
        t.reset();
        assert_eq!(t.seeded_clusters(), 0);
    }
}
