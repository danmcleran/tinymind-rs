//! The gas_sensor_drift example's program, shared by
//! `examples/gas_sensor_drift.rs` (which parses the command line) and
//! `tests/gas_golden.rs` (which runs it in-process against the C++ goldens).
//! Port of tinymind (C++) `examples/gas_sensor_drift/gas_sensor_drift.cpp`;
//! see `examples/gas_sensor_drift.rs`.

use std::f64::consts::PI;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use tinymind::head::{DenseHead, PrototypeHead};
use tinymind::rand::GlibcRand;
use tinymind::{MultilayerPerceptron, Q16_16, Relu, Sigmoid};

#[path = "cppfmt.rs"]
mod cppfmt;
use cppfmt::{fmt_fixed, fmt_g};

#[allow(dead_code)]
#[path = "std_random.rs"]
mod std_random;
use std_random::Mt19937;

/// 16 sensors x 8 features.
const NUMBER_OF_INPUTS: usize = 128;
const NUMBER_OF_NEURONS_PER_HIDDEN_LAYER: usize = 32;
/// 6 gas classes.
const NUMBER_OF_OUTPUTS: usize = 6;

// Q8.8 collapses on a 128-input net: Q16.16.
type NeuralNetwork = MultilayerPerceptron<
    Q16_16,
    Relu,
    Sigmoid,
    NUMBER_OF_INPUTS,
    NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,
    NUMBER_OF_OUTPUTS,
>;

const CLASS_NAMES: [&str; NUMBER_OF_OUTPUTS] = [
    "Ethanol",
    "Ethylene",
    "Ammonia",
    "Acetaldehyde",
    "Acetone",
    "Toluene",
];

const NUM_CLASSES: usize = NUMBER_OF_OUTPUTS;
const NUM_BATCHES: usize = 10;
const SAMPLES_PER_CLASS: usize = 150;
/// Per-sample Gaussian spread.
const SAMPLE_NOISE_STDEV: f64 = 0.18;
/// Multiplicative drift slope.
const DRIFT_GAIN_PER_BATCH: f64 = 0.16;
/// Additive drift slope.
const DRIFT_OFFSET_PER_BATCH: f64 = 0.13;

/// C++ `WEIGHT_INIT_SEED`: seeds the initial weights only.
const WEIGHT_INIT_SEED: u32 = 1;
/// Seeds the data, the shuffle and the training-sample draws.
const RANDOM_SEED: u32 = 7;
const ITERATIONS: u32 = 50_000;
const REPORT_EVERY: u32 = 2_000;

/// The trained MLP split into a frozen backbone (128 -> 32 ReLU) and a head
/// (32 -> 6): the embedding size.
const EMBEDDING: usize = NUMBER_OF_NEURONS_PER_HIDDEN_LAYER;
type Head = DenseHead<Q16_16, EMBEDDING, NUMBER_OF_OUTPUTS>;
type Prototypes = PrototypeHead<Q16_16, EMBEDDING, NUMBER_OF_OUTPUTS>;
type Embedding = [Q16_16; EMBEDDING];

const METHOD_NAMES: [&str; 7] = [
    "none",
    "renorm",
    "renorm_all",
    "prototype",
    "head",
    "bias",
    "bias_head",
];
const NONE: usize = 0;
const RENORM: usize = 1;
const RENORM_ALL: usize = 2;
const PROTOTYPE: usize = 3;
const HEAD: usize = 4;
const BIAS: usize = 5;
const BIAS_HEAD: usize = 6;
const NUM_METHODS: usize = METHOD_NAMES.len();

/// Labelled samples per class to adapt on.
const SHOTS: [usize; 5] = [1, 2, 5, 10, 20];
/// Random draws of the shots per (batch, shots) cell.
const DRAWS: u32 = 10;
const ADAPT_EPOCHS: u32 = 30;
const ADAPT_LEARNING_RATE: f64 = 0.05;

#[derive(Clone, Copy)]
struct Sample {
    /// 128 sensor-feature responses.
    feat: [f64; NUMBER_OF_INPUTS],
    /// Gas class, 0..5.
    label: usize,
    /// Month/batch index, 1..10.
    batch: usize,
}

/// Per-feature z-score statistics.
struct FeatureStats {
    mean: [f64; NUMBER_OF_INPUTS],
    stdev: [f64; NUMBER_OF_INPUTS],
}

/// Synthesize the drift dataset, train on batch 1, write the drift curve, then
/// compare seven ways to adapt to each drifted batch. Writes four CSV files into
/// `out_dir` and the log to `stdout`.
pub fn run(out_dir: &Path, stdout: &mut dyn Write) -> io::Result<()> {
    // As in the xor port: the C++ seeds rand() with WEIGHT_INIT_SEED, builds the
    // network (drawing its weights), then re-seeds with 7. Two generators here.
    let mut init_rng = GlibcRand::new(WEIGHT_INIT_SEED);
    let mut nn = Box::new(NeuralNetwork::new(|| random_weight(init_rng.next_i32())));
    let mut rng = GlibcRand::new(RANDOM_SEED);

    // ---- Synthesize the drift dataset ----
    let data = synthesize_dataset(&mut rng);
    writeln!(
        stdout,
        "Synthesized {} samples across {NUM_BATCHES} batches ({NUM_CLASSES} classes, \
         {SAMPLES_PER_CLASS} samples/class/batch).",
        data.len()
    )?;

    // ---- Split batch 1 into 80/20 train/val ----
    let mut batch1: Vec<Sample> = data.iter().filter(|s| s.batch == 1).copied().collect();
    for i in (2..=batch1.len()).rev() {
        let j = rng.next_i32() as usize % i;
        batch1.swap(i - 1, j);
    }
    let n_train = batch1.len() * 4 / 5;
    let (train, val) = batch1.split_at(n_train);
    writeln!(
        stdout,
        "Batch 1 -> train: {}  val: {}",
        train.len(),
        val.len()
    )?;

    // Normalization fit on the batch-1 training split only.
    let stats = fit_stats(train.iter());

    // ---- Training (batch 1 only) ----
    let mut loss_csv = BufWriter::new(File::create(out_dir.join("gas_loss.csv"))?);
    writeln!(loss_csv, "iter,avg_err")?;
    let mut err_sum = 0.0;
    for it in 0..ITERATIONS {
        let s = &train[rng.next_i32() as usize % train.len()];
        let input = to_input(s, &stats);
        let target = one_hot(s.label);

        nn.feed_forward(&input);
        let err = nn.calculate_error(&target);
        if !err.is_within_zero_tolerance() {
            nn.train(&target);
        }
        err_sum += err.to_f64().abs();

        if (it + 1) % REPORT_EVERY == 0 {
            let avg = err_sum / f64::from(REPORT_EVERY);
            writeln!(
                stdout,
                "iter {:>6}   avg|err| = {}",
                it + 1,
                fmt_fixed(avg, 4)
            )?;
            writeln!(loss_csv, "{},{}", it + 1, fmt_g(avg))?;
            err_sum = 0.0;
        }
    }
    loss_csv.flush()?;

    // ---- Batch-1 validation confusion matrix ----
    let mut confusion = [[0u32; NUM_CLASSES]; NUM_CLASSES];
    let mut val_correct = 0usize;
    for s in val {
        let pred = predict_class(&mut nn, s, &stats);
        confusion[s.label][pred] += 1;
        if pred == s.label {
            val_correct += 1;
        }
    }
    let batch1_val_acc = val_correct as f64 / val.len() as f64;

    let mut cm = BufWriter::new(File::create(out_dir.join("gas_confusion.csv"))?);
    write!(cm, "actual")?;
    for c in 0..NUM_CLASSES {
        write!(cm, ",pred_{c}")?;
    }
    writeln!(cm)?;
    for (name, row) in CLASS_NAMES.iter().zip(&confusion) {
        write!(cm, "{name}")?;
        for count in row {
            write!(cm, ",{count}")?;
        }
        writeln!(cm)?;
    }
    cm.flush()?;

    // ---- Drift curve: accuracy on each batch 1..10 ----
    let mut drift_csv = BufWriter::new(File::create(out_dir.join("gas_drift.csv"))?);
    writeln!(drift_csv, "batch,accuracy")?;
    writeln!(stdout, "\nDrift curve (model trained on batch 1 only):")?;
    let mut batch10_acc = 0.0;
    for b in 1..=NUM_BATCHES {
        // Batch 1 is scored on the held-out val split, so the curve's first
        // point is an honest in-distribution number; later batches are entirely
        // unseen, so all of them are scored.
        let acc = if b == 1 {
            batch1_val_acc
        } else {
            let mut correct = 0usize;
            let mut total = 0usize;
            for s in data.iter().filter(|s| s.batch == b) {
                total += 1;
                if predict_class(&mut nn, s, &stats) == s.label {
                    correct += 1;
                }
            }
            correct as f64 / total as f64
        };
        if b == NUM_BATCHES {
            batch10_acc = acc;
        }
        writeln!(drift_csv, "{b},{}", fmt_fixed(acc, 4))?;
        writeln!(stdout, "  batch {b:>2}   accuracy = {}", fmt_fixed(acc, 4))?;
    }
    drift_csv.flush()?;

    writeln!(
        stdout,
        "\nBatch 1 (val) accuracy: {}",
        fmt_fixed(batch1_val_acc, 4)
    )?;
    writeln!(
        stdout,
        "Batch 10 accuracy:      {}   (drift degradation)",
        fmt_fixed(batch10_acc, 4)
    )?;

    run_adaptation(&mut nn, &data, val, &stats, out_dir, stdout)?;
    stdout.flush()
}

/// The predicted class via argmax of the sigmoid outputs (the first one on a
/// tie, as the C++ strict `>` scan does).
fn predict_class(nn: &mut NeuralNetwork, s: &Sample, stats: &FeatureStats) -> usize {
    nn.feed_forward(&to_input(s, stats));
    let learned = nn.learned_values();
    let mut pred = 0;
    let mut best = learned[0].to_f64();
    for (k, value) in learned.iter().enumerate().skip(1) {
        let v = value.to_f64();
        if v > best {
            best = v;
            pred = k;
        }
    }
    pred
}

// ---------------------------------------------------------------------------
// Adapting to drift on the device
// ---------------------------------------------------------------------------
//
// The trained MLP is split in two: a frozen backbone (the 128 -> 32 ReLU
// layer) and a head (the 32 -> 6 layer). On each drifted batch a few labelled
// samples per class are drawn (the "shots"); the rest of the batch is the test
// set. Seven ways to use them are compared, each starting from the shipped
// model:
//
//   none       the shipped model, unchanged
//   renorm     z-score statistics recomputed from the shots, labels unused
//   renorm_all z-score statistics from the whole batch, labels unused (a
//              reference: unlabelled data is cheap on a device)
//   prototype  a PrototypeHead on the backbone's embedding: one mean
//              embedding per class from the shots, nearest mean wins
//   head       a DenseHead loaded with the shipped head and trained on the
//              shots with softmax cross-entropy (head-only fine-tuning)
//   bias       only the backbone's 32 biases trained, the head frozen; the
//              gradient comes from DenseHead::backward()
//   bias_head  the biases and the head trained together
//
// Each (batch, shots) cell is averaged over 10 random draws of the shots.

/// The frozen backbone: the MLP's input-to-hidden weights and biases.
struct Backbone {
    weights: [[Q16_16; NUMBER_OF_INPUTS]; EMBEDDING],
    biases: Embedding,
}

impl Backbone {
    fn load(nn: &NeuralNetwork) -> Self {
        let input = nn.input_weights();
        Backbone {
            weights: core::array::from_fn(|k| core::array::from_fn(|i| input[i][k])),
            biases: nn.input_bias_weights(),
        }
    }

    /// `W x` without the bias, cached per sample so that a bias change costs
    /// 32 additions instead of a 128 x 32 product.
    fn pre_activation(&self, input: &[Q16_16; NUMBER_OF_INPUTS]) -> Embedding {
        self.weights.map(|row| {
            let mut sum = Q16_16::ZERO;
            for (&w, &x) in row.iter().zip(input) {
                sum += w * x;
            }
            sum
        })
    }
}

/// The ReLU embedding from a cached pre-activation and the backbone biases.
fn embed(pre: &Embedding, bias: &Embedding) -> Embedding {
    core::array::from_fn(|k| {
        let z = pre[k] + bias[k];
        if z > Q16_16::ZERO { z } else { Q16_16::ZERO }
    })
}

/// Accuracy of `head` (or `prototypes` when given) on the samples whose cached
/// pre-activations are `pre`, with backbone biases `bias`.
fn score_head(
    head: &Head,
    prototypes: Option<&Prototypes>,
    pre: &[Embedding],
    labels: &[usize],
    bias: &Embedding,
) -> f64 {
    let mut correct = 0usize;
    for (p, &label) in pre.iter().zip(labels) {
        let h = embed(p, bias);
        let predicted = match prototypes {
            Some(prototypes) => prototypes.predict(&h),
            None => head.predict(&h),
        };
        if predicted == label {
            correct += 1;
        }
    }
    correct as f64 / pre.len() as f64
}

/// Trains the backbone biases (when `train_bias`) and the head (when
/// `train_head`) on the shots for `ADAPT_EPOCHS` epochs of per-sample SGD.
fn adapt(
    head: &mut Head,
    bias: &mut Embedding,
    shot_pre: &[Embedding],
    shot_labels: &[usize],
    train_bias: bool,
    train_head: bool,
    rng: &mut Mt19937,
) {
    let rate = to_q(ADAPT_LEARNING_RATE);
    let mut h_deltas = [Q16_16::ZERO; EMBEDDING];
    let mut order: Vec<usize> = (0..shot_pre.len()).collect();

    for _ in 0..ADAPT_EPOCHS {
        for j in (2..=order.len()).rev() {
            order.swap(j - 1, rng.next_u32() as usize % j);
        }
        for &j in &order {
            let pre = &shot_pre[j];
            let h = embed(pre, bias);
            if train_head {
                let deltas = if train_bias {
                    Some(&mut h_deltas)
                } else {
                    None
                };
                head.train_class(&h, shot_labels[j], rate, deltas);
            } else {
                let logits = head.forward(&h);
                let deltas = head.output_deltas(&logits, &one_hot(shot_labels[j]));
                h_deltas = head.backward(&deltas);
            }
            if train_bias {
                // dL/db through the ReLU: the hidden delta where the unit is on.
                for k in 0..EMBEDDING {
                    if pre[k] + bias[k] > Q16_16::ZERO {
                        bias[k] += rate.mul_round_nearest(h_deltas[k]);
                    }
                }
            }
        }
    }
}

fn run_adaptation(
    nn: &mut NeuralNetwork,
    data: &[Sample],
    val: &[Sample],
    stats: &FeatureStats,
    out_dir: &Path,
    stdout: &mut dyn Write,
) -> io::Result<()> {
    let backbone = Backbone::load(nn);
    let hidden = nn.hidden_weights();
    let shipped_weights: [Embedding; NUMBER_OF_OUTPUTS] =
        core::array::from_fn(|o| core::array::from_fn(|k| hidden[k][o]));
    let shipped_biases = nn.hidden_bias_weights();
    let shipped = Head::from_weights(shipped_weights, shipped_biases);

    // Cache every sample's pre-activation under the batch-1 stats.
    let pre: Vec<Embedding> = data
        .iter()
        .map(|s| backbone.pre_activation(&to_input(s, stats)))
        .collect();
    let labels: Vec<usize> = data.iter().map(|s| s.label).collect();
    let val_pre: Vec<Embedding> = val
        .iter()
        .map(|s| backbone.pre_activation(&to_input(s, stats)))
        .collect();
    let val_labels: Vec<usize> = val.iter().map(|s| s.label).collect();

    // The split must reproduce the network exactly: Q16.16 sums wrap, so the
    // hidden layer and the logits agree bit for bit with the MLP's.
    {
        let mut hidden_mismatches = 0usize;
        let mut argmax_mismatches = 0usize;
        let mut tied_mismatches = 0usize;
        for (s, p) in data.iter().zip(&pre) {
            nn.feed_forward(&to_input(s, stats));
            let h = embed(p, &backbone.biases);
            hidden_mismatches += nn
                .hidden_values()
                .iter()
                .zip(&h)
                .filter(|(a, b)| a != b)
                .count();
            let learned = nn.learned_values();
            let mut best = 0;
            for o in 1..NUMBER_OF_OUTPUTS {
                if learned[o] > learned[best] {
                    best = o;
                }
            }
            let head_best = shipped.predict(&h);
            if best != head_best {
                argmax_mismatches += 1;
                // The sigmoid table saturates, so two confident outputs can
                // both read 1.0 and the MLP takes the first; the head compares
                // the logits underneath.
                if learned[best] == learned[head_best] {
                    tied_mismatches += 1;
                }
            }
        }
        writeln!(
            stdout,
            "\nBackbone + DenseHead vs the MLP over {} samples: {hidden_mismatches} hidden \
             values differ; {argmax_mismatches} predictions differ, {tied_mismatches} of them \
             where two sigmoid outputs tie.",
            data.len()
        )?;
    }

    let mut by_batch_class = vec![vec![Vec::new(); NUM_CLASSES]; NUM_BATCHES + 1];
    for (j, s) in data.iter().enumerate() {
        by_batch_class[s.batch][s.label].push(j);
    }

    let mut csv = BufWriter::new(File::create(out_dir.join("gas_adapt.csv"))?);
    writeln!(
        csv,
        "method,batch,shots,accuracy_mean,accuracy_std,batch1_accuracy_mean"
    )?;

    let mut batch10 = [[0.0; SHOTS.len()]; NUM_METHODS];

    for (b, classes) in by_batch_class.iter().enumerate().skip(2) {
        // renorm_all: statistics from the whole batch, independent of the draw.
        let all_stats = fit_stats(classes.iter().flatten().map(|&j| &data[j]));

        for (si, &shots) in SHOTS.iter().enumerate() {
            let mut sum = [0.0; NUM_METHODS];
            let mut sum_sq = [0.0; NUM_METHODS];
            let mut sum_b1 = [0.0; NUM_METHODS];

            for r in 0..DRAWS {
                let seed = (b * 10_000 + shots * 100) as u32 + r;
                let mut rng = Mt19937::new(seed);

                let mut shot_idx = Vec::new();
                let mut test_idx = Vec::new();
                for pool in classes {
                    let mut pool = pool.clone();
                    for k in 0..shots {
                        let pick = k + rng.next_u32() as usize % (pool.len() - k);
                        pool.swap(k, pick);
                    }
                    shot_idx.extend_from_slice(&pool[..shots]);
                    test_idx.extend_from_slice(&pool[shots..]);
                }

                let shot_pre: Vec<Embedding> = shot_idx.iter().map(|&j| pre[j]).collect();
                let shot_labels: Vec<usize> = shot_idx.iter().map(|&j| labels[j]).collect();
                let test_pre: Vec<Embedding> = test_idx.iter().map(|&j| pre[j]).collect();
                let test_labels: Vec<usize> = test_idx.iter().map(|&j| labels[j]).collect();

                let mut acc = [0.0; NUM_METHODS];
                let mut acc_b1 = [0.0; NUM_METHODS];

                acc[NONE] = score_head(&shipped, None, &test_pre, &test_labels, &backbone.biases);
                acc_b1[NONE] = score_head(&shipped, None, &val_pre, &val_labels, &backbone.biases);

                // renorm / renorm_all: new statistics, shipped weights.
                for m in [RENORM, RENORM_ALL] {
                    let shot_stats;
                    let st = if m == RENORM {
                        shot_stats = fit_stats(shot_idx.iter().map(|&j| &data[j]));
                        &shot_stats
                    } else {
                        &all_stats
                    };
                    let rp: Vec<Embedding> = test_idx
                        .iter()
                        .map(|&j| backbone.pre_activation(&to_input(&data[j], st)))
                        .collect();
                    acc[m] = score_head(&shipped, None, &rp, &test_labels, &backbone.biases);
                    let vp: Vec<Embedding> = val
                        .iter()
                        .map(|s| backbone.pre_activation(&to_input(s, st)))
                        .collect();
                    acc_b1[m] = score_head(&shipped, None, &vp, &val_labels, &backbone.biases);
                }

                // prototype: class means of the shots' embeddings.
                {
                    let mut prototypes = Prototypes::new();
                    for (p, &label) in shot_pre.iter().zip(&shot_labels) {
                        prototypes.learn(&embed(p, &backbone.biases), label);
                    }
                    acc[PROTOTYPE] = score_head(
                        &shipped,
                        Some(&prototypes),
                        &test_pre,
                        &test_labels,
                        &backbone.biases,
                    );
                    acc_b1[PROTOTYPE] = score_head(
                        &shipped,
                        Some(&prototypes),
                        &val_pre,
                        &val_labels,
                        &backbone.biases,
                    );
                }

                // head / bias / bias_head: SGD on the shots.
                for m in [HEAD, BIAS, BIAS_HEAD] {
                    let mut head = shipped;
                    let mut bias = backbone.biases;
                    let mut order = Mt19937::new(seed + 7);
                    adapt(
                        &mut head,
                        &mut bias,
                        &shot_pre,
                        &shot_labels,
                        m != HEAD,
                        m != BIAS,
                        &mut order,
                    );
                    acc[m] = score_head(&head, None, &test_pre, &test_labels, &bias);
                    acc_b1[m] = score_head(&head, None, &val_pre, &val_labels, &bias);
                }

                for m in 0..NUM_METHODS {
                    sum[m] += acc[m];
                    sum_sq[m] += acc[m] * acc[m];
                    sum_b1[m] += acc_b1[m];
                }
            }

            let draws = f64::from(DRAWS);
            for m in 0..NUM_METHODS {
                let mean = sum[m] / draws;
                let var = f64::max(0.0, sum_sq[m] / draws - mean * mean);
                writeln!(
                    csv,
                    "{},{b},{shots},{},{},{}",
                    METHOD_NAMES[m],
                    fmt_fixed(mean, 4),
                    fmt_fixed(var.sqrt(), 4),
                    fmt_fixed(sum_b1[m] / draws, 4)
                )?;
                if b == NUM_BATCHES {
                    batch10[m][si] = mean;
                }
            }
        }
    }
    csv.flush()?;

    writeln!(
        stdout,
        "\nBatch 10 accuracy after adapting on k labelled samples per class (mean of {DRAWS} draws):"
    )?;
    write!(stdout, "  {:<11}", "method")?;
    for shots in SHOTS {
        write!(stdout, "{:>8}", format!("k={shots}"))?;
    }
    writeln!(stdout)?;
    for (name, row) in METHOD_NAMES.iter().zip(&batch10) {
        write!(stdout, "  {name:<11}")?;
        for &v in row {
            write!(stdout, "{:>8}", fmt_fixed(v, 3))?;
        }
        writeln!(stdout)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Synthetic dataset generator (physically-motivated, documented rules)
// ---------------------------------------------------------------------------
//
// 6 gas classes (Ethanol, Ethylene, Ammonia, Acetaldehyde, Acetone, Toluene).
// Each class has a distinct 128-dim mean response vector built so the classes
// are cleanly separable in batch 1. Per-sample Gaussian noise gives spread.
//
// Drift: each of the 10 batches applies a progressively growing per-sensor-
// feature multiplicative gain and additive offset. The drift magnitude scales
// with (batch - 1), so batch 1 is drift-free and batch 10 is the most degraded.

/// C++ `randUniform`: `(rand() + 0.5) / (RAND_MAX + 1.0)`.
fn rand_uniform(rng: &mut GlibcRand) -> f64 {
    (f64::from(rng.next_i32()) + 0.5) / (f64::from(i32::MAX) + 1.0)
}

/// C++ `randNormal`: Box-Muller on two `rand_uniform` draws.
fn rand_normal(rng: &mut GlibcRand) -> f64 {
    let mut u1 = rand_uniform(rng);
    let u2 = rand_uniform(rng);
    if u1 < 1e-12 {
        u1 = 1e-12;
    }
    (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

/// Distinct per-class mean response vectors. Each class lights up a different
/// subset of the 16 sensors with a class-specific profile across the 8 features.
fn build_class_means() -> [[f64; NUMBER_OF_INPUTS]; NUM_CLASSES] {
    let mut means = [[0.0; NUMBER_OF_INPUTS]; NUM_CLASSES];
    for (c, row) in means.iter_mut().enumerate() {
        let c = c as f64;
        for sensor in 0..16 {
            let phase = 0.6 * c + 0.37 * f64::from(sensor);
            let sensor_gain = 1.0 + 0.9 * phase.sin();
            for feat in 0..8 {
                let feat_f = f64::from(feat);
                let feat_profile = (0.5 * feat_f + 0.31 * c).cos() + 0.4 * (0.8 * feat_f).sin();
                row[(sensor * 8 + feat) as usize] = sensor_gain * (0.8 + 0.5 * feat_profile);
            }
        }
    }
    means
}

fn synthesize_dataset(rng: &mut GlibcRand) -> Vec<Sample> {
    let means = build_class_means();

    // Per-sensor-feature drift direction (fixed pattern, scaled per batch).
    let drift_gain_dir: [f64; NUMBER_OF_INPUTS] =
        core::array::from_fn(|f| (0.21 * f as f64 + 1.3).sin());
    let drift_off_dir: [f64; NUMBER_OF_INPUTS] =
        core::array::from_fn(|f| (0.17 * f as f64 + 0.7).cos());

    let mut out = Vec::with_capacity(NUM_BATCHES * NUM_CLASSES * SAMPLES_PER_CLASS);
    for b in 1..=NUM_BATCHES {
        // 0 at batch 1.
        let drift_step = (b - 1) as f64;
        for (c, mean) in means.iter().enumerate() {
            for _ in 0..SAMPLES_PER_CLASS {
                let mut feat = [0.0; NUMBER_OF_INPUTS];
                for (f, x) in feat.iter_mut().enumerate() {
                    let gain = 1.0 + DRIFT_GAIN_PER_BATCH * drift_step * drift_gain_dir[f];
                    let offset = DRIFT_OFFSET_PER_BATCH * drift_step * drift_off_dir[f];
                    let noisy = mean[f] + SAMPLE_NOISE_STDEV * rand_normal(rng);
                    *x = gain * noisy + offset;
                }
                out.push(Sample {
                    feat,
                    label: c,
                    batch: b,
                });
            }
        }
    }
    out
}

/// Symmetric small init in [-0.5, 0.5] raw Q16.16 units (C++ `RandomNumberGenerator`).
fn random_weight(rand: i32) -> Q16_16 {
    let half = Q16_16::ONE.raw() / 2;
    Q16_16::from_raw(rand % (2 * half + 1) - half)
}

/// C++ `toQ`: `lround(v * 2^16)`. `f64::round` rounds halves away from zero, as
/// `lround` does.
fn to_q(v: f64) -> Q16_16 {
    Q16_16::from_raw((v * 65536.0).round() as i32)
}

fn one_hot(label: usize) -> [Q16_16; NUMBER_OF_OUTPUTS] {
    core::array::from_fn(|k| to_q(if label == k { 1.0 } else { 0.0 }))
}

/// z-score with `stats`, then /3 so typical values sit inside Q16.16's stable
/// range. The batch-1 statistics are applied to every later batch: drift
/// pushes them off the trained distribution, and the normalization does not
/// compensate.
fn to_input(s: &Sample, stats: &FeatureStats) -> [Q16_16; NUMBER_OF_INPUTS] {
    core::array::from_fn(|f| to_q((s.feat[f] - stats.mean[f]) / stats.stdev[f] / 3.0))
}

/// Mean and population standard deviation per feature, summed in the C++
/// order (the samples in the order given).
fn fit_stats<'a>(samples: impl Iterator<Item = &'a Sample> + Clone) -> FeatureStats {
    let mut mean = [0.0; NUMBER_OF_INPUTS];
    let mut stdev = [0.0; NUMBER_OF_INPUTS];
    let mut n = 0usize;
    for s in samples.clone() {
        n += 1;
        for (m, x) in mean.iter_mut().zip(s.feat) {
            *m += x;
        }
    }
    let n = n as f64;
    for m in &mut mean {
        *m /= n;
    }
    for s in samples {
        for ((sd, x), m) in stdev.iter_mut().zip(s.feat).zip(mean) {
            let d = x - m;
            *sd += d * d;
        }
    }
    for sd in &mut stdev {
        *sd = (*sd / n).sqrt();
        if *sd < 1e-6 {
            *sd = 1.0;
        }
    }
    FeatureStats { mean, stdev }
}
