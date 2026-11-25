---
title: Vibration anomaly detection
layout: default
nav_order: 13
---

# Vibration anomaly detection example

The `vibration_anomaly` example is a port of tinymind (C++)'s [vibration anomaly example](https://danmcleran.github.io/tinymind/examples/vibration_anomaly.html): a sensor-node pipeline that learns what normal vibration looks like on the device, frame by frame with no buffered data, and then flags faults it has never seen.

## How it works

- A synthetic machine (3,200 samples per second, int16) runs at a speed that wanders between 28 and 32 Hz and a load that moves its level by +-6 dB.
- Each 256-sample frame (hop 128) becomes 32 log-mel bands ([`QLogMelSpectrogram`]({{ site.baseurl }}/logmel)). The frame's mean band is subtracted, which keeps the spectrum's shape and drops its level, and the bands become int8 features at 1/8 nat per step.
- **Learn** (120 s of normal running): an `int8::QKMeansAnomalyTrainer` with 8 clusters learns its centroids one frame at a time, by online k-means, and measures how much each feature varies within its cluster, which sets a weight per feature.
- **Calibrate** (60 s): the trainer freezes the centroids and measures each cluster's radius.
- **Threshold** (60 s): the largest score seen on normal running, times 1.25, becomes the alarm threshold.
- **Test**: normal running, then three faults whose severity grows from nothing to full over 60 s each, with normal running between them: an outer-race bearing defect (impacts ringing a 900 Hz resonance), imbalance (the shaft-speed component grows to three times its normal size) and mechanical looseness (half-order and high-order harmonics).
- For comparison, the same trainer with feature weighting off (a plain Euclidean distance) learns alongside, and an `int8::QDiagGaussianAnomaly` with one component is fitted on the host to the same learning frames (`calibration::build_diag_gaussian_anomaly`). All three are thresholded the same way. Every step up to the k-means score is integer arithmetic.

## Run it

```bash
cargo run --release --features std --example vibration_anomaly -- output
```

It writes the same files the C++ example writes, byte for byte, and the same report. One line of the report differs by design: the trainer takes 1,248 bytes of RAM here and 1,264 in the C++, where the field order needs more padding.

| File | Contents |
|---|---|
| `vibration_anomaly.csv` | Every test frame's three scores and thresholds |
| `vibration_anomaly_summary.csv` | Per detector and condition: frames, AUC, alarm rate and the severity at the first sustained alarm |

## Output

![Anomaly score over threshold through the test run, alarm rates per fault, and the severity at the first sustained alarm]({{ site.baseurl }}/assets/plots/vibration_anomaly.png)

Learned from 2,998 frames, the weighted k-means raises false alarms on 0.17% of 3,000 normal frames. It flags the bearing defect on 92.9% of its frames (AUC 0.985) and sustains an alarm from 8% severity, imbalance on 69.0% (AUC 0.943) from 36%, and looseness on 84.5% (AUC 0.961) from 16%. The Euclidean k-means never alarms on imbalance (AUC 0.761): that fault changes a few low-frequency bands, and an unweighted distance over 32 features loses the change among bands that vary a lot in normal running. With the weights, a band that is normally steady counts for more when it moves. The host-fitted Gaussian performs like the weighted k-means. The on-device state is the 1,248-byte trainer and the 1,028-byte log-mel workspace.

The figure is drawn from this port's output with tinymind (C++)'s plotting style (`scripts/plot-examples.py`).

[Source on GitHub](https://github.com/danmcleran/tinymind-rs/blob/master/examples/vibration_anomaly.rs)
