---
title: Log-mel front end
layout: default
nav_order: 12
---

# Log-mel front end example

The `logmel_vibration` example is a port of tinymind (C++)'s [log-mel vibration example](https://danmcleran.github.io/tinymind/examples/logmel_vibration.html): the fixed-point log-mel spectrogram (`int8::QLogMelSpectrogram`) and MFCCs (`int8::QMfcc`) of a machine-vibration recording, compared frame by frame with the float definition they implement. This is the feature front end a sensor node runs before a classifier or an [anomaly detector]({{ site.baseurl }}/vibration).

## How it works

- The recording is synthetic: 12 seconds at 3,200 samples per second, int16. A motor at 29.5 Hz with four harmonics and broadband noise; halfway through, a bearing fault starts, impacts at 107 Hz ringing a 900 Hz structural resonance. The level ramps from -50 dBFS to -6 dBFS across the run, so the front end sees quiet frames and loud ones.
- Each 256-sample frame (hop 128) goes through `QLogMelSpectrogram` with a Hann window and 32 HTK mel bands from 5 to 1,600 Hz, then `QMfcc` with 13 coefficients. All of it is integer arithmetic: the window, a per-frame block scale that brings every frame to the FFT at full scale, the real FFT (`QRFft1D`), the mel filters and a table-based natural log, with the output in Q8 nats.
- The reference computes the same definition in `f64`: the natural log of the mel-weighted power spectrum of the windowed frame (torchaudio's `MelSpectrogram` followed by a log), then the orthonormal DCT.
- The window, the filters and the DCT come from the host-side builders in `int8::calibration` (`build_hann_window`, `build_mel_filterbank`, `build_dct_matrix`).

## Run it

```bash
cargo run --release --features std --example logmel_vibration -- output
```

It writes the same files the C++ example writes, byte for byte:

| File | Contents |
|---|---|
| `logmel_vibration.csv` | Per frame: time, fault flag, level, then the 32 fixed-point and 32 float log-mel bands |
| `logmel_vibration_mfcc.csv` | Per frame: the 13 fixed-point and float MFCCs |
| `logmel_vibration_error.csv` | The absolute error by how far a band sits below the frame's loudest band |

## Output

![Float and fixed-point log-mel spectrograms of the vibration recording, their difference, and the error by band level]({{ site.baseurl }}/assets/plots/logmel_vibration.png)

Over 299 frames the bands within 10 dB of a frame's loudest band are within 0.0055 nats of the float reference (mean 0.0012); the error grows only for bands far below the peak, where the 16-bit FFT's rounding floor is reached: a mean of 0.016 nats 30 to 40 dB down and at most 0.29 nats beyond 40 dB. The worst MFCC error against the float DCT of the float bands is 0.13. The fixed-point frame needs a 1,028-byte workspace.

The figure is drawn from this port's output with tinymind (C++)'s plotting style (`scripts/plot-examples.py`).

[Source on GitHub](https://github.com/danmcleran/tinymind-rs/blob/master/examples/logmel_vibration.rs)
