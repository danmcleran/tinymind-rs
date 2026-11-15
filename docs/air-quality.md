---
title: Air quality (LSTM forecast)
layout: default
nav_order: 10
---

# Air quality forecast example

The `air_quality` example is a port of tinymind (C++)'s [air quality example](https://danmcleran.github.io/tinymind/examples/air_quality.html): an LSTM network that forecasts the next hour's pollutant concentration from the current hour, trained entirely in Q16.16 fixed point. Where the [activity recognition example]({{ site.baseurl }}/har) classifies a whole window, this one predicts a value at every step, the shape of a sensor forecaster.

## How it works

- An LSTM network with 1 input, 16 LSTM cells (tanh) and 1 sigmoid output, gradients clipped to [-1, 1], learning rate 0.5.
- The series is 2,400 hours of a synthetic carbon-monoxide-like signal: a daily baseline with a morning and an evening rush-hour peak, a slow weekly trend, and a little noise. It is scaled into [0.1, 0.9], inside the range where the sigmoid output still responds.
- Training runs over the first 2,200 hours, predicting each hour from the one before. The recurrent state is reset every 24 hours, so each stretch the network learns from is one day.
- A sample whose error is within 1/128 of full scale is not trained on. Training stops at the first epoch in which no sample trains: the weights did not move in it, and since the state is reset at the start of every epoch, any later epoch would repeat it exactly. The limit is 15,000 epochs; on the synthetic series training stops after 28.
- The last 200 hours are held out. After warming the state up on the day before them, the network forecasts each held-out hour from the true value of the hour before it (one step ahead, not running free on its own predictions).

## Run it

```bash
cargo run --release --example air_quality -- output
```

It takes well under a second. Two optional arguments after the output directory set the epoch limit and the learning rate, as in the C++. If the output directory holds `AirQualityUCI.csv` from the [UCI Air Quality dataset](https://archive.ics.uci.edu/dataset/360/air+quality), the example trains on its CO(GT) column instead of the synthetic series, as the C++ does. The program writes the same files the C++ example writes, byte for byte:

| File | Contents |
|---|---|
| `airq_loss.csv` | Mean absolute error per epoch |
| `airq_forecast.csv` | True and predicted concentration for each held-out hour |

## Output

![Air-quality training loss and one-step forecast over the held-out hours]({{ site.baseurl }}/assets/plots/air_quality.png)

The mean error per step falls from 0.0151 in the first epoch to 0.00156 in the 28th, in which every error is inside the band, so training stops there. Over the 200 held-out hours the forecast follows both daily peaks with a mean absolute error of 0.070, on a series that ranges from 1.34 to 3.01; its largest misses are at the sharpest peaks and troughs.

The figure is drawn from this port's output with tinymind (C++)'s plotting style (`scripts/plot-examples.py`).

[Source on GitHub](https://github.com/danmcleran/tinymind-rs/blob/master/examples/air_quality.rs)
