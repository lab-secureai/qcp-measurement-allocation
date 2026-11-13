# Experimental protocol

## Classifier and prediction sets

A parameterized four-qubit circuit supplies binary-class probabilities. The circuit parameters are trained with statevector simulation and fixed before conformal calibration. Independent Bernoulli measurement streams are sampled for calibration and test prediction.

The target marginal coverage is 90%. The simultaneous confidence failure budget is 0.01. Shot checkpoints are 8, 16, 32, 64, 128, 256, 512, and 1024.

## Factorial comparison

The primary experiment uses Breast Cancer Wisconsin, Wine 0/1, and Digits 3/8 with seeds 50–59 and an unstratified calibration/test split. A stratified replication uses seeds 40–49. The four policies cross rank-based or width-based calibration with decision-based or width-based stopping at test time. Fixed-shot and conventional QCP comparisons appear in the original width-baseline results.

## Bank Marketing

Each seed selects 1,800 observations from the 45,211-row UCI dataset. Training, calibration, and test sizes are 900, 450, and 450. The `duration` feature is omitted. No test-label balancing is performed. Bank results cover seeds 60–64, five noise settings, and six policies.

## Noise model

Gate depolarization and symmetric readout-bit flips are simulated with a four-qubit density matrix. Noise parameters remain fixed within each calibration/test run. The confidence certificates apply to the stationary noisy-readout probabilities.

## Metrics

`shots_total` equals the number of calibration and test measurement shots. `observed_coverage` is the fraction of held-out labels included in their prediction sets. `mean_set_size` averages the number of labels in each set. Positive-class coverage is reported separately for Bank Marketing.

Paired shot savings are computed for each dataset–seed run before averaging:

```text
100 * mean(1 - shots_method / shots_baseline)
```

Training shots and hardware wall-clock time are outside the measurement budget reported here.
