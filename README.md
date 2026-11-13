# Efficient Measurement Allocation for Quantum Conformal Prediction

Code and simulation results for rank-based calibration and decision-based measurement stopping in quantum conformal prediction.

## Experiments

The implementation uses a four-qubit variational classifier. Circuit training is performed with statevector simulation. Shot counts refer to conformal calibration and prediction, excluding training.

The main factorial experiment uses Breast Cancer Wisconsin, Wine (classes 0 and 1), and Digits (classes 3 and 8). Calibration and test observations are split without label stratification for seeds 50–59.

| Calibration | Test stopping | Mean shots | Mean prediction-set size |
| --- | --- | ---: | ---: |
| Rank | Decision | 77,799 | 1.2134 |
| Width | Decision | 92,513 | 1.2154 |
| Rank | Width | 125,841 | 1.2134 |
| Width | Width | 140,553 | 1.2164 |

All four policies have 97.34% observed marginal coverage. The mean paired shot reduction is 14.63% for Rank + Decision versus Width + Decision, and 43.64% for Rank + Decision versus Width + Width.

The repository also contains experiments with stationary gate and readout noise (three tasks, ten seeds, five noise settings) and the UCI Bank Marketing dataset (five seeds, five noise settings). Original run-level CSVs and figures are included.

## Installation

Python 3.10 or later is recommended.

```bash
python -m pip install -r requirements.txt
python scripts/verify_release.py
python scripts/make_figures.py
```

The verification command checks the supplied results. It does not rerun model training.

## Reproducing the main experiments

From the repository root:

```bash
python src/run_factorial.py --start 50 --n-seeds 10 --epochs 130 --split-policy random_holdout
python src/run_factorial.py --start 40 --n-seeds 10 --epochs 130 --split-policy original_stratified
```

The calibration width threshold is 0.18; the test width threshold is 0.12. The target marginal coverage is 0.90, the confidence failure budget is 0.01, and the shot checkpoints are 8, 16, 32, 64, 128, 256, 512, and 1024.

For stationary noise:

```bash
python src/run_application_noise.py --datasets breast wine01 digits38 --seeds 60 61 62 63 64 65 66 67 68 69 --epochs 130 --output runs/noise
```

For Bank Marketing:

```bash
python src/run_application_noise.py --datasets bank --seeds 60 61 62 63 64 --epochs 130 --bank-size 1800 --bank-required --output runs/bank
```

The Bank experiment downloads the UCI archive. A local copy can be specified using `--bank-local /path/to/bank+marketing.zip`. The `duration` variable is excluded before preprocessing.

## Colab notebooks

The notebooks contain readable source code in separate cells and can be run independently using **Runtime → Run all**:

- `notebooks/QCP_Factorial_Colab.ipynb`
- `notebooks/QCP_Application_and_Noise_Colab.ipynb`
- `notebooks/QCP_Adaptive_Baselines_V4_Colab.ipynb`

## Repository layout

```text
src/                 Model, measurement allocation, and experiment scripts
notebooks/           Standalone Google Colab notebooks
results/factorial/   Factorial comparisons for two holdout protocols
results/width_baselines/
                     Confidence-width baselines
results/noise/       Stationary noise experiments
results/bank_marketing/
                     Bank Marketing experiments
figures/             Plots from saved experiment results
scripts/             Result verification and figure generation
docs/                Protocol and data provenance
```

## Measurement and statistical scope

The reported savings measure simulated circuit shots during calibration and prediction. The finite-sample conformal guarantee is marginal and assumes exchangeability and valid simultaneous confidence intervals. The 64-shot conventional QCP comparator targets finite-shot scores, while certified policies target the underlying stationary readout probabilities. Class-conditional coverage and hardware runtime are not evaluated.

## Data and citation

The Bank Marketing data are obtained from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) and are not redistributed. The experiment records the source-file SHA-256 in `results/bank_marketing/RUN_METADATA.json`.

No open-source license has been selected. Please add a license, the authors' information, and the paper citation before publication.
