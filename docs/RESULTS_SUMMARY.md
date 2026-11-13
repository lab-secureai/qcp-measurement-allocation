# Experimental results

## Factorial study

Primary protocol: unstratified calibration/test holdout, three binary tasks, seeds 50–59.

| Calibration | Test stopping | Mean shots | Mean set size |
| --- | --- | ---: | ---: |
| Rank | Decision | 77,799 | 1.2134 |
| Width | Decision | 92,513 | 1.2154 |
| Rank | Width | 125,841 | 1.2134 |
| Width | Width | 140,553 | 1.2164 |

Observed marginal coverage: 97.34% for all four methods. Mean paired savings: 14.63% for Rank + Decision versus Width + Decision; 43.64% for Rank + Decision versus Width + Width. The stratified replication (seeds 40–49) gives 15.43% for the isolated calibration comparison.

## Stationary noise

Mean paired savings for Rank + Decision versus Width + Width, across thirty dataset–seed runs per noise condition:

| Noise | Paired shot savings |
| --- | ---: |
| None | 43.01% |
| Readout bit flip, 2% | 42.33% |
| Gate depolarization | 40.64% |
| Combined, mild | 40.55% |
| Combined, strong | 32.13% |

## UCI Bank Marketing

Five seeds, five stationary noise settings, and six policies per setting.

| Noise | Paired shot savings |
| --- | ---: |
| None | 38.32% |
| Readout bit flip, 2% | 36.76% |
| Gate depolarization | 37.45% |
| Combined, mild | 37.46% |
| Combined, strong | 35.34% |

Positive-class and marginal coverage are both included in `results/bank_marketing/summary.csv`.

To verify the reported summaries:

```bash
python scripts/verify_release.py
```
