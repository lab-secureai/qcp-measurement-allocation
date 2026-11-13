# Data sources

Breast Cancer Wisconsin Diagnostic, Wine, and Digits are loaded from `sklearn.datasets`. Wine uses classes 0 and 1; Digits uses classes 3 and 8.

The Bank Marketing data are downloaded from the [UCI repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) (dataset 222) as `bank+marketing.zip`. The implementation reads `bank-full.csv`, checks for 45,211 records, and excludes `duration` before preprocessing.

Only aggregate and run-level experiment outputs are included. No original Bank Marketing records are distributed in this repository. The source-file checksum for the reported experiment is stored in `results/bank_marketing/RUN_METADATA.json`.

The supplied CSVs can be checked with `python scripts/verify_release.py`. Full reproduction runs the experiment scripts and downloads Bank Marketing if a local copy is not provided.
