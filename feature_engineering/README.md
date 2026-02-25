# Feature Engineering

This folder contains code for creating context-aware features from vitals, diagnoses, labs, and treatments.

## Scripts

- `build_24h_vitals_features.py`: builds first-24h vital features from `chartevents` using DuckDB filtered scan.

## Run

```bash
python feature_engineering/build_24h_vitals_features.py
```
