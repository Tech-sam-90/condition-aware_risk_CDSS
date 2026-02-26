# Feature Engineering

This folder contains code for creating context-aware features from vitals, diagnoses, labs, and treatments.

## Scripts

- `build_24h_vitals_features.py`: builds first-24h vital features from `chartevents` using DuckDB filtered scan.
- `build_rolling_window_timeseries.py`: builds hourly (0-48h) vitals, rolling-window features, and lead-time event targets (`1h/2h/3h`).

## Run

```bash
python feature_engineering/build_24h_vitals_features.py

python feature_engineering/build_rolling_window_timeseries.py
```

## Rolling outputs

Saved under `data/processed/`:

- `hourly_vitals_48h.csv`
- `rolling_window_multicondition_timeseries.csv`
