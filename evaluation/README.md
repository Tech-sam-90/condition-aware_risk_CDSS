# Evaluation

This folder contains evaluation scripts, metrics, and validation outputs for model performance.

## Scripts

- `evaluate_calibration_and_explainability.py`: generates reliability diagram and explainability artifacts.
- `plot_rolling_metrics_and_vitals.py`: generates rolling model metric plots and 6-vital time-series plots for the 4 conditions.

## Run

```bash
python evaluation/evaluate_calibration_and_explainability.py
python evaluation/plot_rolling_metrics_and_vitals.py
```

## Outputs

Saved under `evaluation/artifacts/`:

- `reliability_diagram.png`
- `permutation_importance_boosted.csv`
- `logistic_coefficients.csv`
- `rolling_model_metrics_comparison.png`
- `vitals_timeseries_6_vitals_4_conditions.png`
