# Evaluation

This folder contains evaluation scripts, metrics, and validation outputs for model performance.

## Scripts

- `evaluate_calibration_and_explainability.py`: generates reliability diagram and explainability artifacts.

## Run

```bash
python evaluation/evaluate_calibration_and_explainability.py
```

## Outputs

Saved under `evaluation/artifacts/`:

- `reliability_diagram.png`
- `permutation_importance_boosted.csv`
- `logistic_coefficients.csv`
