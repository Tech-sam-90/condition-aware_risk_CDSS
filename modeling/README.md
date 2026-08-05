# Modeling

The `modeling/` folder now exposes a single executable modeling entry point:

- `train_all_models.py`

All model artifacts are written under:

- `modeling/artifacts/`

## Train All Models

Run the unified training pipeline:

```bash
python modeling/train_all_models.py
```

Optional flags:

```bash
# Skip optional 24h LSTM baseline
python modeling/train_all_models.py --skip-optional-lstm

# Rebuild sequence comparison from existing metrics only
python modeling/train_all_models.py --skip-sequence-compare-train

# Override shared imbalance policy across model families
python modeling/train_all_models.py --minority-ratio 0.15 --oversample-multiplier 2.0
```

## Prediction Mode

Run inference using the calibrated boosted model:

```bash
python modeling/train_all_models.py --predict --condition sepsis --heart-rate 110 --sbp 95 --map 62 --resp-rate 24 --spo2 93 --temp-f 101.2
```

Supported conditions:

- `sepsis`
- `heart_failure`
- `ckd`
- `diabetes`

## Notes

- Training implementations are intentionally moved to `training_core/` to keep `modeling/` clean and single-entry.
- Deployment model selection metadata is written to `deployment/backend/model_artifacts/model_selection.json`.
