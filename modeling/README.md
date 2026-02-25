# Modeling

This folder contains model training code for rule-based, statistical, and planned deep learning approaches.

## Scripts

- `train_calibrated_condition_models.py`: trains logistic, boosted, and isotonic-calibrated boosted models.
- `predict_risk.py`: command-line condition + vitals risk inference using calibrated boosted model.
- `train_lstm_timeseries.py`: optional deep-learning baseline using first-24h hourly vitals sequences.

## Run

```bash
python modeling/train_calibrated_condition_models.py

python modeling/predict_risk.py \
	--condition sepsis \
	--heart_rate 110 \
	--sbp 95 \
	--map 62 \
	--resp_rate 24 \
	--spo2 93 \
	--temp_f 100.4

# Optional deep-learning baseline (requires tensorflow)
python modeling/train_lstm_timeseries.py
```

## Outputs

Saved under `modeling/artifacts/`:

- `model_metrics.csv`
- `condition_thresholds.csv`
- `evaluation_predictions.csv`
- `*.pkl` model files
- `metadata.json`
- `lstm_24h_vitals.keras` (optional)
- `lstm_model_metrics.csv` (optional)
