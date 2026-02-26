# Modeling

This folder contains model training code for rule-based, statistical, and planned deep learning approaches.

## Scripts

- `train_calibrated_condition_models.py`: trains logistic, boosted, and isotonic-calibrated boosted models.
- `predict_risk.py`: command-line condition + vitals risk inference using calibrated boosted model.
- `train_lstm_timeseries.py`: optional deep-learning baseline using first-24h hourly vitals sequences.
- `train_rolling_boosted_models.py`: trains calibrated boosted lead-time models (`1h/2h/3h`) on rolling-window features.
- `train_rolling_sequence_models.py`: trains GRU lead-time models (`1h/2h/3h`) on hourly sequences.

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

# Rolling-window baseline
python modeling/train_rolling_boosted_models.py

# Rolling-window sequence challenger (requires tensorflow)
python modeling/train_rolling_sequence_models.py
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

Rolling-window artifacts:

- `modeling/artifacts/rolling_boosted/metrics_by_lead.csv`
- `modeling/artifacts/rolling_boosted/calibrated_boosted_lead_{1,2,3}h.pkl`
- `modeling/artifacts/rolling_sequence/metrics_by_lead.csv`
- `modeling/artifacts/rolling_sequence/gru_lead_{1,2,3}h.keras`
