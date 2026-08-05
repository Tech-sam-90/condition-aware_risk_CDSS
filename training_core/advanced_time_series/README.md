# Advanced Time-Series Training Pipelines (SMOTE 25:75)

This folder provides complete training pipelines for four suggested sequence architectures,
all using a consistent preprocessing and SMOTE-based balancing strategy.

## Models included

- `train_tcn_smote.py`
- `train_bilstm_attention_smote.py`
- `train_transformer_encoder_smote.py`
- `train_tft_style_smote.py` (TFT-style approximation in Keras)

## Imbalance strategy

All scripts use SMOTE on the training split only with a target class ratio of:
- minority:majority = `25:75`
- equivalent `sampling_strategy = 1/3`

Implementation notes:
- stay-level train/val/test split is done before any resampling
- sequences are flattened for SMOTE and reshaped back to `(seq_len, features)`
- validation and test sets are never resampled

## Dependencies

Required:
- `tensorflow`
- `scikit-learn`
- `imbalanced-learn`
- `pandas`, `numpy`

Install example:

```bash
pip install tensorflow scikit-learn imbalanced-learn pandas numpy
```

## Run examples

```bash
python training_core/advanced_time_series/train_tcn_smote.py
python training_core/advanced_time_series/train_bilstm_attention_smote.py
python training_core/advanced_time_series/train_transformer_encoder_smote.py
python training_core/advanced_time_series/train_tft_style_smote.py
```

Optional common args:

```bash
--seq-len 24 --epochs 12 --batch-size 256 --learning-rate 1e-3 --seed 42
```

## Outputs

Each script saves artifacts to:

- `modeling/artifacts/advanced_time_series/<model_name>/metrics_by_lead.csv`
- `modeling/artifacts/advanced_time_series/<model_name>/history_lead_{1,2,3}h.csv`
- `modeling/artifacts/advanced_time_series/<model_name>/predictions_lead_{1,2,3}h.csv`
- `modeling/artifacts/advanced_time_series/<model_name>/<model_name>_lead_{1,2,3}h.keras`

## Dataset used

By default all scripts read:

- `data/processed/rolling_window_multicondition_timeseries.csv`

Filtered to `hour_from_icu` in `[4, 44]` to align with existing rolling-window training.
