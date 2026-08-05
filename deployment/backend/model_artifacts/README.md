# Deployment Model Artifacts

This directory is managed by model-selection scripts.

Expected contents after running:

- `python modeling/train_all_models.py`

Artifacts created:

- `selected_sequence/*.keras` (best model staged for lead 1h/2h/3h)
- `model_selection.json` (selection metadata)

If `selected_sequence` is missing, backend falls back to:

- `/app/model_artifacts/rolling_lstm_event`
