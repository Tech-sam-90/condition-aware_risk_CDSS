# Complete Folder

This folder contains the current end-to-end assets used in this project.

## What is included
- `datasets/`
  - `rolling_window_multicondition_timeseries.csv`
  - `condition_model_table_v2_with_24h_vitals.csv`
- `models/deployed/rolling_lstm_event/`
  - `lstm_event_lead_1h.keras`
  - `lstm_event_lead_2h.keras`
  - `lstm_event_lead_3h.keras`
  - `metrics_by_lead.csv`
- `models/deployed/rolling_boosted/` (historical baseline)
  - `calibrated_boosted_lead_1h.pkl`
  - `calibrated_boosted_lead_2h.pkl`
  - `calibrated_boosted_lead_3h.pkl`
  - `metadata.json`
  - `metrics_by_lead.csv`
- `models/benchmarks/`
  - `model_metrics.csv`
  - `lstm_model_metrics.csv`
  - `rolling_sequence_metrics_by_lead.csv`
- `plots/`
  - `lstm_overfitting_curve_15pct.png`
  - `gru_overfitting_curves_15pct.png`
  - `overfitting_comparison_auc_15pct.png`
  - `deployed_model_metrics.png`
  - `lr_lstm_boosted_performance_comparison.png`
  - `rolling_model_metrics_comparison.png`
  - `vitals_timeseries_6_vitals_4_conditions.png`
- `docs/`
  - `model_data_deployment_brief.md`
  - `presentation_speaker_notes.md`
- `deployment/`
  - backend and stack files currently used for deployment

## Important note
- These are real copied files (not symlinks).
- This folder is a snapshot and will not auto-update when source files change.
- Re-run the packaging step when you retrain models or regenerate plots.

## Deployment model source
- The backend in `deployment/` loads from `models/deployed/rolling_lstm_event/`.
- The backend image copies `models/deployed/rolling_lstm_event` into `/app/model_artifacts/rolling_lstm_event`.

## To deploy the latest model
From `deployment/`:

```bash
docker compose --env-file .env -f compose.build.yml build risk-api
docker compose --env-file .env -f compose.build.yml push risk-api  # optional
docker stack deploy --with-registry-auth --compose-file docker-stack.yml ${STACK_NAME}
```

The API is published on port `8000` by `docker-stack.yml`.
