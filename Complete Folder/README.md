# Complete Folder

This folder contains the current end-to-end assets used in this project.

## What is included
- `datasets/`
  - `rolling_window_multicondition_timeseries.csv`
  - `condition_model_table_v2_with_24h_vitals.csv`
- `models/deployed/rolling_boosted/`
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
- These are symlinks to source-of-truth files in the repository, so they always reflect the latest generated outputs.

## Deployment model source
- The backend loads from `modeling/artifacts/rolling_boosted/`.
- The backend image now copies only `modeling/artifacts/rolling_boosted` into `/app/model_artifacts/rolling_boosted`.

## To deploy the latest model
From `deployment/`:

```bash
docker compose --env-file .env -f compose.build.yml build risk-api
docker compose --env-file .env -f compose.build.yml push risk-api  # optional
docker stack deploy --with-registry-auth --compose-file docker-stack.yml ${STACK_NAME}
```
