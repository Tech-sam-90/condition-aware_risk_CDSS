# Model, Data, and Deployment Brief

This brief reflects the final handoff package in `Complete Folder/`.

Last refreshed: **2026-03-08**

## Executive Summary

- Final deployed model family: `rolling_lstm_event` (one Keras model per lead horizon).
- Deployment stack: FastAPI backend + Docker image, loading artifacts from `models/deployed/rolling_lstm_event/`.
- Supported horizons: 1h, 2h, 3h ahead event risk.
- Supported conditions in API: `sepsis`, `heart_failure`, `ckd`, `diabetes`.

## Data and Preprocessing Snapshot

### Source and scope

- Dataset source: MIMIC-IV v3.1 (PhysioNet, credentialed access).
- Rolling data table: `2,042,802` hourly rows over first 48h from ICU admission.
- Unique ICU stays in rolling data: `55,156`.
- Condition context used in modeling: sepsis, heart failure, CKD, diabetes, other (encoded in training data).

### Core model inputs

- Vitals: `heart_rate`, `sbp`, `map`, `resp_rate`, `spo2`, `temp_f`.
- Sequence window length at inference: `24` time steps (`SEQUENCE_LENGTH`, default 24).
- Labels: `target_event_in_1h`, `target_event_in_2h`, `target_event_in_3h`.

### Class imbalance context

- Lead-time event targets are rare (sub-1 percent prevalence), so AUPRC remains the key operating metric.
- Resampling controls used for final LSTM-event training:
  - `LSTM_EVENT_RESAMPLE_ENABLED=1`
  - `LSTM_EVENT_TARGET_MINORITY_RATIO=0.15`
  - `LSTM_EVENT_POSITIVE_OVERSAMPLE_MULTIPLIER=2.0`

## Final Model and Benchmarks

### Final deployed model

- Model family: `sequence_lstm_event`.
- Artifacts:
  - `models/deployed/rolling_lstm_event/lstm_event_lead_1h.keras`
  - `models/deployed/rolling_lstm_event/lstm_event_lead_2h.keras`
  - `models/deployed/rolling_lstm_event/lstm_event_lead_3h.keras`
  - `models/deployed/rolling_lstm_event/metrics_by_lead.csv`

### Deployed LSTM-event metrics by lead

Source: `models/deployed/rolling_lstm_event/metrics_by_lead.csv`

| Lead | ROC-AUC | AUPRC | Brier |
| --- | --- | --- | --- |
| 1h | 0.7343 | 0.00609 | 0.01884 |
| 2h | 0.7300 | 0.00574 | 0.01948 |
| 3h | 0.7445 | 0.00620 | 0.02879 |

### Sequence model comparison (presentation context)

Source: combined from `models/deployed/rolling_lstm_event/metrics_by_lead.csv`,
`models/deployed/rolling_boosted/metrics_by_lead.csv`, and
`models/benchmarks/rolling_sequence_metrics_by_lead.csv`

| Lead | LSTM-event ROC-AUC | GRU ROC-AUC | Boosted ROC-AUC |
| --- | --- | --- | --- |
| 1h | 0.7343 | 0.7282 | 0.8892 |
| 2h | 0.7300 | 0.7049 | 0.7753 |
| 3h | 0.7445 | 0.7036 | 0.7616 |

Interpretation for presentation:
- LSTM-event is a consistent improvement over GRU at all leads.
- Boosted remains a strong tabular benchmark.
- This final package deploys LSTM-event as the selected sequence-first production candidate for the presentation milestone.

### Mortality-family benchmark context

Source: `models/benchmarks/model_metrics.csv`, `models/benchmarks/lstm_model_metrics.csv`

| Model | ROC-AUC | AUPRC | Brier |
| --- | --- | --- | --- |
| HistGradientBoosting | 0.7799 | 0.4590 | 0.1079 |
| Calibrated HistGradientBoosting | 0.7781 | 0.4513 | 0.1084 |
| LSTM 24h vitals | 0.7736 | 0.4606 | 0.1086 |
| Logistic Regression | 0.7137 | 0.3270 | 0.1198 |

## Deployment Details (What to Say in the Presentation)

### Serving behavior

- Backend file: `deployment/main.py`.
- On startup, API loads 3 models from `rolling_lstm_event` by lead horizon.
- Inference path uses a 24-step repeated sequence built from current vitals + one-hot condition vector.

### API endpoints

- `GET /health`: readiness probe.
- `POST /predict`: single risk prediction.
- `WS /ws/live`: streaming prediction endpoint.

### Request schema (`POST /predict`)

- `condition`: one of `sepsis`, `heart_failure`, `ckd`, `diabetes`
- `lead_hours`: 1, 2, or 3
- `heart_rate`, `sbp`, `map`, `resp_rate`, `spo2`, `temp_f`

### Risk banding logic

- `HIGH_RISK_THRESHOLD` default is `0.02`.
- Banding in API:
  - `high` if `p >= threshold`
  - `moderate` if `p >= 0.5 * threshold`
  - `low` otherwise

## Presentation Asset Map

Use these files directly from `Complete Folder/plots/`:

- `lr_lstm_boosted_performance_comparison.png`: tabular baseline comparison slide.
- `rolling_model_metrics_comparison.png`: lead-time model comparison slide.
- `deployed_model_metrics.png`: deployed lead-by-lead metrics slide.
- `lstm_overfitting_curve_15pct.png`: LSTM training dynamics slide.
- `gru_overfitting_curves_15pct.png`: GRU training dynamics slide.
- `overfitting_comparison_auc_15pct.png`: overfitting gap comparison slide.
- `vitals_timeseries_6_vitals_4_conditions.png`: data understanding/clinical signal slide.

## Risks and Governance Talking Points

- Rare-event prevalence keeps AUPRC low in absolute terms; thresholding must be conservative.
- A single-site dataset (MIMIC) limits external generalization.
- Alert fatigue risk requires calibration review with clinicians.
- Drift monitoring and periodic threshold re-tuning are required for production safety.

## Recommended Next Steps

1. Add temporal holdout and subgroup fairness reporting to deployment signoff.
2. Calibrate lead-specific alert thresholds against clinician workload constraints.
3. Add confidence/drift monitoring to the API telemetry pipeline.
4. Keep boosted and GRU artifacts as shadow baselines for monthly regression checks.
