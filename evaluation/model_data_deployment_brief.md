# Model, Data, and Deployment Brief

This document summarizes the current project status for presentation.

## Data & Preprocessing

### Dataset source (real, publicly available)
- **MIMIC-IV v3.1** (PhysioNet), a real-world ICU EHR dataset.
- Publicly available via PhysioNet under credentialed access and data use agreement.

### Data characteristics

#### Size
- `data/processed/condition_model_table_v1.csv`: **55,203 rows x 14 columns**
- `data/processed/condition_model_table_v2_with_24h_vitals.csv`: **55,203 x 49**
- `data/processed/hourly_vitals_48h.csv`: **2,042,802 x 34**
- `data/processed/rolling_window_multicondition_timeseries.csv`: **2,042,802 x 39**
- Unique ICU stays in rolling data: **55,156**
- Hour range in rolling data: **0 to 47 hours from ICU admission**

#### Features
- Core vitals: `heart_rate`, `sbp`, `map`, `resp_rate`, `spo2`, `temp_f`
- Rolling features (24 total):
  - means over 1h/3h/5h
  - 5h slopes
- Derived feature: `hr_minus_map`
- Condition context (`condition_input` or encoded `condition_code`)
- Deployed rolling boosted models use **32 features** (see `modeling/artifacts/rolling_boosted/metadata.json`).

#### Target variable
- For rolling lead-time models:
  - `target_event_in_1h`
  - `target_event_in_2h`
  - `target_event_in_3h`
- For earlier mortality model family:
  - `hospital_expire_flag`

### Initial EDA insights

#### Class imbalance
- On full rolling table (`2,042,802` rows):
  - `target_event_in_1h`: **9,998 positives** (0.489%)
  - `target_event_in_2h`: **9,471 positives** (0.464%)
  - `target_event_in_3h`: **8,893 positives** (0.435%)
  - `hospital_expire_flag`: **319,876 positives** (15.66%)
- On training window used by rolling models (`hour_from_icu` 4..44, `1,780,256` rows):
  - `target_event_in_1h`: **8,152 positives** (0.458%)
  - `target_event_in_2h`: **7,639 positives** (0.429%)
  - `target_event_in_3h`: **7,116 positives** (0.400%)
  - `hospital_expire_flag`: **275,284 positives** (15.46%)

#### Missing values
- Very low missingness after feature engineering and fill operations.
- Highest observed rates in training window are still low:
  - `temp_f` and temp rolling means: ~0.0149%
  - `map` and map rolling means: ~0.0085%
  - `sbp` and sbp rolling means: ~0.0062%

#### Distributions
- Robust quantiles in training window show clinically plausible ranges:
  - `heart_rate` median: **84** (q01=50, q99=134)
  - `sbp` median: **114** (q01=74, q99=177)
  - `map` median: **75** (q01=46, q99=122)
  - `resp_rate` median: **19** (q01=8, q99=36)
  - `spo2` median: **97** (q01=87, q99=100)
  - `temp_f` median: **98.2F** (q01=95.7, q99=101.6)
- Mean/std are inflated for some variables due to extreme outliers in raw charted values, so median/IQR/quantiles are preferred for interpretation.

### Preprocessing steps

#### Cleaning
- ICD-based cohort creation and condition mapping in `data_processing/prepare_condition_cohort.py`.
- Condition priority assignment: sepsis > heart_failure > ckd > diabetes > other.
- Vitals extraction from `chartevents` by specific ITEMIDs and ICU-relative time windows.

#### Scaling/encoding
- Logistic model (`train_calibrated_condition_models.py`):
  - Numeric: median imputation + standard scaling
  - Categorical (`condition_input`): most-frequent imputation + one-hot encoding
- Boosted model (same script):
  - Numeric: median imputation
  - Categorical: one-hot encoding
  - No scaling required for tree boosting
- Rolling boosted (`train_rolling_boosted_models.py`):
  - Median imputation in pipeline
  - Condition encoded as integer `condition_code`

#### Train-test split
- Mortality tabular model (`train_calibrated_condition_models.py`):
  - 80/20 stratified split by label (`hospital_expire_flag`)
- Rolling lead-time boosted (`train_rolling_boosted_models.py`):
  - Split by **stay_id** (80/20), stratified on stay-level target presence per lead horizon
- Rolling GRU (`train_rolling_sequence_models.py`):
  - Also split by **stay_id**, stratified at stay level
- LSTM baseline (`train_lstm_timeseries.py`, updated):
  - Per-stay 24h sequences, 80/20 stratified split on mortality label

## Proposed ML/DL Architecture & Deployment Plan

### Industry-deployable architecture (recommended)

**Primary production model:** calibrated HistGradientBoosting (HGB), one model per lead horizon (1h/2h/3h).

Why this is currently the best production candidate:
- **Accuracy:** outperforms GRU in rolling lead-time benchmarks.
- **Latency:** lightweight CPU inference for tabular input; no sequence-state handling at serving time.
- **Compute cost:** low compared with recurrent DL models.
- **Storage:** compact artifacts.
  - `calibrated_boosted_lead_1h.pkl`: 780,642 bytes
  - `calibrated_boosted_lead_2h.pkl`: 703,283 bytes
  - `calibrated_boosted_lead_3h.pkl`: 535,339 bytes
- **Scalability:** stateless FastAPI service, easy horizontal scaling.
- **Ethical risk control:** simpler model behavior and easier threshold governance than opaque deep sequence models.

### Performance snapshot

#### Main tabular comparison (mortality family)
- Calibrated HGB: ROC-AUC 0.7793, AUPRC 0.4547, Brier 0.1081
- LSTM (24h vitals): ROC-AUC 0.7622, AUPRC 0.4483, Brier 0.1117
- Logistic Regression: ROC-AUC 0.7136, AUPRC 0.3272, Brier 0.2135

#### Rolling lead-time comparison (event prediction)
- Boosted (lead 1h/2h/3h ROC-AUC): 0.8898 / 0.7741 / 0.7590
- GRU (lead 1h/2h/3h ROC-AUC): 0.6382 / 0.6148 / 0.6173

### Deployment target

**Recommended target: Hybrid cloud-first**
- **Cloud backend** for model hosting, calibration updates, monitoring, and retraining orchestration.
- **Edge/client bedside app** for UI, local buffering, and fallback rule-based alerts if connectivity is intermittent.

Current implementation is cloud/container style:
- FastAPI backend + static frontend
- Dockerized deployment with stack orchestration under `deployment/`.

### High-level system diagram

```mermaid
flowchart TD
    A[Bedside Monitors / ICU Charting] --> B[Data Ingestion + Validation]
    B --> C[Feature Builder\nHourly + Rolling Features]
    C --> D{Lead Horizon}
    D --> D1[HGB Lead 1h]
    D --> D2[HGB Lead 2h]
    D --> D3[HGB Lead 3h]
    D1 --> E[Calibrated Risk Probability]
    D2 --> E
    D3 --> E
    E --> F[Thresholding + Risk Banding]
    F --> G[Clinician UI / Alerts API]
    E --> H[Monitoring + Drift Dashboard]
    H --> I[Retraining Pipeline]
```

## Challenges & Next Steps

### Technical risks
- Severe class imbalance for lead-time targets (<0.5% positives) can destabilize rare-event performance.
- Potential overfitting to surrogate event definition (`zero_hour` logic).
- Sequence models currently underperforming; need architecture and objective tuning.
- Outlier-prone vitals distributions require robust handling and possibly winsorization/physiology constraints.

### Deployment risks
- Real-time latency spikes under high concurrency if preprocessing grows.
- Model drift from changing patient mix or workflow.
- Data quality drift (missing streams, sensor artifacts, mapping changes).

### Ethical concerns
- Single-center source bias (MIMIC) may limit external generalization.
- Potential subgroup performance disparities by age/sex/condition severity.
- Alert burden can worsen alarm fatigue if thresholds are not carefully tuned with clinicians.

### Roadmap to final project

1. **Finalize dataset QA and robust feature constraints**
   - Add physiologic clipping/winsorization, explicit outlier audits.
2. **External-style validation protocol**
   - Temporal split and subgroup fairness reporting.
3. **Production model hardening**
   - Keep HGB as primary; run GRU in shadow mode for ongoing comparison.
4. **Monitoring layer**
   - Add drift, calibration, and alert-rate dashboards.
5. **Clinical threshold calibration**
   - Tune decision thresholds per condition and workflow with stakeholders.
6. **Pilot deployment**
   - Cloud-first pilot + bedside UI; measure latency, PPV, alert acceptance.
7. **Iterative governance cycle**
   - Monthly retraining review, fairness audit, and performance regression checks.
