# Presentation Speaker Notes (Slide-by-Slide)

Use this as a narration script for your deck.

---

## Slide 1: Title & Goal

### On slide
- **Condition-Aware ICU Early Warning System**
- Goal: Predict short-term deterioration risk with condition-specific context

### Speaker notes
- "Today I will present our condition-aware ICU risk framework built on MIMIC-IV."
- "The core idea is to replace one-size-fits-all alerts with horizon-specific, condition-aware risk estimates."
- "We focus on 1h, 2h, and 3h lead-time risk prediction."

---

## Slide 2: Clinical Problem

### On slide
- Generic thresholds cause alarm fatigue and false positives
- Same vital value can mean different risk by diagnosis

### Speaker notes
- "A heart rate of 110 means very different things in sepsis versus stable chronic disease."
- "Our model adds diagnosis context and time-window dynamics so alerts are more clinically meaningful."

---

## Slide 3: Dataset & Scope

### On slide
- Source: **MIMIC-IV v3.1 (PhysioNet)**
- ICU stays in rolling dataset: **55,156**
- Rolling rows: **2,042,802** across first 48h

### Speaker notes
- "We use a real, publicly available ICU dataset from PhysioNet under credentialed access."
- "Our rolling table has over 2 million hourly records across ~55k ICU stays."
- "Conditions modeled: sepsis, heart failure, CKD, diabetes."

---

## Slide 4: Features, Labels, and Preprocessing

### On slide
- Inputs: HR, SBP, MAP, RR, SpO2, Temp + condition context
- Rolling features: 1h/3h/5h means and 5h slopes
- Lead labels: `target_event_in_1h`, `2h`, `3h`

### Speaker notes
- "We build hourly features from ICU admission hour 0 to 47."
- "Preprocessing includes ICU alignment, forward/backward fill, and median imputation in modeling pipelines."
- "Condition priority is sepsis > heart failure > CKD > diabetes > other for a primary condition input."
- "Train/test splitting is done by stay_id to reduce leakage."

---

## Slide 5: EDA Highlights

### On slide
- Severe imbalance for lead-time events:
  - 1h positive rate: **0.458%** (training window)
  - 2h positive rate: **0.429%**
  - 3h positive rate: **0.400%**
- Missingness very low after engineering (<0.02% on top columns)

### Speaker notes
- "Lead-time tasks are rare-event prediction problems, so AUPRC is especially important."
- "Most missingness is already handled upstream and in-model imputation."
- "We rely on robust quantiles for vitals interpretation because means are inflated by outliers."

---

## Slide 6: Model Candidates

### On slide
- Logistic Regression (interpretable baseline)
- LSTM (sequence baseline)
- Calibrated HistGradientBoosting (HGB)

### Speaker notes
- "We compared classical linear baseline, deep sequence baseline, and boosted trees."
- "For deployment, we also train lead-specific rolling boosted models at 1h, 2h, 3h."

---

## Slide 7: Performance Comparison (Main)

### On slide
Show: `evaluation/artifacts/lr_lstm_boosted_performance_comparison.png`

### Speaker notes
- "Calibrated HGB is currently best overall on tabular mortality-family comparison."
- "Metrics:"
  - "HGB: ROC-AUC 0.7793, AUPRC 0.4547, Brier 0.1081"
  - "LSTM: ROC-AUC 0.7622, AUPRC 0.4483, Brier 0.1117"
  - "Logistic: ROC-AUC 0.7136, AUPRC 0.3272, Brier 0.2135"
- "So LSTM is competitive, but boosted still edges it out in this setting."

---

## Slide 8: Lead-Time Results (Rolling)

### On slide
Show:
- `deployment/artifacts/deployed_model_metrics.png`
- `evaluation/artifacts/rolling_model_metrics_comparison.png`

### Speaker notes
- "Lead 1h means predicting event risk one hour ahead; similarly for 2h and 3h."
- "Boosted model by lead ROC-AUC: 0.8898 (1h), 0.7741 (2h), 0.7590 (3h)."
- "GRU baseline underperforms boosted at every lead in current experiments."
- "This is why HGB is deployed now while sequence models remain challengers."

---

## Slide 9: Proposed Deployment Architecture

### On slide
- Cloud-first API + frontend
- Lead-specific HGB models (1h/2h/3h)
- Monitoring + drift + retraining loop

### Speaker notes
- "Current serving stack is FastAPI + Dockerized deployment."
- "HGB is favorable for latency, compute cost, and easier operational scaling."
- "Recommended strategy: keep GRU in shadow mode for periodic re-evaluation."

---

## Slide 10: Risks, Ethics, and Roadmap

### On slide
- Risks: imbalance, drift, overfitting to surrogate event definition
- Ethics: subgroup fairness, alarm burden, generalization limits
- Next steps: external validation, fairness reporting, calibration governance

### Speaker notes
- "We should treat this as decision support, not autonomous diagnosis."
- "Roadmap: robust QA, temporal validation, subgroup fairness checks, pilot deployment, then iterative governance."
- "Final goal is clinically useful alerts with lower noise and better early detection."

---

## Optional Q&A backup points

- **Why not only LSTM?**
  - In current runs, boosted has better discrimination and precision-recall; lower operational overhead.
- **What does Brier score mean?**
  - It is mean squared probability error; lower is better calibrated probability quality.
- **Can lead-time models run live?**
  - Yes. Current deployment supports `lead_hours` = 1, 2, 3 via API.
