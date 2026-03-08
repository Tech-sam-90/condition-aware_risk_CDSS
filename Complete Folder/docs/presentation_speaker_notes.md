# Presentation Speaker Notes (Slide-by-Slide)

Use this file as the narration script for the final presentation package.

---

## Slide 1: Title and Thesis

### On slide
- **Condition-Aware ICU Early Warning System**
- Final deployed model: **Rolling LSTM-event (1h/2h/3h)**

### Speaker notes
- "This project delivers a condition-aware ICU risk API that predicts short-horizon deterioration risk."
- "The final deployment package uses an LSTM-event model family, one model per lead horizon."

---

## Slide 2: Clinical Problem

### On slide
- Generic thresholds create alert fatigue
- Same vital sign can imply different risk under different conditions

### Speaker notes
- "A single thresholding rule misses diagnosis context and temporal trend behavior."
- "Our approach combines vitals + condition context + hourly sequence dynamics."

---

## Slide 3: Dataset and Scope

### On slide
- Source: **MIMIC-IV v3.1**
- Rolling rows: **2,042,802**
- ICU stays: **55,156**
- Conditions: sepsis, heart failure, CKD, diabetes

### Speaker notes
- "We use real ICU EHR data from PhysioNet MIMIC-IV under credentialed access."
- "The modeling scope is first 48 hours from ICU admission with hourly windows."

---

## Slide 4: Features and Labels

### On slide
- Vitals: HR, SBP, MAP, RR, SpO2, Temp
- Sequence window: 24 time steps
- Labels: event in 1h, 2h, 3h

### Speaker notes
- "The deployed API accepts current vitals and condition, then forms a 24-step feature sequence for inference."
- "Lead-specific models estimate the risk of deterioration at 1h, 2h, and 3h horizons."

---

## Slide 5: Imbalance and Training Strategy

### On slide
- Rare-event task (sub-1 percent prevalence)
- Training used targeted minority balancing (15 percent)

### Speaker notes
- "Because positives are rare, AUPRC is the key metric alongside ROC-AUC and Brier score."
- "For LSTM-event training we used controlled resampling to stabilize learning under imbalance."

---

## Slide 6: Model Journey

### On slide
- Baselines: Logistic, HGB, GRU
- Final deployed model: **LSTM-event by lead horizon**

### Speaker notes
- "We evaluated both tabular and sequence families."
- "For this final delivery, we selected sequence-first deployment using the LSTM-event family."
- "Boosted and GRU are retained as comparison baselines in the package."

---

## Slide 7: Final LSTM-event Metrics

### On slide
Show: `Complete Folder/plots/deployed_model_metrics.png`

### Speaker notes
- "Final deployed LSTM-event results are:"
- "Lead 1h: ROC-AUC 0.7343, AUPRC 0.00609, Brier 0.01884."
- "Lead 2h: ROC-AUC 0.7300, AUPRC 0.00574, Brier 0.01948."
- "Lead 3h: ROC-AUC 0.7445, AUPRC 0.00620, Brier 0.02879."

---

## Slide 8: Comparison and Overfitting Checks

### On slide
Show:
- `Complete Folder/plots/rolling_model_metrics_comparison.png`
- `Complete Folder/plots/lstm_overfitting_curve_15pct.png`
- `Complete Folder/plots/overfitting_comparison_auc_15pct.png`

### Speaker notes
- "LSTM-event consistently outperforms GRU in ROC-AUC across all lead horizons."
- "Boosted remains a strong tabular benchmark and is kept for shadow comparison."
- "Overfitting curves are included to show training-validation behavior and monitoring needs."

---

## Slide 9: Deployment Architecture

### On slide
- FastAPI backend + Docker
- Artifacts loaded from `models/deployed/rolling_lstm_event/`
- Endpoints: `/health`, `/predict`, `/ws/live`

### Speaker notes
- "At startup, the service loads 3 Keras artifacts, one per lead horizon."
- "Risk banding uses a configurable high-risk threshold, default 0.02."
- "The API returns probability and a low/moderate/high risk band for decision support."

---

## Slide 10: Demo Script

### On slide
- Input: condition + vitals + lead horizon
- Output: probability + risk band

### Speaker notes
- "Example payload:"

```json
{
  "condition": "sepsis",
  "lead_hours": 1,
  "heart_rate": 112,
  "sbp": 95,
  "map": 63,
  "resp_rate": 26,
  "spo2": 92,
  "temp_f": 101.3
}
```

- "Example response includes `risk_probability`, `high_risk_threshold`, and `risk_band`."

---

## Slide 11: Risks and Guardrails

### On slide
- Data drift and calibration drift
- Alarm fatigue risk
- External validity and fairness checks

### Speaker notes
- "This is clinical decision support, not autonomous diagnosis."
- "Threshold governance, monitoring, and subgroup reporting are required before broad rollout."

---

## Slide 12: Next Steps and Close

### On slide
- Temporal validation and fairness reporting
- Lead-specific threshold tuning with clinicians
- Pilot deployment and monitoring loop

### Speaker notes
- "The next milestone is a monitored pilot with threshold tuning tied to clinician workflow."
- "The package already includes data, models, deployment assets, plots, and notes for immediate presentation."

---

## Optional Q&A Backup Points

- **Why choose LSTM-event as final deployment?**
  - The final milestone prioritizes sequence-aware risk modeling and direct lead-horizon inference with a unified API contract.

- **Why is AUPRC numerically small?**
  - This is a rare-event task; positive prevalence is very low, so precision-recall values are naturally compressed.

- **Can this run in Docker now?**
  - Yes. `deployment/Dockerfile`, `compose.build.yml`, and `docker-stack.yml` are aligned to `rolling_lstm_event` artifacts.
