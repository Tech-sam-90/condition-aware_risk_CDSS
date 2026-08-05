# Condition-Aware ICU Risk CDSS

A condition-aware ICU deterioration-risk project built on MIMIC-IV v3.1: it trains and compares six sequence-model architectures that predict near-term deterioration risk (1h/2h/3h ahead), conditioned on diagnosis category, and serves the best one through a containerized FastAPI + WebSocket service with a browser demo UI.

## What this is (and isn't)

This is a **research/portfolio project**, not a clinically validated or clinically deployed system. Every claim below is scoped to what was actually verified against this codebase:

- The models are real, trained, and evaluated on held-out MIMIC-IV data — not notebook sketches or "planned" work.
- The API, Docker build, and WebSocket streaming are real and were verified end-to-end (build succeeds, `/health`, `/predict`, and `/ws/live` all respond correctly — see "Deployment" below).
- The "live streaming" demo simulates bedside vitals client-side; it is **not** connected to a real monitor, EHR, or live MIMIC feed.
- There has been no external validation, no fairness auditing, and no prospective/clinical deployment. See "Limitations" at the end.

## Project Goals

- Predict near-term deterioration risk at 1h, 2h, 3h horizons.
- Use condition context (sepsis, heart failure, CKD, diabetes, other) as a model input, not just a fixed alarm threshold.
- Compare multiple sequence architectures under identical preprocessing, splitting, and evaluation.
- Automatically select and deploy the best-performing model behind a real API.

## Models

Every model conditions on diagnosis category by one-hot-encoding it and concatenating it to the vitals vector at each timestep (heart rate, SBP, MAP, respiratory rate, SpO2, temperature). All sequence models use a 24-hour lookback window, stay-level train/val/test splitting (no patient leakage), and predict a binary deterioration event at 1h/2h/3h lead time.

| Model | Architecture | Class-imbalance handling |
|---|---|---|
| `sequence_lstm_event` | 2-layer stacked LSTM (64→32 units) | negative-window downsampling + positive oversampling |
| `sequence_gru` | GRU sequence model | negative-window downsampling + positive oversampling |
| `tcn_smote2575` | Temporal convolutional network | SMOTE 25:75 on train split only |
| `bilstm_attention_smote2575` | Bidirectional LSTM (64→32) + multi-head self-attention | SMOTE 25:75 on train split only |
| `transformer_encoder_smote2575` | Transformer encoder (positional embedding + 2 attention blocks) | SMOTE 25:75 on train split only |
| `tft_style_smote2575` | Temporal Fusion Transformer-*style* approximation (not a full TFT) | SMOTE 25:75 on train split only |

A separate tabular pipeline (`training_core/train_calibrated_condition_models.py`) trains isotonic-**calibrated** gradient-boosted (HistGradientBoosting) and logistic models per condition; this is the model used by the `--predict` CLI and for per-condition threshold tuning. **It is not the model served by the API** — see "What 'calibrated' applies to" below.

## Results (real, held-out test metrics)

Six sequence architectures were trained and ranked by composite rank across ROC-AUC, AUPRC, and Brier score (`modeling/artifacts/model_comparison_all_sequence_models_summary.csv`):

| Model | Mean ROC-AUC | Mean AUPRC | Mean Brier | Composite rank |
|---|---|---|---|---|
| **bilstm_attention_smote2575** ✅ deployed | **0.760** | 0.0075 | 0.052 | 1st |
| tft_style_smote2575 | 0.748 | 0.0069 | 0.063 | 2nd |
| sequence_lstm_event | 0.730 | 0.0056 | 0.026 | 3rd |
| tcn_smote2575 | 0.730 | 0.0059 | 0.086 | 4th |
| sequence_gru | 0.693 | 0.0047 | 0.022 | 5th |
| transformer_encoder_smote2575 | 0.661 | 0.0042 | 0.068 | 6th |

The BiLSTM-with-attention model was auto-selected (`modeling/train_all_models.py`) and is the model actually loaded by the deployed API. Per-lead ROC-AUC for the deployed model ranges from 0.74 to 0.80 (`modeling/artifacts/advanced_time_series/bilstm_attention_smote2575/metrics_by_lead.csv`).

**Read the AUPRC numbers honestly**: positive events are ~0.23–0.25% of all evaluation windows (e.g. 315 positive out of 124,124 test windows at the 1h horizon), so AUPRC is necessarily small in absolute terms even for a model with real discriminative signal — a random classifier would score AUPRC ≈ 0.0025 on this base rate, so 0.0075–0.01 reflects a real, if modest, lift, not "the model doesn't work." The plain Transformer-encoder scoring worst (0.66 AUC) is also a real, reported finding, not cherry-picked — full self-attention likely needs more training data than this cohort provides at this scale.

## Repository Structure

```text
condition-aware_risk_CDSS/
├── data/                          # raw/processed data folders (payloads gitignored; see Data section)
├── data_processing/                # cohort assembly, context extraction, source prep
├── feature_engineering/            # rolling-window and modeling features
├── modeling/                       # single training entry point + all artifacts
│   ├── artifacts/                  # trained weights, metrics, comparison outputs
│   └── train_all_models.py
├── training_core/                  # per-architecture training implementations
│   ├── advanced_time_series/       # TCN / BiLSTM-attention / Transformer / TFT-style
│   └── train_*.py
├── evaluation/                     # calibration, comparison, and overfitting plots
├── deployment/                     # Dockerized backend (FastAPI) + frontend (nginx)
│   ├── backend/app/main.py         # API: /health, /predict, /ws/live
│   └── backend/model_artifacts/    # staged best-model artifacts for the image
├── requirements.txt                 # training-pipeline dependencies
└── deployment/backend/requirements.txt  # serving-API dependencies (separate, smaller)
```

## Quickstart: train

```bash
pip install -r requirements.txt
python run_pipeline.py --stage all          # data -> features -> train -> evaluate
# or individually:
python run_pipeline.py --stage train
```

Training six sequence models end-to-end and generating the comparison/staging outputs:

```bash
python modeling/train_all_models.py
```

This writes `modeling/artifacts/model_comparison_all_sequence_models.csv`, a comparison plot at `evaluation/artifacts/all_sequence_models_comparison.png`, and stages the best model's weights to `deployment/backend/model_artifacts/selected_sequence/` for the Docker image to pick up.

## Deployment (verified)

The backend is a FastAPI service (`deployment/backend/app/main.py`) exposing:

- `GET /health`
- `POST /predict` — single-shot risk prediction
- `WEBSOCKET /ws/live` — streaming prediction over an open socket

It loads `deployment/backend/model_artifacts/selected_sequence` if present (currently the BiLSTM-attention model), falling back to `modeling/artifacts/rolling_lstm_event` (the plain LSTM) otherwise. Both are baked into the Docker image at build time — no external model download needed.

**Build and quick local smoke test:**

```bash
cd deployment
docker compose -f compose.build.yml build

docker run -d -p 8000:8000 local/condition-aware-risk-api:latest
curl http://localhost:8000/health
# -> {"status":"ok"}

curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d \
  '{"condition":"heart_failure","lead_hours":3,"heart_rate":88,"sbp":110,"map":78,"resp_rate":18,"spo2":97,"temp_f":98.6}'
# -> {"condition":"heart_failure","lead_hours":3,"risk_probability":0.9364,"high_risk_threshold":0.02,"risk_band":"high"}
```

(Both the build and these exact requests were run against this repository and returned the responses shown above.)

**Full stack with the demo UI, via Docker Swarm:**

```bash
cd deployment
cp .env.example .env
docker compose --env-file .env -f compose.build.yml build
docker swarm init            # if not already a swarm manager
docker stack deploy --with-registry-auth --compose-file docker-stack.yml condition-aware-risk
# UI:          http://localhost:8080
# API health:  http://localhost:8080/api/health
```

### About the "real-time streaming" demo

`/ws/live` is a real WebSocket endpoint: it accepts a vitals JSON payload per message and returns a prediction JSON per message, over one open connection — this was verified directly (two consecutive messages on one socket, two independent predictions back). The browser demo UI's "Start stream" button drives this by jittering the vitals you entered in the form every 1.2 seconds and sending them over the socket. **This is a client-side simulation for demonstrating the streaming inference pattern — it is not connected to a real bedside monitor, EHR feed, or live MIMIC-IV data.**

### What "calibrated" applies to

Isotonic calibration (`training_core/train_calibrated_condition_models.py`, `evaluation/evaluate_calibration_and_explainability.py`) was applied to the **tabular gradient-boosted model**, which is used by the `modeling/train_all_models.py --predict` CLI and for per-condition threshold tuning — not to the sequence model served by the API. The deployed BiLSTM-attention (and fallback LSTM) model's `/predict` and `/ws/live` outputs are raw sigmoid probabilities, uncalibrated.

## Evaluation & Visualization

```bash
python evaluation/evaluate_calibration_and_explainability.py   # reliability diagram, permutation importance (tabular model)
python evaluation/plot_rolling_metrics_and_vitals.py
python evaluation/plot_all_sequence_model_comparison.py        # the 6-model comparison figure used above
```

## Data & MIMIC-IV Access

No raw patient data is stored in this repository (`data/**` is gitignored except for README placeholders and non-payload processed-folder structure). Access to MIMIC-IV v3.1 requires PhysioNet credentialing, CITI training certification, and a signed data use agreement — see `download.sh` for the (credentialed) download pattern and `data/raw/aws/README.md`.

## Limitations — not for clinical use

- Retrospective, offline evaluation on one critical-care database (MIMIC-IV); no external or multi-site validation.
- No fairness/bias auditing across demographic subgroups.
- No prospective or live-patient evaluation; the "streaming" demo uses simulated, not live, vitals.
- Very low positive-event base rate (~0.25%) limits precision-oriented metrics regardless of architecture.
- This repository demonstrates an ML modeling + deployment pipeline for a portfolio/research context. It is not a certified or clinically deployed decision-support system.

## License

MIT — see [LICENSE](LICENSE).

## Author

Samuel Adeniji
