# Deployment (Docker + Swarm)

This folder contains deployment code for a local web application:

- `risk-api` (FastAPI backend) for model inference
- `risk-web` (Nginx static frontend) for user input + prediction display

Streaming support:

- Frontend can run a simulated bedside stream mode.
- Backend exposes WebSocket endpoint at `/api/ws/live` (proxied to FastAPI `/ws/live`).

## 1) Ensure model artifacts exist

From repository root:

```bash
python feature_engineering/build_rolling_window_timeseries.py
python modeling/train_all_models.py
```

This generates/collects sequence metrics for six models, produces a comparison figure, and stages the selected best model into:

- `deployment/backend/model_artifacts/selected_sequence/`
- `deployment/backend/model_artifacts/model_selection.json`

If you already trained all models and only want to re-rank/re-stage:

```bash
python modeling/train_all_models.py --skip-sequence-compare-train
```

Note:

- Backend first loads models from `/app/model_artifacts/selected_sequence`.
- If no selected model exists, it falls back to `/app/model_artifacts/rolling_lstm_event`.
- Rebuild `risk-api` after re-running comparison so deployment uses newest selected artifacts.

## 2) Build local images

```bash
cd deployment
cp .env.example .env
docker compose --env-file .env -f compose.build.yml build
```

## 3) (Optional) Push to your registry

```bash
docker compose --env-file .env -f compose.build.yml push
```

## 4) Deploy with Docker Swarm

Initialize swarm once (if not already):

```bash
docker swarm init
```

Deploy stack:

```bash
docker stack deploy --with-registry-auth --compose-file docker-stack.yml ${STACK_NAME}
```

Check services:

```bash
docker stack services ${STACK_NAME}
```

Access app:

- UI: `http://localhost:8080`
- API health: `http://localhost:8080/api/health`

## 4.1) Plot deployed model metrics

From repository root:

```bash
python deployment/plot_deployed_model_metrics.py
```

Output:

- `deployment/artifacts/deployed_model_metrics.png`

## 5) Remove stack

```bash
docker stack rm ${STACK_NAME}
```
