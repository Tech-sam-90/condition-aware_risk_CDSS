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
/home/ubuntu/care_risk_env/bin/python feature_engineering/build_rolling_window_timeseries.py
/home/ubuntu/care_risk_env/bin/python modeling/train_rolling_boosted_models.py
```

This generates the rolling boosted files used by backend image under `modeling/artifacts/rolling_boosted/`.

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

## 5) Remove stack

```bash
docker stack rm ${STACK_NAME}
```
