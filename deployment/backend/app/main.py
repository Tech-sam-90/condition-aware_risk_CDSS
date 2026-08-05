import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field, ValidationError


MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/model_artifacts"))
SEQ_LEN = int(os.getenv("SEQUENCE_LENGTH", "24"))
SELECTED_SUBDIR = os.getenv("SELECTED_MODEL_SUBDIR", "selected_sequence")
CONDITIONS = ["sepsis", "heart_failure", "ckd", "diabetes", "other"]
VITAL_COLS = ["heart_rate", "sbp", "map", "resp_rate", "spo2", "temp_f"]


class RiskRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    condition: Literal["sepsis", "heart_failure", "ckd", "diabetes"]
    lead_hours: Literal[1, 2, 3] = 1
    heart_rate: float = Field(..., ge=10, le=260)
    sbp: float = Field(..., ge=30, le=300)
    map_value: float = Field(..., alias="map", ge=20, le=220)
    resp_rate: float = Field(..., ge=4, le=80)
    spo2: float = Field(..., ge=40, le=100)
    temp_f: float = Field(..., ge=80, le=112)


class RiskResponse(BaseModel):
    condition: str
    lead_hours: int
    risk_probability: float
    high_risk_threshold: float
    risk_band: Literal["low", "moderate", "high"]


def risk_band(prob: float, high_threshold: float) -> str:
    if prob >= high_threshold:
        return "high"
    if prob >= 0.5 * high_threshold:
        return "moderate"
    return "low"


def load_artifacts():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("TensorFlow is required to serve LSTM artifacts.") from exc

    preferred_dir = MODEL_DIR / SELECTED_SUBDIR
    fallback_dir = MODEL_DIR / "rolling_lstm_event"

    if preferred_dir.exists():
        rolling_dir = preferred_dir
    elif fallback_dir.exists():
        rolling_dir = fallback_dir
    else:
        raise FileNotFoundError(
            f"No model directory found under {MODEL_DIR}. "
            f"Expected '{SELECTED_SUBDIR}' or 'rolling_lstm_event'."
        )

    models = {}
    missing = []
    prefixes = []
    for lead in [1, 2, 3]:
        matches = sorted(rolling_dir.glob(f"*_lead_{lead}h.keras"))
        if len(matches) == 0:
            missing.append(str(rolling_dir / f"*_lead_{lead}h.keras"))
            continue
        if len(matches) > 1:
            raise RuntimeError(
                f"Multiple model files found for lead={lead}h in {rolling_dir}: {matches}"
            )

        path = matches[0]
        prefixes.append(path.name.replace(f"_lead_{lead}h.keras", ""))
        models[lead] = tf.keras.models.load_model(path, compile=False)

    if missing:
        raise FileNotFoundError(f"Missing model artifacts: {', '.join(missing)}")

    model_family = prefixes[0] if prefixes else "unknown"
    metadata = {
        "model_family": model_family,
        "sequence_length": SEQ_LEN,
        "vitals": VITAL_COLS,
        "conditions": CONDITIONS,
        "source_dir": str(rolling_dir),
    }
    return models, metadata


def condition_one_hot(condition: str) -> np.ndarray:
    idx = {name: i for i, name in enumerate(CONDITIONS)}
    out = np.zeros((len(CONDITIONS),), dtype=np.float32)
    out[idx.get(str(condition).lower(), idx["other"])] = 1.0
    return out


def feature_sequence(payload: RiskRequest) -> np.ndarray:
    vitals_vec = np.array(
        [
            payload.heart_rate,
            payload.sbp,
            payload.map_value,
            payload.resp_rate,
            payload.spo2,
            payload.temp_f,
        ],
        dtype=np.float32,
    )
    cond_vec = condition_one_hot(payload.condition)
    step_vec = np.concatenate([vitals_vec, cond_vec], axis=0)
    seq = np.tile(step_vec, (SEQ_LEN, 1)).astype(np.float32)
    return np.expand_dims(seq, axis=0)


def predict_risk(payload: RiskRequest) -> RiskResponse:
    x = feature_sequence(payload)

    model = MODELS.get(payload.lead_hours)
    if model is None:
        raise HTTPException(status_code=500, detail=f"Model for lead={payload.lead_hours}h not loaded")

    prob = float(model.predict(x, verbose=0).reshape(-1)[0])
    high_thr = float(os.getenv("HIGH_RISK_THRESHOLD", "0.02"))

    return RiskResponse(
        condition=payload.condition,
        lead_hours=payload.lead_hours,
        risk_probability=round(prob, 4),
        high_risk_threshold=round(high_thr, 4),
        risk_band=risk_band(prob, high_thr),
    )


app = FastAPI(title="Condition-Aware Risk API", version="1.0.0")

try:
    MODELS, METADATA = load_artifacts()
except Exception as exc:
    MODELS = None
    METADATA = None
    STARTUP_ERROR = str(exc)
else:
    STARTUP_ERROR = None


@app.get("/health")
def health():
    if STARTUP_ERROR:
        raise HTTPException(status_code=500, detail=STARTUP_ERROR)
    return {"status": "ok"}


@app.post("/predict", response_model=RiskResponse)
def predict(payload: RiskRequest):
    if STARTUP_ERROR:
        raise HTTPException(status_code=500, detail=STARTUP_ERROR)
    return predict_risk(payload)


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await websocket.accept()

    if STARTUP_ERROR:
        await websocket.send_json({"type": "error", "detail": STARTUP_ERROR})
        await websocket.close(code=1011)
        return

    try:
        while True:
            raw = await websocket.receive_json()
            try:
                payload = RiskRequest.model_validate(raw)
            except ValidationError as exc:
                await websocket.send_json(
                    {
                        "type": "error",
                        "detail": "Invalid payload",
                        "errors": exc.errors(),
                    }
                )
                continue

            prediction = predict_risk(payload)
            await websocket.send_json(
                {
                    "type": "prediction",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **prediction.model_dump(),
                }
            )
    except WebSocketDisconnect:
        return
