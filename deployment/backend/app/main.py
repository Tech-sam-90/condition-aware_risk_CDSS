import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field, ValidationError


MODEL_DIR = Path("/app/model_artifacts")


class RiskRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    condition: Literal["sepsis", "heart_failure", "ckd", "diabetes"]
    heart_rate: float = Field(..., ge=10, le=260)
    sbp: float = Field(..., ge=30, le=300)
    map_value: float = Field(..., alias="map", ge=20, le=220)
    resp_rate: float = Field(..., ge=4, le=80)
    spo2: float = Field(..., ge=40, le=100)
    temp_f: float = Field(..., ge=80, le=112)


class RiskResponse(BaseModel):
    condition: str
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
    model_path = MODEL_DIR / "calibrated_boosted_model.pkl"
    metadata_path = MODEL_DIR / "metadata.json"

    if not model_path.exists() or not metadata_path.exists():
        missing = []
        if not model_path.exists():
            missing.append(str(model_path))
        if not metadata_path.exists():
            missing.append(str(metadata_path))
        raise FileNotFoundError(f"Missing model artifacts: {', '.join(missing)}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, metadata


def predict_risk(payload: RiskRequest) -> RiskResponse:
    row = {
        "condition_input": payload.condition,
        "heart_rate_mean": payload.heart_rate,
        "sbp_mean": payload.sbp,
        "map_mean": payload.map_value,
        "resp_rate_mean": payload.resp_rate,
        "spo2_mean": payload.spo2,
        "temp_f_mean": payload.temp_f,
        "hr_minus_map_mean": payload.heart_rate - payload.map_value,
    }
    x = pd.DataFrame([row])
    prob = float(MODEL.predict_proba(x)[:, 1][0])

    high_thr = float(
        METADATA["thresholds_by_condition"].get(
            payload.condition, METADATA["default_threshold_high_risk"]
        )
    )

    return RiskResponse(
        condition=payload.condition,
        risk_probability=round(prob, 4),
        high_risk_threshold=round(high_thr, 4),
        risk_band=risk_band(prob, high_thr),
    )


app = FastAPI(title="Condition-Aware Risk API", version="1.0.0")

try:
    MODEL, METADATA = load_artifacts()
except Exception as exc:
    MODEL = None
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
