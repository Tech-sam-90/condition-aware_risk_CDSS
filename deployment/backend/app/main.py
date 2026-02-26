import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field, ValidationError


MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/model_artifacts"))


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
    rolling_dir = MODEL_DIR / "rolling_boosted"
    metadata_path = rolling_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    models = {}
    missing = []
    for lead in [1, 2, 3]:
        path = rolling_dir / f"calibrated_boosted_lead_{lead}h.pkl"
        if not path.exists():
            missing.append(str(path))
            continue
        with open(path, "rb") as f:
            models[lead] = pickle.load(f)

    if missing:
        raise FileNotFoundError(f"Missing model artifacts: {', '.join(missing)}")

    return models, metadata


def condition_code(condition: str) -> int:
    default_map = {"ckd": 0, "diabetes": 1, "heart_failure": 2, "other": 3, "sepsis": 4}
    metadata_map = METADATA.get("condition_code_mapping", {}) if METADATA else {}
    if metadata_map:
        return int(metadata_map.get(condition, metadata_map.get("other", 3)))
    return default_map.get(condition, 3)


def feature_row(payload: RiskRequest):
    hr = payload.heart_rate
    sbp = payload.sbp
    map_value = payload.map_value
    rr = payload.resp_rate
    spo2 = payload.spo2
    temp_f = payload.temp_f

    row = {
        "condition_code": condition_code(payload.condition),
        "heart_rate": hr,
        "sbp": sbp,
        "map": map_value,
        "resp_rate": rr,
        "spo2": spo2,
        "temp_f": temp_f,
        "hr_minus_map": hr - map_value,
        "heart_rate_mean_1h": hr,
        "heart_rate_mean_3h": hr,
        "heart_rate_mean_5h": hr,
        "heart_rate_slope_5h": 0.0,
        "sbp_mean_1h": sbp,
        "sbp_mean_3h": sbp,
        "sbp_mean_5h": sbp,
        "sbp_slope_5h": 0.0,
        "map_mean_1h": map_value,
        "map_mean_3h": map_value,
        "map_mean_5h": map_value,
        "map_slope_5h": 0.0,
        "resp_rate_mean_1h": rr,
        "resp_rate_mean_3h": rr,
        "resp_rate_mean_5h": rr,
        "resp_rate_slope_5h": 0.0,
        "spo2_mean_1h": spo2,
        "spo2_mean_3h": spo2,
        "spo2_mean_5h": spo2,
        "spo2_slope_5h": 0.0,
        "temp_f_mean_1h": temp_f,
        "temp_f_mean_3h": temp_f,
        "temp_f_mean_5h": temp_f,
        "temp_f_slope_5h": 0.0,
    }
    return row


def predict_risk(payload: RiskRequest) -> RiskResponse:
    feature_cols = METADATA.get("feature_columns", [])
    row = feature_row(payload)
    x = pd.DataFrame([{col: row.get(col, 0.0) for col in feature_cols}])

    model = MODELS.get(payload.lead_hours)
    if model is None:
        raise HTTPException(status_code=500, detail=f"Model for lead={payload.lead_hours}h not loaded")

    prob = float(model.predict_proba(x)[:, 1][0])
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
