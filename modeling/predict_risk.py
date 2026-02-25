import argparse
import json
import pickle
from pathlib import Path

import pandas as pd


MODEL_DIR = Path("/home/ubuntu/condition-aware_risk_CDSS/modeling/artifacts")


def risk_band(prob, high_threshold):
    if prob >= high_threshold:
        return "high"
    if prob >= 0.5 * high_threshold:
        return "moderate"
    return "low"


def main():
    parser = argparse.ArgumentParser(description="Condition-aware risk prediction")
    parser.add_argument("--condition", required=True, choices=["sepsis", "heart_failure", "ckd", "diabetes"])
    parser.add_argument("--heart_rate", type=float, required=True)
    parser.add_argument("--sbp", type=float, required=True)
    parser.add_argument("--map", dest="map_value", type=float, required=True)
    parser.add_argument("--resp_rate", type=float, required=True)
    parser.add_argument("--spo2", type=float, required=True)
    parser.add_argument("--temp_f", type=float, required=True)
    args = parser.parse_args()

    with open(MODEL_DIR / "calibrated_boosted_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODEL_DIR / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    x = pd.DataFrame(
        [
            {
                "condition_input": args.condition,
                "heart_rate_mean": args.heart_rate,
                "sbp_mean": args.sbp,
                "map_mean": args.map_value,
                "resp_rate_mean": args.resp_rate,
                "spo2_mean": args.spo2,
                "temp_f_mean": args.temp_f,
                "hr_minus_map_mean": args.heart_rate - args.map_value,
            }
        ]
    )

    p = float(model.predict_proba(x)[:, 1][0])
    high_thr = metadata["thresholds_by_condition"].get(args.condition, metadata["default_threshold_high_risk"])
    out = {
        "condition": args.condition,
        "risk_probability": round(p, 4),
        "high_risk_threshold": round(float(high_thr), 4),
        "risk_band": risk_band(p, float(high_thr)),
    }
    print(out)


if __name__ == "__main__":
    main()
