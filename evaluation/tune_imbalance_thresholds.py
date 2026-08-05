from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "modeling" / "artifacts"
EVAL_DIR = PROJECT_ROOT / "evaluation" / "artifacts"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def _safe_div(n: float, d: float) -> float:
    return float(n / d) if d else 0.0


def _metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    fp = float(((y_pred == 1) & (y_true == 0)).sum())
    fn = float(((y_pred == 0) & (y_true == 1)).sum())

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    beta = 2.0
    f2 = _safe_div((1 + beta * beta) * precision * recall, (beta * beta * precision + recall))

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
    }


def _find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_recall: float = 0.85,
) -> tuple[float, dict[str, float]]:
    thresholds = np.unique(np.round(y_prob, 6))
    thresholds = np.sort(thresholds)
    best_thr = 0.5
    best_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "f2": -1.0}

    for thr in thresholds:
        m = _metrics_at_threshold(y_true, y_prob, float(thr))
        if m["recall"] < min_recall:
            continue
        if m["f2"] > best_metrics["f2"]:
            best_thr = float(thr)
            best_metrics = m

    # If min recall is unattainable, fall back to global best F2.
    if best_metrics["f2"] < 0:
        best_thr = 0.5
        for thr in thresholds:
            m = _metrics_at_threshold(y_true, y_prob, float(thr))
            if m["f2"] > best_metrics["f2"]:
                best_thr = float(thr)
                best_metrics = m

    return best_thr, best_metrics


def _tune_tabular(min_recall: float) -> pd.DataFrame:
    pred_path = MODEL_DIR / "evaluation_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing tabular predictions: {pred_path}")

    df = pd.read_csv(pred_path)
    rows = []

    for condition, sub in df.groupby("condition_input"):
        y = sub["y_true"].astype(int).to_numpy()
        p = sub["p_calibrated_boosted"].astype(float).to_numpy()
        if len(sub) < 30 or int(y.sum()) < 5:
            continue

        threshold, metrics = _find_best_threshold(y, p, min_recall=min_recall)
        rows.append(
            {
                "family": "tabular",
                "model": "calibrated_hist_gradient_boosting",
                "lead_hours": np.nan,
                "condition": str(condition),
                "threshold": threshold,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "f2": metrics["f2"],
                "n_rows": int(len(sub)),
                "n_pos": int(y.sum()),
            }
        )

    return pd.DataFrame(rows)


def _tune_sequence_family(family_name: str, proba_col: str, min_recall: float) -> pd.DataFrame:
    family_dir = MODEL_DIR / family_name
    rows = []

    for lead in [1, 2, 3]:
        path = family_dir / f"predictions_lead_{lead}h.csv"
        if not path.exists():
            continue

        df = pd.read_csv(path)
        target_col = f"target_event_in_{lead}h"
        if target_col not in df.columns or proba_col not in df.columns:
            continue

        y = df[target_col].astype(int).to_numpy()
        p = df[proba_col].astype(float).to_numpy()
        if len(df) < 100 or int(y.sum()) < 10:
            continue

        threshold, metrics = _find_best_threshold(y, p, min_recall=min_recall)
        rows.append(
            {
                "family": "sequence",
                "model": family_name,
                "lead_hours": int(lead),
                "condition": "all",
                "threshold": threshold,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "f2": metrics["f2"],
                "n_rows": int(len(df)),
                "n_pos": int(y.sum()),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    min_recall = 0.85
    all_rows = []

    tabular = _tune_tabular(min_recall=min_recall)
    if not tabular.empty:
        all_rows.append(tabular)

    seq_map = {
        "rolling_boosted": "p_calibrated_boosted",
        "rolling_sequence": "p_gru",
        "rolling_lstm_event": "p_lstm_event",
    }

    for family_name, proba_col in seq_map.items():
        tuned = _tune_sequence_family(family_name, proba_col, min_recall=min_recall)
        if not tuned.empty:
            all_rows.append(tuned)

    if not all_rows:
        raise RuntimeError("No prediction files available for threshold tuning.")

    out = pd.concat(all_rows, ignore_index=True)
    out = out.sort_values(["family", "model", "lead_hours", "condition"], na_position="last").reset_index(drop=True)

    out_csv = EVAL_DIR / "imbalance_threshold_recommendations.csv"
    out.to_csv(out_csv, index=False)

    out_json = EVAL_DIR / "imbalance_threshold_recommendations.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out.fillna("").to_dict(orient="records"), f, indent=2)

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
