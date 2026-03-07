from pathlib import Path
import json
import os
import pickle

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

try:
    from resampling_utils import rebalance_binary_dataframe
except ImportError:
    from modeling.resampling_utils import rebalance_binary_dataframe


PROCESSED_DIR = Path("/home/ubuntu/condition-aware_risk_CDSS/data/processed")
ART_DIR = Path("/home/ubuntu/condition-aware_risk_CDSS/modeling/artifacts/rolling_boosted")
ART_DIR.mkdir(parents=True, exist_ok=True)

RESAMPLE_ENABLED = os.getenv("ROLLING_RESAMPLE_ENABLED", "1") == "1"
TARGET_MINORITY_RATIO = float(os.getenv("ROLLING_TARGET_MINORITY_RATIO", "0.15"))
OVERSAMPLE_MULTIPLIER = float(os.getenv("ROLLING_OVERSAMPLE_MULTIPLIER", "2.0"))


def make_preprocessor(num_cols):
    return Pipeline([("imputer", SimpleImputer(strategy="median"))])


def feature_columns(df: pd.DataFrame):
    base_num = [
        "heart_rate",
        "sbp",
        "map",
        "resp_rate",
        "spo2",
        "temp_f",
        "hr_minus_map",
    ]
    roll_cols = [c for c in df.columns if any(c.endswith(s) for s in ["_mean_1h", "_mean_3h", "_mean_5h", "_slope_5h"])]
    num_cols = sorted(set(base_num + roll_cols))
    num_cols.append("condition_code")
    return sorted(set(num_cols))


def main():
    data_path = PROCESSED_DIR / "rolling_window_multicondition_timeseries.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Run feature_engineering/build_rolling_window_timeseries.py first."
        )

    df = pd.read_csv(data_path)
    df["stay_id"] = pd.to_numeric(df["stay_id"], errors="coerce")
    df["hour_from_icu"] = pd.to_numeric(df["hour_from_icu"], errors="coerce")
    df = df.dropna(subset=["stay_id", "hour_from_icu", "condition_input"])
    df["condition_code"] = df["condition_input"].astype("category").cat.codes.astype("int16")

    df = df[(df["hour_from_icu"] >= 4) & (df["hour_from_icu"] <= 44)].copy()
    for col in [c for c in df.columns if c not in ["condition_input"]]:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")

    feat_cols = feature_columns(df)

    metrics_rows = []

    for lead in [1, 2, 3]:
        target_col = f"target_event_in_{lead}h"
        work = df[feat_cols + [target_col, "stay_id"]].copy()
        work[target_col] = work[target_col].fillna(0).astype(int)

        stay_labels = work.groupby("stay_id")[target_col].max().reset_index()
        train_stays, test_stays = train_test_split(
            stay_labels,
            test_size=0.2,
            random_state=42,
            stratify=stay_labels[target_col],
        )

        train_ids = set(train_stays["stay_id"])
        test_ids = set(test_stays["stay_id"])

        train_df = work[work["stay_id"].isin(train_ids)].copy()
        test_df = work[work["stay_id"].isin(test_ids)].copy()

        resample_plan = {
            "n_pos_original": int(train_df[target_col].sum()),
            "n_neg_original": int(len(train_df) - train_df[target_col].sum()),
            "n_pos_sample": int(train_df[target_col].sum()),
            "n_neg_sample": int(len(train_df) - train_df[target_col].sum()),
            "target_minority_ratio": float(TARGET_MINORITY_RATIO),
            "achieved_minority_ratio": float(train_df[target_col].mean()) if len(train_df) else 0.0,
            "oversample_multiplier": float(OVERSAMPLE_MULTIPLIER),
            "pos_replace": False,
            "neg_replace": False,
        }
        if RESAMPLE_ENABLED:
            train_df, resample_plan = rebalance_binary_dataframe(
                train_df,
                target_col=target_col,
                target_minority_ratio=TARGET_MINORITY_RATIO,
                oversample_multiplier=OVERSAMPLE_MULTIPLIER,
                random_state=42 + lead,
            )
        print(
            f"lead={lead}h resampling: "
            f"before pos={resample_plan['n_pos_original']} neg={resample_plan['n_neg_original']} | "
            f"after pos={resample_plan['n_pos_sample']} neg={resample_plan['n_neg_sample']} | "
            f"minority_rate={resample_plan['achieved_minority_ratio']:.4f}"
        )

        X_train = train_df[feat_cols]
        y_train = train_df[target_col]
        X_test = test_df[feat_cols]
        y_test = test_df[target_col]

        pre = make_preprocessor(feat_cols)
        base = Pipeline(
            steps=[
                ("preprocess", pre),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_depth=4,
                        max_iter=180,
                        min_samples_leaf=40,
                        random_state=42,
                    ),
                ),
            ]
        )

        calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=2)
        calibrated.fit(X_train, y_train)

        p = calibrated.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, p)
        auprc = average_precision_score(y_test, p)
        brier = brier_score_loss(y_test, p)

        metrics_rows.append(
            {
                "lead_hours": lead,
                "roc_auc": auc,
                "auprc": auprc,
                "brier": brier,
                "n_train_rows": len(train_df),
                "n_test_rows": len(test_df),
                "n_test_pos": int(y_test.sum()),
                "train_minority_rate": float(y_train.mean()) if len(y_train) else 0.0,
                "train_pos_before": int(resample_plan["n_pos_original"]),
                "train_neg_before": int(resample_plan["n_neg_original"]),
                "train_pos_after": int(resample_plan["n_pos_sample"]),
                "train_neg_after": int(resample_plan["n_neg_sample"]),
            }
        )

        pred_out = test_df[["stay_id", target_col]].copy()
        pred_out["p_calibrated_boosted"] = p
        pred_out.to_csv(ART_DIR / f"predictions_lead_{lead}h.csv", index=False)

        with open(ART_DIR / f"calibrated_boosted_lead_{lead}h.pkl", "wb") as f:
            pickle.dump(calibrated, f)

    metrics_df = pd.DataFrame(metrics_rows).sort_values("lead_hours")
    metrics_df.to_csv(ART_DIR / "metrics_by_lead.csv", index=False)

    metadata = {
        "model_family": "calibrated_hist_gradient_boosting",
        "feature_columns": feat_cols,
        "lead_hours": [1, 2, 3],
        "resampling": {
            "enabled": RESAMPLE_ENABLED,
            "target_minority_ratio": TARGET_MINORITY_RATIO,
            "oversample_multiplier": OVERSAMPLE_MULTIPLIER,
        },
    }
    with open(ART_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(metrics_df)
    print(f"Saved rolling boosted artifacts to: {ART_DIR}")


if __name__ == "__main__":
    main()
