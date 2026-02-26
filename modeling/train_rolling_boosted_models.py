from pathlib import Path
import json
import pickle

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


PROCESSED_DIR = Path("/home/ubuntu/condition-aware_risk_CDSS/data/processed")
ART_DIR = Path("/home/ubuntu/condition-aware_risk_CDSS/modeling/artifacts/rolling_boosted")
ART_DIR.mkdir(parents=True, exist_ok=True)


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
    }
    with open(ART_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(metrics_df)
    print(f"Saved rolling boosted artifacts to: {ART_DIR}")


if __name__ == "__main__":
    main()
