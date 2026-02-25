from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROCESSED_DIR = Path("/home/ubuntu/condition-aware_risk_CDSS/data/processed")
MODEL_DIR = Path("/home/ubuntu/condition-aware_risk_CDSS/modeling/artifacts")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def pick_primary_condition(row):
    if row.get("has_sepsis", 0) == 1:
        return "sepsis"
    if row.get("has_heart_failure", 0) == 1:
        return "heart_failure"
    if row.get("has_ckd", 0) == 1:
        return "ckd"
    if row.get("has_diabetes", 0) == 1:
        return "diabetes"
    return "other"


def build_preprocessors(num_cols, cat_cols):
    try:
        dense_ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        dense_ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    log_pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", dense_ohe),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    boost_pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]), num_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", dense_ohe),
                    ]
                ),
                cat_cols,
            ),
        ]
    )
    return log_pre, boost_pre


def threshold_for_sensitivity(y_true, y_prob, target_sensitivity=0.85):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    thresholds = np.unique(np.round(y_prob, 6))
    thresholds = np.sort(thresholds)[::-1]

    best_thr = float(np.min(y_prob))
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if sens >= target_sensitivity:
            best_thr = float(thr)
            break
    return best_thr


def main():
    data_path = PROCESSED_DIR / "condition_model_table_v2_with_24h_vitals.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Run feature_engineering/build_24h_vitals_features.py first."
        )

    df = pd.read_csv(data_path)
    df["condition_input"] = df.apply(pick_primary_condition, axis=1)
    df["hr_minus_map_mean"] = df["heart_rate_mean"] - df["map_mean"]

    feature_cols = [
        "condition_input",
        "heart_rate_mean",
        "sbp_mean",
        "map_mean",
        "resp_rate_mean",
        "spo2_mean",
        "temp_f_mean",
        "hr_minus_map_mean",
    ]
    target_col = "hospital_expire_flag"

    model_df = df[feature_cols + [target_col]].copy()
    keep_rows = model_df[
        ["heart_rate_mean", "sbp_mean", "map_mean", "resp_rate_mean", "spo2_mean", "temp_f_mean"]
    ].notna().any(axis=1)
    model_df = model_df[keep_rows].copy()

    X = model_df[feature_cols]
    y = model_df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    num_cols = [
        "heart_rate_mean",
        "sbp_mean",
        "map_mean",
        "resp_rate_mean",
        "spo2_mean",
        "temp_f_mean",
        "hr_minus_map_mean",
    ]
    cat_cols = ["condition_input"]

    log_pre, boost_pre = build_preprocessors(num_cols, cat_cols)

    logistic = Pipeline(
        steps=[
            ("preprocess", log_pre),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    boosted = Pipeline(
        steps=[
            ("preprocess", boost_pre),
            (
                "clf",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_depth=4,
                    max_iter=300,
                    min_samples_leaf=50,
                    random_state=42,
                ),
            ),
        ]
    )

    logistic.fit(X_train, y_train)
    boosted.fit(X_train, y_train)

    calibrated_boosted = CalibratedClassifierCV(boosted, method="isotonic", cv=3)
    calibrated_boosted.fit(X_train, y_train)

    proba_log = logistic.predict_proba(X_test)[:, 1]
    proba_boost = boosted.predict_proba(X_test)[:, 1]
    proba_cal = calibrated_boosted.predict_proba(X_test)[:, 1]

    metrics = pd.DataFrame(
        [
            {
                "model": "logistic_regression",
                "roc_auc": roc_auc_score(y_test, proba_log),
                "auprc": average_precision_score(y_test, proba_log),
                "brier": brier_score_loss(y_test, proba_log),
            },
            {
                "model": "hist_gradient_boosting",
                "roc_auc": roc_auc_score(y_test, proba_boost),
                "auprc": average_precision_score(y_test, proba_boost),
                "brier": brier_score_loss(y_test, proba_boost),
            },
            {
                "model": "calibrated_hist_gradient_boosting",
                "roc_auc": roc_auc_score(y_test, proba_cal),
                "auprc": average_precision_score(y_test, proba_cal),
                "brier": brier_score_loss(y_test, proba_cal),
            },
        ]
    ).sort_values("roc_auc", ascending=False)

    threshold_rows = []
    for cond in ["sepsis", "heart_failure", "ckd", "diabetes"]:
        mask = X_test["condition_input"] == cond
        if mask.sum() < 20 or y_test[mask].sum() < 5:
            continue
        thr = threshold_for_sensitivity(y_test[mask], proba_cal[mask], target_sensitivity=0.85)
        threshold_rows.append({"condition_input": cond, "threshold_high_risk": thr})

    thresholds = pd.DataFrame(threshold_rows)
    default_thr = float(threshold_for_sensitivity(y_test, proba_cal, target_sensitivity=0.85))

    artifacts = {
        "feature_cols": feature_cols,
        "default_threshold_high_risk": default_thr,
        "thresholds_by_condition": {
            row["condition_input"]: float(row["threshold_high_risk"])
            for _, row in thresholds.iterrows()
        },
    }

    with open(MODEL_DIR / "logistic_model.pkl", "wb") as f:
        pickle.dump(logistic, f)
    with open(MODEL_DIR / "boosted_model.pkl", "wb") as f:
        pickle.dump(boosted, f)
    with open(MODEL_DIR / "calibrated_boosted_model.pkl", "wb") as f:
        pickle.dump(calibrated_boosted, f)
    with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(artifacts, f, indent=2)

    metrics.to_csv(MODEL_DIR / "model_metrics.csv", index=False)
    thresholds.to_csv(MODEL_DIR / "condition_thresholds.csv", index=False)

    eval_df = X_test.copy()
    eval_df["y_true"] = y_test.values
    eval_df["p_logistic"] = proba_log
    eval_df["p_boosted"] = proba_boost
    eval_df["p_calibrated_boosted"] = proba_cal
    eval_df.to_csv(MODEL_DIR / "evaluation_predictions.csv", index=False)

    print(metrics)
    print(f"Saved modeling artifacts to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
