from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance


MODEL_DIR = Path("/home/ubuntu/condition-aware_risk_CDSS/modeling/artifacts")
EVAL_DIR = Path("/home/ubuntu/condition-aware_risk_CDSS/evaluation/artifacts")
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def get_feature_names_from_preprocessor(preprocessor):
    out = []
    for name, transformer, cols in preprocessor.transformers_:
        if transformer == "drop":
            continue
        if hasattr(transformer, "named_steps") and "onehot" in transformer.named_steps:
            onehot = transformer.named_steps["onehot"]
            out.extend(onehot.get_feature_names_out(cols).tolist())
        else:
            out.extend(cols)
    return out


def main():
    pred_path = MODEL_DIR / "evaluation_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"{pred_path} not found. Run modeling/train_calibrated_condition_models.py first."
        )

    preds = pd.read_csv(pred_path)
    y_true = preds["y_true"].astype(int)

    # Calibration plot
    plt.figure(figsize=(7, 6))
    for col, label in [
        ("p_logistic", "Logistic"),
        ("p_boosted", "Boosted"),
        ("p_calibrated_boosted", "Calibrated Boosted"),
    ]:
        frac_pos, mean_pred = calibration_curve(y_true, preds[col], n_bins=10, strategy="quantile")
        plt.plot(mean_pred, frac_pos, marker="o", label=label)

    plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    plt.xlabel("Mean predicted risk")
    plt.ylabel("Observed event rate")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    calib_plot_path = EVAL_DIR / "reliability_diagram.png"
    plt.savefig(calib_plot_path, dpi=160)
    plt.close()

    # Load models for explainability
    with open(MODEL_DIR / "boosted_model.pkl", "rb") as f:
        boosted = pickle.load(f)
    with open(MODEL_DIR / "logistic_model.pkl", "rb") as f:
        logistic = pickle.load(f)

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
    X = preds[feature_cols].copy()

    # Permutation importance for boosted model
    pi = permutation_importance(
        boosted,
        X,
        y_true,
        scoring="roc_auc",
        n_repeats=10,
        random_state=42,
        n_jobs=1,
    )

    perm_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_mean": pi.importances_mean,
            "importance_std": pi.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    perm_df.to_csv(EVAL_DIR / "permutation_importance_boosted.csv", index=False)

    # Logistic coefficients (directional explainability)
    preprocessor = logistic.named_steps["preprocess"]
    clf = logistic.named_steps["clf"]
    transformed_feature_names = get_feature_names_from_preprocessor(preprocessor)

    coef_df = pd.DataFrame(
        {
            "feature": transformed_feature_names,
            "coefficient": clf.coef_[0],
            "abs_coefficient": abs(clf.coef_[0]),
        }
    ).sort_values("abs_coefficient", ascending=False)
    coef_df.to_csv(EVAL_DIR / "logistic_coefficients.csv", index=False)

    print(f"Saved: {calib_plot_path}")
    print(f"Saved: {EVAL_DIR / 'permutation_importance_boosted.csv'}")
    print(f"Saved: {EVAL_DIR / 'logistic_coefficients.csv'}")


if __name__ == "__main__":
    main()
