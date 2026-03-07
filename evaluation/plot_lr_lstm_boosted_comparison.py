from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/home/ubuntu/condition-aware_risk_CDSS")
MODEL_ART_DIR = ROOT / "modeling" / "artifacts"
OUT_DIR = ROOT / "evaluation" / "artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_METRICS_PATH = MODEL_ART_DIR / "model_metrics.csv"
LSTM_METRICS_PATH = MODEL_ART_DIR / "lstm_model_metrics.csv"
OUT_PATH = OUT_DIR / "lr_lstm_boosted_performance_comparison.png"


def load_metrics() -> pd.DataFrame:
    if not BASELINE_METRICS_PATH.exists():
        raise FileNotFoundError(f"Missing baseline metrics file: {BASELINE_METRICS_PATH}")
    if not LSTM_METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Missing LSTM metrics file: {LSTM_METRICS_PATH}. Run modeling/train_lstm_timeseries.py first."
        )

    baseline = pd.read_csv(BASELINE_METRICS_PATH)
    lstm = pd.read_csv(LSTM_METRICS_PATH)

    wanted = {
        "logistic_regression": "Logistic Regression",
        "calibrated_hist_gradient_boosting": "Boosted (Calibrated HGB)",
        "lstm_24h_vitals": "LSTM (24h Vitals)",
    }

    baseline = baseline[baseline["model"].isin(["logistic_regression", "calibrated_hist_gradient_boosting"])]
    combined = pd.concat([baseline, lstm], ignore_index=True)

    missing = [m for m in wanted if m not in combined["model"].values]
    if missing:
        raise ValueError(f"Missing metrics for models: {missing}")

    combined["model_label"] = combined["model"].map(wanted)
    order = [
        "Logistic Regression",
        "LSTM (24h Vitals)",
        "Boosted (Calibrated HGB)",
    ]
    combined["model_label"] = pd.Categorical(combined["model_label"], categories=order, ordered=True)
    combined = combined.sort_values("model_label")
    return combined


def make_plot(metrics: pd.DataFrame) -> None:
    model_labels = metrics["model_label"].tolist()
    roc_auc = metrics["roc_auc"].tolist()
    auprc = metrics["auprc"].tolist()
    brier = metrics["brier"].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Model Performance Comparison: Logistic vs LSTM vs Boosted", fontsize=13)

    bars0 = axes[0].bar(model_labels, roc_auc, color=["#5DA5DA", "#60BD68", "#F17CB0"])
    axes[0].set_title("ROC-AUC")
    axes[0].set_ylabel("Score")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].bar_label(bars0, labels=[f"{v:.4f}" for v in roc_auc], padding=3, fontsize=9)

    bars1 = axes[1].bar(model_labels, auprc, color=["#5DA5DA", "#60BD68", "#F17CB0"])
    axes[1].set_title("AUPRC")
    axes[1].set_ylabel("Score")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].bar_label(bars1, labels=[f"{v:.4f}" for v in auprc], padding=3, fontsize=9)

    bars2 = axes[2].bar(model_labels, brier, color=["#5DA5DA", "#60BD68", "#F17CB0"])
    axes[2].set_title("Brier Score (lower is better)")
    axes[2].set_ylabel("Score")
    axes[2].tick_params(axis="x", rotation=20)
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].bar_label(bars2, labels=[f"{v:.4f}" for v in brier], padding=3, fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    metrics = load_metrics()
    make_plot(metrics)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
