from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ROLLING_ARTIFACT_DIR = ROOT / "modeling" / "artifacts" / "rolling_lstm_event"
METRICS_PATH = ROLLING_ARTIFACT_DIR / "metrics_by_lead.csv"
METADATA_PATH = ROLLING_ARTIFACT_DIR / "metadata.json"
OUTPUT_DIR = ROOT / "deployment" / "artifacts"
OUTPUT_PATH = OUTPUT_DIR / "deployed_model_metrics.png"


def main() -> None:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Missing metrics file: {METRICS_PATH}")

    metrics = pd.read_csv(METRICS_PATH).sort_values("lead_hours")
    model_family = "lstm_event_timeseries"
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        model_family = metadata.get("model_family", model_family)
    leads = metrics["lead_hours"].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    fig.suptitle(f"Deployed Model Evaluation Metrics ({model_family})", fontsize=12)

    roc_vals = metrics["roc_auc"].tolist()
    auprc_vals = metrics["auprc"].tolist()
    brier_vals = metrics["brier"].tolist()

    axes[0].plot(leads, roc_vals, marker="o")
    axes[0].set_title("ROC-AUC")
    axes[0].set_xlabel("Lead Hours")
    axes[0].set_ylabel("Score")
    axes[0].grid(alpha=0.25)
    for x, y in zip(leads, roc_vals):
        axes[0].annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9)

    axes[1].plot(leads, auprc_vals, marker="o", color="tab:orange")
    axes[1].set_title("AUPRC")
    axes[1].set_xlabel("Lead Hours")
    axes[1].set_ylabel("Score")
    axes[1].grid(alpha=0.25)
    for x, y in zip(leads, auprc_vals):
        axes[1].annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9)

    axes[2].plot(leads, brier_vals, marker="o", color="tab:green")
    axes[2].set_title("Brier Score")
    axes[2].set_xlabel("Lead Hours")
    axes[2].set_ylabel("Score")
    axes[2].grid(alpha=0.25)
    for x, y in zip(leads, brier_vals):
        axes[2].annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=180)
    plt.close(fig)

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
