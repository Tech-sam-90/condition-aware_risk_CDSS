from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ROLLING_ARTIFACT_DIR = ROOT / "modeling" / "artifacts" / "rolling_boosted"
METRICS_PATH = ROLLING_ARTIFACT_DIR / "metrics_by_lead.csv"
METADATA_PATH = ROLLING_ARTIFACT_DIR / "metadata.json"
OUTPUT_DIR = ROOT / "deployment" / "artifacts"
OUTPUT_PATH = OUTPUT_DIR / "deployed_model_metrics.png"


def main() -> None:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Missing metrics file: {METRICS_PATH}")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing metadata file: {METADATA_PATH}")

    metrics = pd.read_csv(METRICS_PATH).sort_values("lead_hours")
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    model_family = metadata.get("model_family", "deployed_model")
    leads = metrics["lead_hours"].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    fig.suptitle(f"Deployed Model Evaluation Metrics ({model_family})", fontsize=12)

    axes[0].plot(leads, metrics["roc_auc"], marker="o")
    axes[0].set_title("ROC-AUC")
    axes[0].set_xlabel("Lead Hours")
    axes[0].set_ylabel("Score")
    axes[0].grid(alpha=0.25)

    axes[1].plot(leads, metrics["auprc"], marker="o", color="tab:orange")
    axes[1].set_title("AUPRC")
    axes[1].set_xlabel("Lead Hours")
    axes[1].set_ylabel("Score")
    axes[1].grid(alpha=0.25)

    axes[2].plot(leads, metrics["brier"], marker="o", color="tab:green")
    axes[2].set_title("Brier Score")
    axes[2].set_xlabel("Lead Hours")
    axes[2].set_ylabel("Score")
    axes[2].grid(alpha=0.25)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=180)
    plt.close(fig)

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
