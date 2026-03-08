from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/home/ubuntu/condition-aware_risk_CDSS")
MODEL_ART_DIR = ROOT / "modeling" / "artifacts"
OUT_DIR = ROOT / "evaluation" / "artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BOOSTED_METRICS_PATH = MODEL_ART_DIR / "rolling_boosted" / "metrics_by_lead.csv"
GRU_METRICS_PATH = MODEL_ART_DIR / "rolling_sequence" / "metrics_by_lead.csv"
LSTM_EVENT_METRICS_PATH = MODEL_ART_DIR / "rolling_lstm_event" / "metrics_by_lead.csv"
OUT_PATH = OUT_DIR / "lr_lstm_boosted_performance_comparison.png"


def load_metrics() -> pd.DataFrame:
    for path in [BOOSTED_METRICS_PATH, GRU_METRICS_PATH, LSTM_EVENT_METRICS_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing metrics file: {path}")

    def macro_from_lead_metrics(path: Path, model_label: str) -> dict:
        df = pd.read_csv(path)
        needed = ["roc_auc", "auprc", "brier"]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns {missing} in {path}")

        return {
            "model_label": model_label,
            "roc_auc": float(df["roc_auc"].mean()),
            "auprc": float(df["auprc"].mean()),
            "brier": float(df["brier"].mean()),
        }

    rows = [
        macro_from_lead_metrics(BOOSTED_METRICS_PATH, "Boosted (Calibrated HGB)"),
        macro_from_lead_metrics(GRU_METRICS_PATH, "GRU (Rolling Sequence)"),
        macro_from_lead_metrics(LSTM_EVENT_METRICS_PATH, "LSTM-event (Rolling Sequence)"),
    ]
    combined = pd.DataFrame(rows)
    order = [
        "Boosted (Calibrated HGB)",
        "GRU (Rolling Sequence)",
        "LSTM-event (Rolling Sequence)",
    ]
    combined["model_label"] = pd.Categorical(combined["model_label"], categories=order, ordered=True)
    return combined.sort_values("model_label")


def make_plot(metrics: pd.DataFrame) -> None:
    model_labels = metrics["model_label"].tolist()
    roc_auc = metrics["roc_auc"].tolist()
    auprc = metrics["auprc"].tolist()
    brier = metrics["brier"].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Rolling Lead-Event Model Family Comparison (Macro Across 1h/2h/3h)", fontsize=12)

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
