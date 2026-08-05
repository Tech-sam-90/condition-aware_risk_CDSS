from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "modeling" / "artifacts" / "model_comparison_all_sequence_models.csv"
OUTPUT_PNG = ROOT / "evaluation" / "artifacts" / "all_sequence_models_comparison.png"


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"{INPUT_CSV} not found. Run modeling/train_all_models.py first."
        )

    df = pd.read_csv(INPUT_CSV)
    required = {"model", "lead_hours", "roc_auc", "auprc", "brier"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in comparison CSV: {sorted(missing)}")

    leads = sorted(df["lead_hours"].unique().tolist())
    models = df["model"].drop_duplicates().tolist()

    fig, axes = plt.subplots(1, 3, figsize=(19, 5), constrained_layout=True)
    metric_specs = [
        ("roc_auc", "ROC-AUC (higher better)"),
        ("auprc", "AUPRC (higher better)"),
        ("brier", "Brier (lower better)"),
    ]

    for ax, (metric, title) in zip(axes, metric_specs):
        width = 0.12
        x = range(len(models))
        for i, lead in enumerate(leads):
            vals = []
            for model in models:
                v = df[(df["model"] == model) & (df["lead_hours"] == lead)][metric]
                vals.append(float(v.iloc[0]) if len(v) else float("nan"))
            pos = [xi + (i - 1) * width for xi in x]
            ax.bar(pos, vals, width=width, label=f"lead {lead}h")

        ax.set_title(title)
        ax.set_xticks(list(x))
        ax.set_xticklabels(models, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.3)

    axes[0].legend()
    fig.suptitle("Sequence Model Comparison: LSTM/GRU + 4 New Architectures", fontsize=13)

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=200)
    plt.close(fig)

    print(f"Saved comparison figure to: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
