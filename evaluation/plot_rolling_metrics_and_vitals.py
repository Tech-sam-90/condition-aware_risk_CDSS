from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/home/ubuntu/condition-aware_risk_CDSS")
ART_DIR = ROOT / "evaluation" / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

COMPARISON_PATH = ROOT / "modeling" / "artifacts" / "model_comparison_by_lead.csv"
ROLLING_TS_PATH = ROOT / "data" / "processed" / "rolling_window_multicondition_timeseries.csv"

CONDITION_ORDER = ["sepsis", "heart_failure", "ckd", "diabetes"]
VITALS = ["heart_rate", "sbp", "map", "resp_rate", "spo2", "temp_f"]
METRICS = ["roc_auc", "auprc", "brier"]
METRIC_TITLES = {
    "roc_auc": "ROC-AUC",
    "auprc": "AUPRC",
    "brier": "Brier Score",
}


def plot_metric_comparison() -> None:
    if not COMPARISON_PATH.exists():
        raise FileNotFoundError(f"Missing file: {COMPARISON_PATH}")

    comparison = pd.read_csv(COMPARISON_PATH)
    comparison = comparison.sort_values(["lead_hours", "model"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)

    for idx, metric in enumerate(METRICS):
        pivot = comparison.pivot(index="lead_hours", columns="model", values=metric)
        pivot.plot(kind="bar", ax=axes[idx], width=0.82)

        axes[idx].set_title(METRIC_TITLES[metric])
        axes[idx].set_xlabel("Lead hours")
        axes[idx].set_ylabel(metric)
        axes[idx].grid(alpha=0.25, axis="y")
        axes[idx].legend(title="Model")

        for container in axes[idx].containers:
            vals = [p.get_height() for p in container]
            labels = [f"{v:.4f}" if pd.notna(v) else "" for v in vals]
            axes[idx].bar_label(container, labels=labels, padding=2, fontsize=8)

    fig.suptitle("Rolling-Window Model Comparison by Lead Time", y=1.03)
    fig.tight_layout()
    out_path = ART_DIR / "rolling_model_metrics_comparison.png"
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_vitals_timeseries_by_condition() -> None:
    if not ROLLING_TS_PATH.exists():
        raise FileNotFoundError(f"Missing file: {ROLLING_TS_PATH}")

    usecols = ["condition_input", "hour_from_icu"] + VITALS
    df = pd.read_csv(ROLLING_TS_PATH, usecols=usecols)
    df = df[df["condition_input"].isin(CONDITION_ORDER)].copy()

    agg = (
        df.groupby(["condition_input", "hour_from_icu"], as_index=False)[VITALS]
        .median()
        .sort_values(["condition_input", "hour_from_icu"])
    )

    fig, axes = plt.subplots(2, 3, figsize=(17, 8), sharex=True)
    axes = axes.flatten()

    label_map = {
        "sepsis": "Sepsis",
        "heart_failure": "Heart Failure",
        "ckd": "CKD",
        "diabetes": "Diabetes",
    }

    for ax, vital in zip(axes, VITALS):
        for condition in CONDITION_ORDER:
            subset = agg[agg["condition_input"] == condition]
            ax.plot(subset["hour_from_icu"], subset[vital], label=label_map.get(condition, condition))

        ax.set_title(vital)
        ax.set_xlabel("Hour from ICU admission")
        ax.set_ylabel("Median value")
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=2,
        frameon=False,
        columnspacing=2.0,
        handlelength=2.6,
    )
    fig.suptitle("Vital Time Series (Median by Hour) Across Conditions", y=1.08)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path = ART_DIR / "vitals_timeseries_6_vitals_4_conditions.png"
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plot_metric_comparison()
    plot_vitals_timeseries_by_condition()

    print(f"Saved: {ART_DIR / 'rolling_model_metrics_comparison.png'}")
    print(f"Saved: {ART_DIR / 'vitals_timeseries_6_vitals_4_conditions.png'}")


if __name__ == "__main__":
    main()
