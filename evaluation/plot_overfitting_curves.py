import re
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split


ROOT = Path("/home/ubuntu/condition-aware_risk_CDSS")
MODEL_ART = ROOT / "modeling" / "artifacts"
EVAL_ART = ROOT / "evaluation" / "artifacts"
EVAL_ART.mkdir(parents=True, exist_ok=True)

LSTM_EVENT_LOG = MODEL_ART / "rolling_lstm_event" / "train_lstm_event_15pct.log"
GRU_LOG = MODEL_ART / "rolling_sequence" / "train_gru_15pct.log"

LSTM_EVENT_METRICS = MODEL_ART / "rolling_lstm_event" / "metrics_by_lead.csv"
GRU_METRICS = MODEL_ART / "rolling_sequence" / "metrics_by_lead.csv"
TABULAR_DATA = ROOT / "data" / "processed" / "condition_model_table_v2_with_24h_vitals.csv"
CALIBRATED_TABULAR_MODEL = MODEL_ART / "calibrated_boosted_model.pkl"


METRIC_RE = re.compile(
    r"auc:\s*([0-9.]+)\s*-\s*loss:\s*([0-9.]+)\s*-\s*val_auc:\s*([0-9.]+)\s*-\s*val_loss:\s*([0-9.]+)"
)
LEAD_RE = re.compile(r"lead=(\d+)h")


def pick_primary_condition(row: pd.Series) -> str:
    if row.get("has_sepsis", 0) == 1:
        return "sepsis"
    if row.get("has_heart_failure", 0) == 1:
        return "heart_failure"
    if row.get("has_ckd", 0) == 1:
        return "ckd"
    if row.get("has_diabetes", 0) == 1:
        return "diabetes"
    return "other"


def parse_sequence_history(log_path: Path) -> pd.DataFrame:
    rows = []
    current_lead = None
    epoch_by_lead = {}

    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        lm = LEAD_RE.search(line)
        if lm:
            current_lead = int(lm.group(1))
            epoch_by_lead.setdefault(current_lead, 0)

        mm = METRIC_RE.search(line)
        if not mm or current_lead is None:
            continue

        epoch_by_lead[current_lead] += 1
        rows.append(
            {
                "lead_hours": current_lead,
                "epoch": epoch_by_lead[current_lead],
                "train_auc": float(mm.group(1)),
                "train_loss": float(mm.group(2)),
                "val_auc": float(mm.group(3)),
                "val_loss": float(mm.group(4)),
            }
        )

    return pd.DataFrame(rows)


def plot_sequence_family(history: pd.DataFrame, test_auc_map: dict, model_name: str, out_name: str) -> Path:
    out_path = EVAL_ART / out_name
    leads = sorted(history["lead_hours"].unique().tolist())

    fig, axes = plt.subplots(len(leads), 2, figsize=(12, 4.2 * len(leads)), sharex=False)
    if len(leads) == 1:
        axes = [axes]

    for row_idx, lead in enumerate(leads):
        sub = history[history["lead_hours"] == lead].sort_values("epoch")
        ax_auc = axes[row_idx][0]
        ax_loss = axes[row_idx][1]

        ax_auc.plot(sub["epoch"], sub["train_auc"], marker="o", label="Train AUC")
        ax_auc.plot(sub["epoch"], sub["val_auc"], marker="o", label="Validation AUC")
        test_auc = test_auc_map.get(lead)
        if test_auc is not None:
            ax_auc.axhline(test_auc, linestyle="--", linewidth=1.2, label=f"Test AUC ({test_auc:.4f})")
        ax_auc.set_title(f"{model_name} Lead {lead}h AUC by Epoch")
        ax_auc.set_xlabel("Epoch")
        ax_auc.set_ylabel("AUC")
        ax_auc.grid(alpha=0.25)
        ax_auc.legend()

        ax_loss.plot(sub["epoch"], sub["train_loss"], marker="o", label="Train Loss")
        ax_loss.plot(sub["epoch"], sub["val_loss"], marker="o", label="Validation Loss")
        ax_loss.set_title(f"{model_name} Lead {lead}h Loss by Epoch")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.grid(alpha=0.25)
        ax_loss.legend()

    fig.suptitle(f"{model_name} Overfitting Check by Lead (15% Minority Resampling)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out_path


def calibrated_hist_train_test_metrics() -> dict:
    if not TABULAR_DATA.exists():
        raise FileNotFoundError(f"Missing tabular dataset: {TABULAR_DATA}")
    if not CALIBRATED_TABULAR_MODEL.exists():
        raise FileNotFoundError(f"Missing calibrated model artifact: {CALIBRATED_TABULAR_MODEL}")

    df = pd.read_csv(TABULAR_DATA)
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

    with open(CALIBRATED_TABULAR_MODEL, "rb") as f:
        calibrated = pickle.load(f)

    p_train = calibrated.predict_proba(X_train)[:, 1]
    p_test = calibrated.predict_proba(X_test)[:, 1]

    train_auc = float(roc_auc_score(y_train, p_train))
    test_auc = float(roc_auc_score(y_test, p_test))
    train_auprc = float(average_precision_score(y_train, p_train))
    test_auprc = float(average_precision_score(y_test, p_test))
    test_brier = float(((p_test - y_test.to_numpy()) ** 2).mean())

    return {
        "model": "Calibrated Hist (tabular)",
        "lead_hours": None,
        "reference_type": "test",
        "train_auc": train_auc,
        "reference_auc": test_auc,
        "gap_auc": float(train_auc - test_auc),
        "test_auc": test_auc,
        "test_auprc": test_auprc,
        "test_brier": test_brier,
        "train_auprc": train_auprc,
        "reference_auprc": test_auprc,
    }


def plot_overfitting_comparison(
    lstm_event_hist: pd.DataFrame,
    gru_hist: pd.DataFrame,
    lstm_event_test_metrics: pd.DataFrame,
    gru_test_metrics: pd.DataFrame,
) -> tuple[Path, Path]:
    rows = []

    rows.append(calibrated_hist_train_test_metrics())

    def metrics_map(df: pd.DataFrame) -> dict:
        out = {}
        for _, r in df.iterrows():
            lead = int(r["lead_hours"])
            out[lead] = {
                "roc_auc": float(r["roc_auc"]),
                "auprc": float(r["auprc"]),
                "brier": float(r["brier"]),
            }
        return out

    lstm_test_map = metrics_map(lstm_event_test_metrics)
    gru_test_map = metrics_map(gru_test_metrics)

    for lead in sorted(lstm_event_hist["lead_hours"].unique().tolist()):
        sub = lstm_event_hist[lstm_event_hist["lead_hours"] == lead].sort_values("epoch")
        last = sub.iloc[-1]
        rows.append(
            {
                "model": "LSTM-event",
                "lead_hours": lead,
                "reference_type": "validation",
                "train_auc": float(last["train_auc"]),
                "reference_auc": float(last["val_auc"]),
                "gap_auc": float(last["train_auc"] - last["val_auc"]),
                "test_auc": float(lstm_test_map[lead]["roc_auc"]),
                "test_auprc": float(lstm_test_map[lead]["auprc"]),
                "test_brier": float(lstm_test_map[lead]["brier"]),
            }
        )

    for lead in sorted(gru_hist["lead_hours"].unique().tolist()):
        sub = gru_hist[gru_hist["lead_hours"] == lead].sort_values("epoch")
        last = sub.iloc[-1]
        rows.append(
            {
                "model": "GRU",
                "lead_hours": lead,
                "reference_type": "validation",
                "train_auc": float(last["train_auc"]),
                "reference_auc": float(last["val_auc"]),
                "gap_auc": float(last["train_auc"] - last["val_auc"]),
                "test_auc": float(gru_test_map[lead]["roc_auc"]),
                "test_auprc": float(gru_test_map[lead]["auprc"]),
                "test_brier": float(gru_test_map[lead]["brier"]),
            }
        )

    comp = pd.DataFrame(rows)

    def row_label(row: pd.Series) -> str:
        lead = row.get("lead_hours")
        if pd.notna(lead):
            return f"{row['model']} ({int(lead)}h)"
        return str(row["model"])

    comp["label"] = comp.apply(row_label, axis=1)
    csv_path = EVAL_ART / "overfitting_comparison_15pct.csv"
    comp.to_csv(csv_path, index=False)

    out_path = EVAL_ART / "overfitting_comparison_auc_15pct.png"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), width_ratios=[1.45, 1.0])

    x = list(range(len(comp)))
    labels = comp["label"].tolist()
    train = comp["train_auc"].to_list()
    ref = comp["reference_auc"].to_list()

    axes[0].bar([i - 0.18 for i in x], train, width=0.36, label="Train AUC")
    axes[0].bar([i + 0.18 for i in x], ref, width=0.36, label="Reference AUC")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=18, ha="right")
    axes[0].set_ylabel("AUC")
    axes[0].set_title("Train vs Reference AUC")
    axes[0].grid(alpha=0.25, axis="y")
    axes[0].legend()

    gaps = comp["gap_auc"].to_list()
    gap_colors = ["#d62728" if g > 0 else "#2ca02c" for g in gaps]
    bars = axes[1].bar(x, gaps, color=gap_colors)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=18, ha="right")
    axes[1].set_ylabel("AUC Gap")
    axes[1].set_title("Overfitting Gap (Train - Reference)")
    axes[1].grid(alpha=0.25, axis="y")
    axes[1].bar_label(bars, labels=[f"{g:.4f}" for g in gaps], padding=3, fontsize=8)

    fig.suptitle("Overfitting Comparison Across Models (15% Minority Resampling)")
    fig.text(
        0.5,
        0.01,
        "Reference AUC: test for Calibrated Hist; validation for LSTM-event/GRU. Test AUC/AUPRC/Brier are included in CSV.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out_path, csv_path


def main() -> None:
    if not LSTM_EVENT_LOG.exists():
        raise FileNotFoundError(f"Missing LSTM-event log: {LSTM_EVENT_LOG}")
    if not GRU_LOG.exists():
        raise FileNotFoundError(f"Missing GRU log: {GRU_LOG}")
    if not LSTM_EVENT_METRICS.exists():
        raise FileNotFoundError(f"Missing LSTM-event metrics: {LSTM_EVENT_METRICS}")
    if not GRU_METRICS.exists():
        raise FileNotFoundError(f"Missing GRU metrics: {GRU_METRICS}")

    lstm_event_hist = parse_sequence_history(LSTM_EVENT_LOG)
    if lstm_event_hist.empty:
        raise RuntimeError("No epoch metrics found in LSTM-event log.")

    gru_hist = parse_sequence_history(GRU_LOG)
    if gru_hist.empty:
        raise RuntimeError("No epoch metrics found in GRU log.")

    # Save extracted histories for traceability.
    lstm_hist_path = MODEL_ART / "rolling_lstm_event" / "lstm_event_training_history_from_log.csv"
    lstm_event_hist.to_csv(lstm_hist_path, index=False)
    gru_hist.to_csv(MODEL_ART / "rolling_sequence" / "gru_training_history_from_log.csv", index=False)

    lstm_event_metrics = pd.read_csv(LSTM_EVENT_METRICS)
    gru_metrics = pd.read_csv(GRU_METRICS)
    lstm_test_auc = {int(r["lead_hours"]): float(r["roc_auc"]) for _, r in lstm_event_metrics.iterrows()}
    gru_test_auc = {int(r["lead_hours"]): float(r["roc_auc"]) for _, r in gru_metrics.iterrows()}

    lstm_plot = plot_sequence_family(
        lstm_event_hist,
        lstm_test_auc,
        model_name="LSTM-event",
        out_name="lstm_overfitting_curve_15pct.png",
    )
    gru_plot = plot_sequence_family(
        gru_hist,
        gru_test_auc,
        model_name="GRU",
        out_name="gru_overfitting_curves_15pct.png",
    )
    comparison_plot, comparison_csv = plot_overfitting_comparison(
        lstm_event_hist,
        gru_hist,
        lstm_event_metrics,
        gru_metrics,
    )

    print(f"Saved: {lstm_plot}")
    print(f"Saved: {gru_plot}")
    print(f"Saved: {comparison_plot}")
    print(f"Saved: {comparison_csv}")
    print(f"Saved: {lstm_hist_path}")
    print(f"Saved: {MODEL_ART / 'rolling_sequence' / 'gru_training_history_from_log.csv'}")


if __name__ == "__main__":
    main()
