from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELING_DIR = PROJECT_ROOT / "modeling"
TRAINING_CORE_DIR = PROJECT_ROOT / "training_core"
ART_DIR = PROJECT_ROOT / "modeling" / "artifacts"
DEPLOY_SELECT_DIR = PROJECT_ROOT / "deployment" / "backend" / "model_artifacts" / "selected_sequence"
DEPLOY_META_PATH = PROJECT_ROOT / "deployment" / "backend" / "model_artifacts" / "model_selection.json"
OUT_COMPARE_CSV = ART_DIR / "model_comparison_all_sequence_models.csv"
OUT_SUMMARY_CSV = ART_DIR / "model_comparison_all_sequence_models_summary.csv"
OUT_PLOT = PROJECT_ROOT / "evaluation" / "artifacts" / "all_sequence_models_comparison.png"

MODEL_CONFIG = [
    {
        "model": "sequence_lstm_event",
        "train_script": TRAINING_CORE_DIR / "train_rolling_lstm_event_models.py",
        "metrics_path": ART_DIR / "rolling_lstm_event" / "metrics_by_lead.csv",
        "artifacts_dir": ART_DIR / "rolling_lstm_event",
    },
    {
        "model": "sequence_gru",
        "train_script": TRAINING_CORE_DIR / "train_rolling_sequence_models.py",
        "metrics_path": ART_DIR / "rolling_sequence" / "metrics_by_lead.csv",
        "artifacts_dir": ART_DIR / "rolling_sequence",
    },
    {
        "model": "tcn_smote2575",
        "train_script": TRAINING_CORE_DIR / "advanced_time_series" / "train_tcn_smote.py",
        "metrics_path": ART_DIR / "advanced_time_series" / "tcn_smote2575" / "metrics_by_lead.csv",
        "artifacts_dir": ART_DIR / "advanced_time_series" / "tcn_smote2575",
    },
    {
        "model": "bilstm_attention_smote2575",
        "train_script": TRAINING_CORE_DIR / "advanced_time_series" / "train_bilstm_attention_smote.py",
        "metrics_path": ART_DIR / "advanced_time_series" / "bilstm_attention_smote2575" / "metrics_by_lead.csv",
        "artifacts_dir": ART_DIR / "advanced_time_series" / "bilstm_attention_smote2575",
    },
    {
        "model": "transformer_encoder_smote2575",
        "train_script": TRAINING_CORE_DIR / "advanced_time_series" / "train_transformer_encoder_smote.py",
        "metrics_path": ART_DIR / "advanced_time_series" / "transformer_encoder_smote2575" / "metrics_by_lead.csv",
        "artifacts_dir": ART_DIR / "advanced_time_series" / "transformer_encoder_smote2575",
    },
    {
        "model": "tft_style_smote2575",
        "train_script": TRAINING_CORE_DIR / "advanced_time_series" / "train_tft_style_smote.py",
        "metrics_path": ART_DIR / "advanced_time_series" / "tft_style_smote2575" / "metrics_by_lead.csv",
        "artifacts_dir": ART_DIR / "advanced_time_series" / "tft_style_smote2575",
    },
]


def _run_step(step_name: str, script_path: Path, env: dict[str, str]) -> None:
    """Run one training script with inherited+override environment variables."""
    cmd = [sys.executable, str(script_path)]
    print(f"\n=== [{step_name}] {' '.join(cmd)} ===")
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, check=True)


def _run_sequence_compare(env: dict[str, str], skip_train: bool) -> None:
    selected_cfgs = MODEL_CONFIG

    if not skip_train:
        for cfg in selected_cfgs:
            _run_step(f"sequence-train:{cfg['model']}", cfg["train_script"], env)

    rows = []
    for cfg in selected_cfgs:
        path = cfg["metrics_path"]
        if not path.exists():
            raise FileNotFoundError(f"Missing metrics for {cfg['model']}: {path}")

        df = pd.read_csv(path)
        keep = [c for c in ["lead_hours", "roc_auc", "auprc", "brier", "n_test_rows", "n_test_pos"] if c in df.columns]
        sub = df[keep].copy()
        sub["model"] = cfg["model"]
        rows.append(sub)

    compare_df = pd.concat(rows, ignore_index=True).sort_values(["lead_hours", "model"]).reset_index(drop=True)
    compare_df.to_csv(OUT_COMPARE_CSV, index=False)

    summary = (
        compare_df.groupby("model", as_index=False)[["roc_auc", "auprc", "brier"]]
        .mean()
        .rename(columns={"roc_auc": "mean_roc_auc", "auprc": "mean_auprc", "brier": "mean_brier"})
    )
    summary["rank_auc"] = summary["mean_roc_auc"].rank(ascending=False, method="min")
    summary["rank_auprc"] = summary["mean_auprc"].rank(ascending=False, method="min")
    summary["rank_brier"] = summary["mean_brier"].rank(ascending=True, method="min")
    summary["composite_rank"] = summary[["rank_auc", "rank_auprc", "rank_brier"]].mean(axis=1)
    summary = summary.sort_values(
        ["composite_rank", "mean_roc_auc", "mean_auprc", "mean_brier"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)
    summary.to_csv(OUT_SUMMARY_CSV, index=False)

    import matplotlib.pyplot as plt

    OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    leads = sorted(compare_df["lead_hours"].unique().tolist())
    models = compare_df["model"].drop_duplicates().tolist()
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
                v = compare_df[(compare_df["model"] == model) & (compare_df["lead_hours"] == lead)][metric]
                vals.append(float(v.iloc[0]) if len(v) else float("nan"))
            pos = [xi + (i - 1) * width for xi in x]
            ax.bar(pos, vals, width=width, label=f"lead {lead}h")

        ax.set_title(title)
        ax.set_xticks(list(x))
        ax.set_xticklabels(models, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.3)

    axes[0].legend()
    fig.suptitle("Sequence Model Comparison: LSTM/GRU + 4 New Architectures", fontsize=13)
    fig.savefig(OUT_PLOT, dpi=200)
    plt.close(fig)

    best_model = str(summary.iloc[0]["model"])
    model_cfg = next((c for c in selected_cfgs if c["model"] == best_model), None)
    if model_cfg is None:
        raise ValueError(f"No config found for best model: {best_model}")

    src_dir = model_cfg["artifacts_dir"]
    DEPLOY_SELECT_DIR.mkdir(parents=True, exist_ok=True)
    for p in DEPLOY_SELECT_DIR.glob("*.keras"):
        p.unlink()

    staged = []
    for lead in [1, 2, 3]:
        matches = sorted(src_dir.glob(f"*_lead_{lead}h.keras"))
        if not matches:
            raise FileNotFoundError(f"No artifact found for lead={lead}h in {src_dir}")
        if len(matches) > 1:
            raise RuntimeError(f"Multiple artifacts found for lead={lead}h in {src_dir}: {matches}")

        src = matches[0]
        dst = DEPLOY_SELECT_DIR / src.name
        shutil.copy2(src, dst)
        staged.append(dst.name)

    DEPLOY_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DEPLOY_META_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "selected_model": best_model,
                "source_artifacts_dir": str(src_dir),
                "staged_artifacts": staged,
                "selection_method": "lowest composite rank (AUC/AUPRC/Brier)",
            },
            f,
            indent=2,
        )


def _risk_band(prob: float, high_threshold: float) -> str:
    if prob >= high_threshold:
        return "high"
    if prob >= 0.5 * high_threshold:
        return "moderate"
    return "low"


def _run_prediction(args: argparse.Namespace) -> None:
    model_dir = PROJECT_ROOT / "modeling" / "artifacts"

    with open(model_dir / "calibrated_boosted_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(model_dir / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    x = pd.DataFrame(
        [
            {
                "condition_input": args.condition,
                "heart_rate_mean": args.heart_rate,
                "sbp_mean": args.sbp,
                "map_mean": args.map_value,
                "resp_rate_mean": args.resp_rate,
                "spo2_mean": args.spo2,
                "temp_f_mean": args.temp_f,
                "hr_minus_map_mean": args.heart_rate - args.map_value,
            }
        ]
    )

    p = float(model.predict_proba(x)[:, 1][0])
    high_thr = metadata["thresholds_by_condition"].get(args.condition, metadata["default_threshold_high_risk"])
    out = {
        "condition": args.condition,
        "risk_probability": round(p, 4),
        "high_risk_threshold": round(float(high_thr), 4),
        "risk_band": _risk_band(p, float(high_thr)),
    }
    print(out)


def _build_resample_env(args: argparse.Namespace) -> dict[str, str]:
    """Create a unified imbalance policy applied consistently across model families."""
    env = os.environ.copy()

    ratio = str(args.minority_ratio)
    mult = str(args.oversample_multiplier)

    # Tabular models
    env["TABULAR_RESAMPLE_ENABLED"] = "1"
    env["TABULAR_TARGET_MINORITY_RATIO"] = ratio
    env["TABULAR_OVERSAMPLE_MULTIPLIER"] = mult

    # Rolling boosted models
    env["ROLLING_RESAMPLE_ENABLED"] = "1"
    env["ROLLING_TARGET_MINORITY_RATIO"] = ratio
    env["ROLLING_OVERSAMPLE_MULTIPLIER"] = mult

    # LSTM baseline
    env["LSTM_RESAMPLE_ENABLED"] = "1"
    env["LSTM_TARGET_MINORITY_RATIO"] = ratio
    env["LSTM_OVERSAMPLE_MULTIPLIER"] = mult

    # Sequence windows (GRU/LSTM-event)
    env["SEQUENCE_RESAMPLE_ENABLED"] = "1"
    env["SEQUENCE_TARGET_MINORITY_RATIO"] = ratio
    env["SEQUENCE_POSITIVE_OVERSAMPLE_MULTIPLIER"] = mult
    env["LSTM_EVENT_RESAMPLE_ENABLED"] = "1"
    env["LSTM_EVENT_TARGET_MINORITY_RATIO"] = ratio
    env["LSTM_EVENT_POSITIVE_OVERSAMPLE_MULTIPLIER"] = mult

    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Single-entry training launcher for the project. "
            "Runs tabular + sequence + advanced sequence comparison with one command."
        )
    )
    parser.add_argument(
        "--minority-ratio",
        type=float,
        default=0.15,
        help="Target minority class ratio used by internal resampling logic.",
    )
    parser.add_argument(
        "--oversample-multiplier",
        type=float,
        default=2.0,
        help="Positive-class oversampling multiplier where supported.",
    )
    parser.add_argument(
        "--target-sensitivity",
        type=float,
        default=0.85,
        help="Target sensitivity used for tabular high-risk thresholding.",
    )
    parser.add_argument(
        "--skip-optional-lstm",
        action="store_true",
        help="Skip the optional 24h LSTM baseline to reduce runtime/dependency load.",
    )
    parser.add_argument(
        "--skip-sequence-compare-train",
        action="store_true",
        help="Reuse existing comparison metrics without re-training sequence models.",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Run inference mode using calibrated boosted model artifacts.",
    )
    parser.add_argument(
        "--condition",
        choices=["sepsis", "heart_failure", "ckd", "diabetes"],
        help="Condition for inference mode.",
    )
    parser.add_argument("--heart-rate", dest="heart_rate", type=float, help="Heart rate for inference mode.")
    parser.add_argument("--sbp", type=float, help="Systolic blood pressure for inference mode.")
    parser.add_argument("--map", dest="map_value", type=float, help="MAP for inference mode.")
    parser.add_argument("--resp-rate", dest="resp_rate", type=float, help="Respiratory rate for inference mode.")
    parser.add_argument("--spo2", type=float, help="SpO2 for inference mode.")
    parser.add_argument("--temp-f", dest="temp_f", type=float, help="Temperature Fahrenheit for inference mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.predict:
        needed = [
            args.condition,
            args.heart_rate,
            args.sbp,
            args.map_value,
            args.resp_rate,
            args.spo2,
            args.temp_f,
        ]
        if any(v is None for v in needed):
            raise ValueError(
                "Prediction mode requires --condition, --heart-rate, --sbp, --map, --resp-rate, --spo2, --temp-f"
            )
        _run_prediction(args)
        return

    env = _build_resample_env(args)
    env["TABULAR_TARGET_SENSITIVITY"] = str(args.target_sensitivity)

    _run_step(
        "tabular-calibrated",
        TRAINING_CORE_DIR / "train_calibrated_condition_models.py",
        env,
    )
    _run_step(
        "rolling-boosted",
        TRAINING_CORE_DIR / "train_rolling_boosted_models.py",
        env,
    )

    if not args.skip_optional_lstm:
        _run_step(
            "lstm-24h-baseline",
            TRAINING_CORE_DIR / "train_lstm_timeseries.py",
            env,
        )

    _run_sequence_compare(env=env, skip_train=args.skip_sequence_compare_train)

    print("\nTraining pipeline complete.")
    print("- Tabular artifacts: modeling/artifacts/")
    print("- Rolling boosted artifacts: modeling/artifacts/rolling_boosted/")
    print("- Sequence comparison: modeling/artifacts/model_comparison_all_sequence_models.csv")
    print("- Deployment staging metadata: deployment/backend/model_artifacts/model_selection.json")


if __name__ == "__main__":
    main()
