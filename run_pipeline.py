from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def _run(script_rel_path: str, extra_args: list[str] | None = None, env: dict[str, str] | None = None) -> None:
    """Execute a project script using the current Python interpreter."""
    script_path = PROJECT_ROOT / script_rel_path
    if not script_path.exists():
        raise FileNotFoundError(f"Required script not found: {script_path}")

    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n>>> Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env or os.environ.copy(), check=True)


def _build_global_resample_env(minority_ratio: float, oversample_multiplier: float) -> dict[str, str]:
    """Apply one consistent imbalance strategy across all model families."""
    env = os.environ.copy()
    ratio = str(minority_ratio)
    mult = str(oversample_multiplier)

    env["TABULAR_RESAMPLE_ENABLED"] = "1"
    env["TABULAR_TARGET_MINORITY_RATIO"] = ratio
    env["TABULAR_OVERSAMPLE_MULTIPLIER"] = mult

    env["ROLLING_RESAMPLE_ENABLED"] = "1"
    env["ROLLING_TARGET_MINORITY_RATIO"] = ratio
    env["ROLLING_OVERSAMPLE_MULTIPLIER"] = mult

    env["LSTM_RESAMPLE_ENABLED"] = "1"
    env["LSTM_TARGET_MINORITY_RATIO"] = ratio
    env["LSTM_OVERSAMPLE_MULTIPLIER"] = mult

    env["SEQUENCE_RESAMPLE_ENABLED"] = "1"
    env["SEQUENCE_TARGET_MINORITY_RATIO"] = ratio
    env["SEQUENCE_POSITIVE_OVERSAMPLE_MULTIPLIER"] = mult

    env["LSTM_EVENT_RESAMPLE_ENABLED"] = "1"
    env["LSTM_EVENT_TARGET_MINORITY_RATIO"] = ratio
    env["LSTM_EVENT_POSITIVE_OVERSAMPLE_MULTIPLIER"] = mult

    return env


def run_data_stage() -> None:
    _run("data_processing/prepare_condition_cohort.py")


def run_feature_stage() -> None:
    _run("feature_engineering/build_24h_vitals_features.py")
    _run("feature_engineering/build_rolling_window_timeseries.py")


def run_training_stage(minority_ratio: float, oversample_multiplier: float, skip_optional_lstm: bool) -> None:
    env = _build_global_resample_env(minority_ratio, oversample_multiplier)
    train_args = [
        "--minority-ratio",
        str(minority_ratio),
        "--oversample-multiplier",
        str(oversample_multiplier),
    ]
    if skip_optional_lstm:
        train_args.append("--skip-optional-lstm")

    _run("modeling/train_all_models.py", extra_args=train_args, env=env)


def run_evaluation_stage() -> None:
    _run("evaluation/evaluate_calibration_and_explainability.py")
    _run("evaluation/plot_rolling_metrics_and_vitals.py")
    _run("evaluation/plot_lr_lstm_boosted_comparison.py")
    _run("evaluation/plot_all_sequence_model_comparison.py")
    _run("evaluation/plot_overfitting_curves.py")
    _run("evaluation/tune_imbalance_thresholds.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Single-file pipeline runner for the Condition-Aware ICU Risk CDSS project. "
            "Use --stage all for full end-to-end execution."
        )
    )
    parser.add_argument(
        "--stage",
        choices=["all", "data", "features", "train", "evaluate"],
        default="all",
        help="Pipeline stage to execute.",
    )
    parser.add_argument(
        "--minority-ratio",
        type=float,
        default=0.15,
        help="Target minority ratio for class-imbalance handling during training.",
    )
    parser.add_argument(
        "--oversample-multiplier",
        type=float,
        default=2.0,
        help="Positive class oversampling multiplier where supported.",
    )
    parser.add_argument(
        "--skip-optional-lstm",
        action="store_true",
        help="Skip optional 24-hour LSTM baseline training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.stage in {"all", "data"}:
        run_data_stage()

    if args.stage in {"all", "features"}:
        run_feature_stage()

    if args.stage in {"all", "train"}:
        run_training_stage(
            minority_ratio=args.minority_ratio,
            oversample_multiplier=args.oversample_multiplier,
            skip_optional_lstm=args.skip_optional_lstm,
        )

    if args.stage in {"all", "evaluate"}:
        run_evaluation_stage()

    print("\nPipeline execution finished successfully.")


if __name__ == "__main__":
    main()
