from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ART_ROOT = PROJECT_ROOT / "modeling" / "artifacts" / "advanced_time_series"

VITAL_COLS = ["heart_rate", "sbp", "map", "resp_rate", "spo2", "temp_f"]
CONDITIONS = ["sepsis", "heart_failure", "ckd", "diabetes", "other"]


def load_tf():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("TensorFlow is required. Install with: pip install tensorflow") from exc
    return tf


def parse_args(default_model_name: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Train {default_model_name} rolling models with SMOTE 25:75 balancing."
    )
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", type=str, default=default_model_name)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=PROCESSED_DIR / "rolling_window_multicondition_timeseries.csv",
    )
    return parser.parse_args()


def condition_one_hot(series: pd.Series) -> np.ndarray:
    idx = {name: i for i, name in enumerate(CONDITIONS)}
    arr = np.zeros((len(series), len(CONDITIONS)), dtype=np.float32)
    for row_idx, val in enumerate(series.fillna("other").astype(str).str.lower().tolist()):
        arr[row_idx, idx.get(val, idx["other"])] = 1.0
    return arr


def prepare_dataframe(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Run feature engineering first to produce rolling data."
        )

    df = pd.read_csv(data_path)
    df = df[(df["hour_from_icu"] >= 4) & (df["hour_from_icu"] <= 44)].copy()
    df["stay_id"] = pd.to_numeric(df["stay_id"], errors="coerce").astype("Int64")
    df["hour_from_icu"] = pd.to_numeric(df["hour_from_icu"], errors="coerce")

    for col in VITAL_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
        df[col] = df.groupby("stay_id")[col].ffill().bfill()
        df[col] = df[col].fillna(float(df[col].median()))

    for lead in [1, 2, 3]:
        tcol = f"target_event_in_{lead}h"
        df[tcol] = pd.to_numeric(df[tcol], errors="coerce").fillna(0).astype("int8")

    df = df.dropna(subset=["stay_id", "hour_from_icu", "condition_input"]).copy()
    return df


def build_sequences_for_stays(
    df: pd.DataFrame,
    stay_ids: list[int],
    target_col: str,
    seq_len: int,
):
    X_parts = []
    y_parts = []
    sid_parts = []

    for stay_id in stay_ids:
        group = df[df["stay_id"] == stay_id].sort_values("hour_from_icu").reset_index(drop=True)
        if len(group) < seq_len:
            continue

        cond_ohe = condition_one_hot(group["condition_input"])
        y_vals = group[target_col].fillna(0).astype(int).to_numpy()
        x_num_all = group[VITAL_COLS].to_numpy(dtype=np.float32)

        x_stay = []
        y_stay = []
        for end_idx in range(seq_len - 1, len(group)):
            start = end_idx - seq_len + 1
            x_num = x_num_all[start : end_idx + 1]
            x_cond = cond_ohe[start : end_idx + 1]
            x = np.concatenate([x_num, x_cond], axis=1).astype(np.float32)
            x_stay.append(x)
            y_stay.append(int(y_vals[end_idx]))

        if x_stay:
            X_parts.append(np.stack(x_stay, axis=0))
            y_parts.append(np.array(y_stay, dtype=np.int32))
            sid_parts.append(np.full(len(y_stay), stay_id))

    if not X_parts:
        raise RuntimeError("No sequences could be built for requested stay IDs.")

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    sids = np.concatenate(sid_parts, axis=0)
    return X, y, sids


def apply_smote_25_75(X_train: np.ndarray, y_train: np.ndarray, seed: int):
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as exc:
        raise ImportError(
            "imbalanced-learn is required for SMOTE. Install with: pip install imbalanced-learn"
        ) from exc

    y_train = np.asarray(y_train).astype(int)
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())

    if n_pos < 2 or n_neg < 2:
        stats = {
            "train_pos_before": n_pos,
            "train_neg_before": n_neg,
            "train_pos_after": n_pos,
            "train_neg_after": n_neg,
            "minority_ratio_after": float(n_pos / max(n_pos + n_neg, 1)),
            "smote_applied": False,
        }
        return X_train, y_train, stats

    X_flat = X_train.reshape(len(X_train), -1)

    # minority/majority ratio = 0.25 / 0.75 = 1/3
    smote = SMOTE(
        sampling_strategy=1.0 / 3.0,
        random_state=seed,
        k_neighbors=max(1, min(5, n_pos - 1)),
    )
    X_res, y_res = smote.fit_resample(X_flat, y_train)

    X_res = X_res.reshape(len(X_res), X_train.shape[1], X_train.shape[2]).astype(np.float32)
    y_res = y_res.astype(np.int32)

    pos_after = int((y_res == 1).sum())
    neg_after = int((y_res == 0).sum())
    stats = {
        "train_pos_before": n_pos,
        "train_neg_before": n_neg,
        "train_pos_after": pos_after,
        "train_neg_after": neg_after,
        "minority_ratio_after": float(pos_after / max(pos_after + neg_after, 1)),
        "smote_applied": True,
    }
    return X_res, y_res, stats


def get_stay_splits(df: pd.DataFrame, target_col: str, seed: int):
    stay_level = df.groupby("stay_id", as_index=False)[target_col].max().rename(columns={target_col: "y"})
    stay_level["y"] = stay_level["y"].fillna(0).astype(int)

    train_stays, test_stays = train_test_split(
        stay_level,
        test_size=0.2,
        random_state=seed,
        stratify=stay_level["y"],
    )

    train_only, val_only = train_test_split(
        train_stays,
        test_size=0.15,
        random_state=seed,
        stratify=train_stays["y"],
    )

    return (
        train_only["stay_id"].astype(int).tolist(),
        val_only["stay_id"].astype(int).tolist(),
        test_stays["stay_id"].astype(int).tolist(),
    )


def run_training_pipeline(
    model_name: str,
    build_model_fn: Callable,
    args: argparse.Namespace,
):
    tf = load_tf()
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    artifact_dir = ART_ROOT / model_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_dataframe(args.data_path)

    all_metrics = []
    for lead in [1, 2, 3]:
        target_col = f"target_event_in_{lead}h"
        train_ids, val_ids, test_ids = get_stay_splits(df, target_col, args.seed)

        X_train, y_train, _ = build_sequences_for_stays(df, train_ids, target_col, args.seq_len)
        X_val, y_val, _ = build_sequences_for_stays(df, val_ids, target_col, args.seq_len)
        X_test, y_test, test_sids = build_sequences_for_stays(df, test_ids, target_col, args.seq_len)

        X_train_bal, y_train_bal, smote_stats = apply_smote_25_75(X_train, y_train, args.seed + lead)

        model = build_model_fn(
            tf=tf,
            seq_len=args.seq_len,
            input_dim=X_train_bal.shape[-1],
            learning_rate=args.learning_rate,
            lead_hours=lead,
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=3,
                restore_best_weights=True,
            )
        ]

        history = model.fit(
            X_train_bal,
            y_train_bal,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        p = model.predict(X_test, batch_size=args.batch_size, verbose=0).reshape(-1)

        metrics = {
            "lead_hours": lead,
            "roc_auc": float(roc_auc_score(y_test, p)),
            "auprc": float(average_precision_score(y_test, p)),
            "brier": float(brier_score_loss(y_test, p)),
            "n_train_rows_before_smote": int(len(X_train)),
            "n_train_rows_after_smote": int(len(X_train_bal)),
            "n_val_rows": int(len(X_val)),
            "n_test_rows": int(len(X_test)),
            "n_test_pos": int(y_test.sum()),
            **smote_stats,
        }
        all_metrics.append(metrics)

        model.save(artifact_dir / f"{model_name}_lead_{lead}h.keras")

        pred_df = pd.DataFrame(
            {
                "stay_id": test_sids,
                target_col: y_test,
                f"p_{model_name}": p,
            }
        )
        pred_df.to_csv(artifact_dir / f"predictions_lead_{lead}h.csv", index=False)

        hist_df = pd.DataFrame(history.history)
        hist_df.insert(0, "epoch", np.arange(1, len(hist_df) + 1))
        hist_df.to_csv(artifact_dir / f"history_lead_{lead}h.csv", index=False)

    metrics_df = pd.DataFrame(all_metrics).sort_values("lead_hours")
    metrics_df.to_csv(artifact_dir / "metrics_by_lead.csv", index=False)

    print(metrics_df)
    print(f"Saved artifacts to: {artifact_dir}")
