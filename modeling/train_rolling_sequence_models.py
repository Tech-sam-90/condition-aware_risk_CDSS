from pathlib import Path
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split


PROCESSED_DIR = Path("/home/ubuntu/condition-aware_risk_CDSS/data/processed")
ART_DIR = Path("/home/ubuntu/condition-aware_risk_CDSS/modeling/artifacts/rolling_sequence")
ART_DIR.mkdir(parents=True, exist_ok=True)

VITAL_COLS = ["heart_rate", "sbp", "map", "resp_rate", "spo2", "temp_f"]
CONDITIONS = ["sepsis", "heart_failure", "ckd", "diabetes", "other"]
SEQ_LEN = 24
NEGATIVE_KEEP_PROB = float(os.getenv("SEQUENCE_NEGATIVE_KEEP_PROB", "0.2"))
EPOCHS = int(os.getenv("SEQUENCE_EPOCHS", "8"))
BATCH_SIZE = int(os.getenv("SEQUENCE_BATCH_SIZE", "256"))


def load_tf():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required. Install with: pip install tensorflow"
        ) from exc
    return tf


def condition_one_hot(series: pd.Series) -> np.ndarray:
    idx = {name: i for i, name in enumerate(CONDITIONS)}
    arr = np.zeros((len(series), len(CONDITIONS)), dtype=np.float32)
    for row_idx, val in enumerate(series.fillna("other").astype(str).str.lower().tolist()):
        arr[row_idx, idx.get(val, idx["other"])] = 1.0
    return arr


def iter_sequences(
    df: pd.DataFrame,
    stay_ids: list,
    target_col: str,
    seq_len: int,
    neg_keep_prob: float,
    seed: int,
):
    rng = np.random.default_rng(seed)
    for stay_id in stay_ids:
        group = df[df["stay_id"] == stay_id].sort_values("hour_from_icu").reset_index(drop=True)
        if len(group) < seq_len:
            continue

        cond_ohe = condition_one_hot(group["condition_input"])
        y_vals = group[target_col].fillna(0).astype(int).to_numpy()
        x_num_all = group[VITAL_COLS].to_numpy(dtype=np.float32)

        for end_idx in range(seq_len - 1, len(group)):
            y = int(y_vals[end_idx])
            if y == 0 and rng.random() > neg_keep_prob:
                continue

            start = end_idx - seq_len + 1
            x_num = x_num_all[start : end_idx + 1]
            x_cond = cond_ohe[start : end_idx + 1]
            x = np.concatenate([x_num, x_cond], axis=1).astype(np.float32)
            yield x, np.float32(y)


def count_sequences(df: pd.DataFrame, stay_ids: list, target_col: str, seq_len: int, neg_keep_prob: float) -> int:
    total = 0
    for stay_id in stay_ids:
        group = df[df["stay_id"] == stay_id]
        n = len(group)
        if n < seq_len:
            continue
        positives = int(group[target_col].fillna(0).astype(int).iloc[seq_len - 1 :].sum())
        windows = n - seq_len + 1
        negatives = windows - positives
        total += positives + int(negatives * neg_keep_prob)
    return max(total, 1)


def build_eval_arrays(df: pd.DataFrame, stay_ids: list, target_col: str, seq_len: int):
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
        raise RuntimeError("No evaluation sequences available.")

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    sids = np.concatenate(sid_parts, axis=0)
    return X, y, sids


def train_for_lead(tf, df: pd.DataFrame, lead_hours: int):
    target_col = f"target_event_in_{lead_hours}h"
    stay_level = df.groupby("stay_id", as_index=False)[target_col].max().rename(columns={target_col: "y"})
    stay_level["y"] = stay_level["y"].fillna(0).astype(int)
    train_stays, test_stays = train_test_split(
        stay_level,
        test_size=0.2,
        random_state=42,
        stratify=stay_level["y"],
    )

    train_ids_all = train_stays["stay_id"].astype(int).tolist()
    test_ids = test_stays["stay_id"].astype(int).tolist()

    train_stay_level = stay_level[stay_level["stay_id"].isin(train_ids_all)].copy()
    train_only, val_only = train_test_split(
        train_stay_level,
        test_size=0.15,
        random_state=42,
        stratify=train_stay_level["y"],
    )
    train_ids = train_only["stay_id"].astype(int).tolist()
    val_ids = val_only["stay_id"].astype(int).tolist()

    input_dim = len(VITAL_COLS) + len(CONDITIONS)
    train_count = count_sequences(df, train_ids, target_col, SEQ_LEN, NEGATIVE_KEEP_PROB)

    train_ds = tf.data.Dataset.from_generator(
        lambda: iter_sequences(
            df,
            train_ids,
            target_col=target_col,
            seq_len=SEQ_LEN,
            neg_keep_prob=NEGATIVE_KEEP_PROB,
            seed=42 + lead_hours,
        ),
        output_signature=(
            tf.TensorSpec(shape=(SEQ_LEN, input_dim), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        ),
    )
    train_ds = train_ds.shuffle(4096).repeat().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_count = count_sequences(df, val_ids, target_col, SEQ_LEN, 1.0)
    val_ds = tf.data.Dataset.from_generator(
        lambda: iter_sequences(
            df,
            val_ids,
            target_col=target_col,
            seq_len=SEQ_LEN,
            neg_keep_prob=1.0,
            seed=1042 + lead_hours,
        ),
        output_signature=(
            tf.TensorSpec(shape=(SEQ_LEN, input_dim), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        ),
    )
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(SEQ_LEN, input_dim)),
            tf.keras.layers.Masking(mask_value=0.0),
            tf.keras.layers.GRU(48, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.GRU(24),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=2, restore_best_weights=True
        )
    ]

    steps_per_epoch = max(train_count // BATCH_SIZE, 50)
    model.fit(
        train_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=max(val_count // BATCH_SIZE, 10),
        callbacks=callbacks,
        verbose=1,
    )

    X_test, y_test, test_sequence_stays = build_eval_arrays(df, test_ids, target_col, SEQ_LEN)
    p = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0).reshape(-1)

    metrics = {
        "lead_hours": lead_hours,
        "roc_auc": float(roc_auc_score(y_test, p)),
        "auprc": float(average_precision_score(y_test, p)),
        "brier": float(brier_score_loss(y_test, p)),
        "n_train_rows": int(train_count),
        "n_test_rows": int(len(X_test)),
        "n_test_pos": int(y_test.sum()),
    }

    pred_df = pd.DataFrame(
        {
            "stay_id": test_sequence_stays,
            f"target_event_in_{lead_hours}h": y_test,
            "p_gru": p,
        }
    )

    model.save(ART_DIR / f"gru_lead_{lead_hours}h.keras")
    pred_df.to_csv(ART_DIR / f"predictions_lead_{lead_hours}h.csv", index=False)

    return metrics


def main():
    data_path = PROCESSED_DIR / "rolling_window_multicondition_timeseries.csv"
    if not data_path.exists():
        raise FileNotFoundError(
            f"{data_path} not found. Run feature_engineering/build_rolling_window_timeseries.py first."
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
    df = df.dropna(subset=["stay_id", "hour_from_icu", "condition_input"])

    tf = load_tf()

    all_metrics = []
    for lead in [1, 2, 3]:
        m = train_for_lead(tf, df, lead_hours=lead)
        all_metrics.append(m)

    metrics_df = pd.DataFrame(all_metrics).sort_values("lead_hours")
    metrics_df.to_csv(ART_DIR / "metrics_by_lead.csv", index=False)

    print(metrics_df)
    print(f"Saved rolling sequence artifacts to: {ART_DIR}")


if __name__ == "__main__":
    main()
