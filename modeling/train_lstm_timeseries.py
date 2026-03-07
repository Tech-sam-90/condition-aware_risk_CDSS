from pathlib import Path
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split


PROCESSED_DIR = Path("/home/ubuntu/condition-aware_risk_CDSS/data/processed")
OUTPUT_DIR = Path("/home/ubuntu/condition-aware_risk_CDSS/modeling/artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VITALS = ["heart_rate", "sbp", "map", "resp_rate", "spo2", "temp_f"]
CONDITIONS = ["sepsis", "heart_failure", "ckd", "diabetes", "other"]
HORIZON_HOURS = 24
EPOCHS = int(os.getenv("LSTM_EPOCHS", "12"))
BATCH_SIZE = int(os.getenv("LSTM_BATCH_SIZE", "128"))


def _load_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for LSTM baseline. Install with: pip install tensorflow"
        ) from exc
    return tf


def condition_one_hot(condition: str) -> np.ndarray:
    idx = {name: i for i, name in enumerate(CONDITIONS)}
    arr = np.zeros((len(CONDITIONS),), dtype=np.float32)
    arr[idx.get(str(condition).lower(), idx["other"])] = 1.0
    return arr


def build_sequences(rolling_df: pd.DataFrame, horizon_hours: int = HORIZON_HOURS):
    all_hours = pd.DataFrame({"hour_from_icu": np.arange(horizon_hours)})

    global_medians = rolling_df[VITALS].median().to_dict()
    sequences = []
    stay_ids = []
    labels = []

    for stay_id, grp in rolling_df.groupby("stay_id"):
        g = grp[["hour_from_icu", "condition_input", "hospital_expire_flag", *VITALS]].copy()
        g = all_hours.merge(g, on="hour_from_icu", how="left")

        condition_val = (
            grp["condition_input"].dropna().astype(str).str.lower().iloc[0]
            if grp["condition_input"].notna().any()
            else "other"
        )
        y_val = int(pd.to_numeric(grp["hospital_expire_flag"], errors="coerce").fillna(0).max())

        g[VITALS] = g[VITALS].ffill().bfill()
        for col in VITALS:
            g[col] = g[col].fillna(float(global_medians.get(col, 0.0)))

        cond_vec = condition_one_hot(condition_val)
        cond_mat = np.tile(cond_vec, (len(g), 1))
        x_num = g[VITALS].to_numpy(dtype=np.float32)
        x = np.concatenate([x_num, cond_mat], axis=1)

        sequences.append(x)
        stay_ids.append(int(stay_id))
        labels.append(y_val)

    X_seq = np.stack(sequences, axis=0)
    y = np.array(labels, dtype="int32")
    return stay_ids, X_seq, y


def main():
    rolling_path = PROCESSED_DIR / "rolling_window_multicondition_timeseries.csv"
    if not rolling_path.exists():
        raise FileNotFoundError(
            "Missing rolling file. Run feature_engineering/build_rolling_window_timeseries.py first."
        )

    tf = _load_tensorflow()

    rolling_df = pd.read_csv(rolling_path)
    rolling_df["stay_id"] = pd.to_numeric(rolling_df["stay_id"], errors="coerce")
    rolling_df["hour_from_icu"] = pd.to_numeric(rolling_df["hour_from_icu"], errors="coerce")
    for col in VITALS:
        rolling_df[col] = pd.to_numeric(rolling_df[col], errors="coerce")

    rolling_df = rolling_df[(rolling_df["hour_from_icu"] >= 0) & (rolling_df["hour_from_icu"] < HORIZON_HOURS)]
    rolling_df = rolling_df.dropna(subset=["stay_id", "hour_from_icu", "condition_input"]) 

    stay_ids, X_seq, y = build_sequences(rolling_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y, test_size=0.2, random_state=42, stratify=y
    )

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(X_seq.shape[1], X_seq.shape[2])),
            tf.keras.layers.Masking(mask_value=0.0),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
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
            monitor="val_auc", mode="max", patience=5, restore_best_weights=True
        )
    ]

    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    p = model.predict(X_test, verbose=0).reshape(-1)

    metrics = {
        "model": "lstm_24h_vitals",
        "roc_auc": float(roc_auc_score(y_test, p)),
        "auprc": float(average_precision_score(y_test, p)),
        "brier": float(brier_score_loss(y_test, p)),
    }

    pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / "lstm_model_metrics.csv", index=False)
    model.save(OUTPUT_DIR / "lstm_24h_vitals.keras")

    print(metrics)
    print(f"Saved LSTM artifacts to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
