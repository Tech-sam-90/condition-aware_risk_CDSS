from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split


PROCESSED_DIR = Path("/home/ubuntu/condition-aware_risk_CDSS/data/processed")
OUTPUT_DIR = Path("/home/ubuntu/condition-aware_risk_CDSS/modeling/artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for LSTM baseline. Install with: pip install tensorflow"
        ) from exc
    return tf


def build_sequences(long_df, horizon_hours=24):
    vitals = ["heart_rate", "sbp", "map", "resp_rate", "spo2", "temp_f"]
    all_hours = pd.DataFrame({"hour_since_icu_admit": np.arange(horizon_hours)})

    sequences = []
    stay_ids = []

    for stay_id, grp in long_df.groupby("stay_id"):
        g = grp[["hour_since_icu_admit", *vitals]].copy()
        g = all_hours.merge(g, on="hour_since_icu_admit", how="left")
        g[vitals] = g[vitals].ffill().bfill()
        g[vitals] = g[vitals].fillna(g[vitals].median())
        sequences.append(g[vitals].values.astype("float32"))
        stay_ids.append(stay_id)

    X_seq = np.stack(sequences, axis=0)
    return stay_ids, X_seq


def build_labels(stay_ids, model_table):
    y_map = model_table.set_index("stay_id")["hospital_expire_flag"].astype(int).to_dict()
    y = np.array([y_map.get(s, 0) for s in stay_ids], dtype="int32")
    return y


def main():
    long_path = PROCESSED_DIR / "vitals_24h_long.csv"
    table_path = PROCESSED_DIR / "condition_model_table_v2_with_24h_vitals.csv"
    if not long_path.exists() or not table_path.exists():
        raise FileNotFoundError(
            "Missing processed files. Run feature_engineering/build_24h_vitals_features.py first."
        )

    tf = _load_tensorflow()

    long_df = pd.read_csv(long_path)
    model_table = pd.read_csv(table_path)

    stay_ids, X_seq = build_sequences(long_df)
    y = build_labels(stay_ids, model_table)

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
        epochs=30,
        batch_size=128,
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
