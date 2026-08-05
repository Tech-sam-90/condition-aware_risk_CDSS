from __future__ import annotations

from common_pipeline import parse_args, run_training_pipeline


def gated_residual_network(tf, x, units: int, dropout: float):
    residual = tf.keras.layers.Dense(units)(x)
    h = tf.keras.layers.Dense(units, activation="elu")(x)
    h = tf.keras.layers.Dropout(dropout)(h)
    h = tf.keras.layers.Dense(units)(h)

    gate = tf.keras.layers.Dense(units, activation="sigmoid")(x)
    h = tf.keras.layers.Multiply()([h, gate])

    out = tf.keras.layers.Add()([residual, h])
    out = tf.keras.layers.LayerNormalization()(out)
    return out


def build_tft_style_model(
    tf,
    seq_len: int,
    input_dim: int,
    learning_rate: float,
    lead_hours: int,
):
    inputs = tf.keras.layers.Input(shape=(seq_len, input_dim))

    x = tf.keras.layers.Dense(64)(inputs)
    x = gated_residual_network(tf, x, units=64, dropout=0.1)

    # Temporal processing
    x = tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)(x)

    # Temporal fusion style self-attention
    attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.1)(x, x)
    x = tf.keras.layers.Add()([x, attn])
    x = tf.keras.layers.LayerNormalization()(x)

    x = gated_residual_network(tf, x, units=64, dropout=0.1)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"tft_style_lead_{lead_hours}h")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model


if __name__ == "__main__":
    args = parse_args("tft_style_smote2575")
    run_training_pipeline(model_name=args.model_name, build_model_fn=build_tft_style_model, args=args)
