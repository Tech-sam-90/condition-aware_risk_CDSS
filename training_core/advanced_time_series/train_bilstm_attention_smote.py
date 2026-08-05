from __future__ import annotations

from common_pipeline import parse_args, run_training_pipeline


def build_bilstm_attention_model(
    tf,
    seq_len: int,
    input_dim: int,
    learning_rate: float,
    lead_hours: int,
):
    inputs = tf.keras.layers.Input(shape=(seq_len, input_dim))

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True, dropout=0.2)
    )(inputs)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(32, return_sequences=True, dropout=0.2)
    )(x)

    attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.1)(x, x)
    x = tf.keras.layers.Add()([x, attn])
    x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"bilstm_attn_lead_{lead_hours}h")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model


if __name__ == "__main__":
    args = parse_args("bilstm_attention_smote2575")
    run_training_pipeline(model_name=args.model_name, build_model_fn=build_bilstm_attention_model, args=args)
