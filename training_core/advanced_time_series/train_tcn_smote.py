from __future__ import annotations

from common_pipeline import parse_args, run_training_pipeline


def build_tcn_model(tf, seq_len: int, input_dim: int, learning_rate: float, lead_hours: int):
    inputs = tf.keras.layers.Input(shape=(seq_len, input_dim))

    x = inputs
    for dilation in [1, 2, 4, 8]:
        residual = x
        x = tf.keras.layers.Conv1D(
            filters=64,
            kernel_size=3,
            dilation_rate=dilation,
            padding="causal",
            activation="relu",
        )(x)
        x = tf.keras.layers.Conv1D(
            filters=64,
            kernel_size=3,
            dilation_rate=dilation,
            padding="causal",
            activation="relu",
        )(x)

        if residual.shape[-1] != x.shape[-1]:
            residual = tf.keras.layers.Conv1D(64, kernel_size=1, padding="same")(residual)

        x = tf.keras.layers.Add()([x, residual])
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"tcn_lead_{lead_hours}h")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model


if __name__ == "__main__":
    args = parse_args("tcn_smote2575")
    run_training_pipeline(model_name=args.model_name, build_model_fn=build_tcn_model, args=args)
