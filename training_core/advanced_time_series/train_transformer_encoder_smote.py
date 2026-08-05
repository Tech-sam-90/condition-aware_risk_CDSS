from __future__ import annotations

from common_pipeline import parse_args, run_training_pipeline


def transformer_block(tf, x, num_heads: int, key_dim: int, ff_dim: int, dropout: float):
    attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout)(x, x)
    x = tf.keras.layers.Add()([x, attn])
    x = tf.keras.layers.LayerNormalization()(x)

    ff = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
    ff = tf.keras.layers.Dropout(dropout)(ff)
    ff = tf.keras.layers.Dense(x.shape[-1])(ff)

    x = tf.keras.layers.Add()([x, ff])
    x = tf.keras.layers.LayerNormalization()(x)
    return x


def build_transformer_encoder_model(
    tf,
    seq_len: int,
    input_dim: int,
    learning_rate: float,
    lead_hours: int,
):
    inputs = tf.keras.layers.Input(shape=(seq_len, input_dim))

    d_model = 64
    proj = tf.keras.layers.Dense(d_model)(inputs)

    positions = tf.range(start=0, limit=seq_len, delta=1)
    pos_embed = tf.keras.layers.Embedding(input_dim=seq_len, output_dim=d_model)(positions)
    x = proj + pos_embed

    x = transformer_block(tf, x, num_heads=4, key_dim=16, ff_dim=128, dropout=0.1)
    x = transformer_block(tf, x, num_heads=4, key_dim=16, ff_dim=128, dropout=0.1)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"transformer_lead_{lead_hours}h")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model


if __name__ == "__main__":
    args = parse_args("transformer_encoder_smote2575")
    run_training_pipeline(
        model_name=args.model_name,
        build_model_fn=build_transformer_encoder_model,
        args=args,
    )
