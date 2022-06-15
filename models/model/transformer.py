from keras import layers, models

from datahandler.constants import location_labels, activity_labels

TRANSFORMER_V1_NAME = "Simple Transformer model v1 from Keras tutorial"
TRANSFORMER_V2_NAME = "Simplified Transformer model v2"
TRANSFORMER_V3_NAME = "Simple Transformer model v3 with multi-task learning"


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size,
        num_heads=num_heads,
        dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="gelu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def make_transformer_model_v1(
        input_shape,
        head_size,
        num_heads,
        ff_dim,
        num_transformer_blocks,
        mlp_units,
        dropout=0,
        mlp_dropout=0,
):
    num_classes = len(location_labels)
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return TRANSFORMER_V1_NAME, models.Model(inputs, outputs)


def make_transformer_model_v3(
        input_shape,
        head_size,
        num_heads,
        ff_dim,
        num_transformer_blocks,
        dropout=0,
        mlp_dropout=0,
):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)

    dim = 128
    dense_context = layers.Dense(dim, activation="relu")(x)
    dropout_context = layers.Dropout(mlp_dropout)(dense_context)
    output_context = layers.Dense(len(location_labels), activation="softmax", name='context_output')(dropout_context)

    dense_activity = layers.Dense(dim, activation="relu")(x)
    dropout_activity = layers.Dropout(mlp_dropout)(dense_activity)
    output_activity = layers.Dense(len(activity_labels), activation="softmax", name='activity_output')(dropout_activity)

    outputs = [output_context, output_activity]

    return TRANSFORMER_V3_NAME, models.Model(inputs, outputs)


def make_transformer_model_v2(input_shape):
    num_classes = len(location_labels)
    inputs = layers.Input(shape=input_shape)
    x = transformer_encoder(
        inputs=inputs,
        head_size=64,
        num_heads=4,
        ff_dim=64,
        dropout=0
    )

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return TRANSFORMER_V2_NAME, models.Model(inputs, outputs)
