from keras import layers, models

from datahandler.constants import location_labels


def make_cnn_model(input_shape):
    num_classes = len(location_labels)
    input_layer = layers.Input(input_shape)

    conv1 = layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.ReLU()(conv1)

    conv2 = layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.ReLU()(conv2)

    conv3 = layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.ReLU()(conv3)

    gap = layers.GlobalAveragePooling1D()(conv3)

    output_layer = layers.Dense(num_classes, activation="softmax")(gap)

    return models.Model(inputs=input_layer, outputs=output_layer)
