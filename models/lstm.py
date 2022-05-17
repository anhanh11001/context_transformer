from keras import layers, Sequential

from datahandler.constants import location_labels

LSTM_V1_NAME = "Simple LSTM model v1 from Keras tutorial"


def make_lstm_model_v1(input_shape):
    num_classes = len(location_labels)
    model = Sequential()
    model.add(layers.LSTM(100, input_shape=input_shape))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return LSTM_V1_NAME, model