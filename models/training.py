from keras import callbacks, models

from datahandler.constants import supported_features
from datahandler.data_preprocessing import get_train_test_data
from transformer import make_transformer_model
from cnn import make_cnn_model_v1, make_cnn_model_v2
import matplotlib.pyplot as plt

# Configuration
from utils import print_line_divider

print("STARTING THE TRAINING PROCESS")
window_time_in_seconds = 2
window_size = 40
epochs = 200
batch_size = 32
validation_split = 1 / 9
optimizer = 'adam'
loss_function = "sparse_categorical_crossentropy"

# Data
print_line_divider()
print("Preparing data...")
x_train, y_train, x_test, y_test = get_train_test_data(window_time_in_seconds, window_size)
print("Train data shape: " + str(x_train.shape) + " | Train label shape: " + str(y_train.shape))
print("Test data shape: " + str(x_test.shape) + " | Test label shape: " + str(y_test.shape))
print_line_divider()

# Setting up model
model = make_cnn_model_v1(input_shape=(window_size, len(supported_features)))
print("Model Summary:")
print(model.summary())
print(print_line_divider())

callbacks = [
    callbacks.ModelCheckpoint("best_model.h5", save_best_only=True, monitor="val_loss"),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001),
    callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
]
model.compile(
    optimizer=optimizer,
    loss=loss_function,
    metrics=["sparse_categorical_accuracy"],
)

# Training
print_line_divider()
print("Starting to train...")
history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_split=validation_split,
    verbose=1,
    shuffle=True
)
print("Highest validation accuracy: ", max(history.history['val_sparse_categorical_accuracy']))
# Plotting
metric = "sparse_categorical_accuracy"
plt.figure()
plt.plot(history.history[metric])
plt.plot(history.history["val_" + metric])
plt.title("model " + metric)
plt.ylabel(metric, fontsize="large")
plt.xlabel("epoch", fontsize="large")
plt.legend(["train", "val"], loc="best")
plt.show()
plt.close()

# Model evaluation
model = models.load_model("best_model.h5")

test_loss, test_acc = model.evaluate(x_test, y_test)
print_line_divider()
print("Test accuracy", test_acc)
print("Test loss", test_loss)
