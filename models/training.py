import os.path

from keras import callbacks, models

from datahandler.constants import all_features, data_version
from datahandler.data_preprocessing import get_train_test_data
from models.log_writer import LogWriter
from transformer import make_transformer_model
from cnn import make_cnn_model_v1, make_cnn_model_v2
import matplotlib.pyplot as plt
from utils import print_line_divider

log_writer = LogWriter()

# Configuration
print("STARTING THE TRAINING PROCESS")
window_time_in_seconds = 2
window_size = 40
epochs = 20
batch_size = 32
validation_split = 2 / 9
optimizer = 'adam'
loss_function = "sparse_categorical_crossentropy"
supported_features = all_features

log_writer.write("Configuration", line_divider=True)
log_writer.write(
    f"""Window size: {window_size}
Window time in second: ${window_time_in_seconds}
Maximum number of epochs: ${epochs}
Batch size: ${batch_size}
Validation split: ${validation_split}
Optimizer: ${optimizer}
Loss function: ${loss_function}
Features used for training: ${str(supported_features)}"""
)

# Data
print_line_divider()
print("Preparing data...")
x_train, y_train, x_test, y_test = get_train_test_data(supported_features, window_time_in_seconds, window_size)
print("Train data shape: " + str(x_train.shape) + " | Train label shape: " + str(y_train.shape))
print("Test data shape: " + str(x_test.shape) + " | Test label shape: " + str(y_test.shape))
print_line_divider()
log_writer.write("Data", line_divider=True)
log_writer.write("Data loaded from version " + data_version)
log_writer.write(
    f"""Data training shape: ${x_train.shape}
Data testing shape: ${x_test.shape}"""
)

# Setting up model
name, model = make_cnn_model_v1(input_shape=(window_size, len(supported_features)))
# model = make_transformer_model(
#     input_shape=(window_size, len(supported_features)),
#     head_size=256,
#     num_heads=4,
#     ff_dim=4,
#     num_transformer_blocks=4,
#     mlp_units=[128],
#     mlp_dropout=0.4,
#     dropout=0.25,
# )
print("Model Summary:")
stringlist = []
model.summary(print_fn=lambda x: stringlist.append(x))
short_model_summary = "\n".join(stringlist)
print(short_model_summary)
print(print_line_divider())

# Logging model
log_writer.write("Model", line_divider=True)
log_writer.write("Model name: " + name)
log_writer.write(short_model_summary)

callbacks = [
    callbacks.ModelCheckpoint("best_model.h5", save_best_only=True, monitor="val_loss"),
    callbacks.ModelCheckpoint(log_writer.base_folder + "/model.h5", save_best_only=True, monitor="val_loss"),
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
log_writer.write("Result", line_divider=True)
log_writer.write("Highest validation accuracy: " + str(max(history.history['val_sparse_categorical_accuracy'])))

# Plotting
metric = "sparse_categorical_accuracy"
plt.figure()
plt.plot(history.history[metric])
plt.plot(history.history["val_" + metric])
plt.title("model " + metric)
plt.ylabel(metric, fontsize="large")
plt.xlabel("epoch", fontsize="large")
plt.legend(["train", "val"], loc="best")
plt.savefig(os.path.join(log_writer.base_folder, "Validation progress.png"))
plt.show()
plt.close()
log_writer.close()

# Model evaluation
model = models.load_model("best_model.h5")

test_loss, test_acc = model.evaluate(x_test, y_test)
print_line_divider()
# print("Test accuracy", test_acc)
# print("Test loss", test_loss)
