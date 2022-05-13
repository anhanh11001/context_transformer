from keras import callbacks, models
from transformer import make_transformer_model
from cnn import make_cnn_model
import matplotlib.pyplot as plt

# Configuration
from utils import print_line_divider

epochs = 500
batch_size = 32
validation_split = 0.2
optimizer = 'adam'
loss_function = "sparse_categorical_crossentropy"


# Model & Data
def get_model():
    return make_cnn_model((1, 2))


def get_data():
    pass


model = get_model()
print("Model Summary:")
print(model.summary())
(x_train, y_train, x_test, y_test) = get_data()

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
)

# Model evaluation
model = models.load_model("best_model.h5")

test_loss, test_acc = model.evaluate(x_test, y_test)
print_line_divider()
print("Test accuracy", test_acc)
print("Test loss", test_loss)

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
