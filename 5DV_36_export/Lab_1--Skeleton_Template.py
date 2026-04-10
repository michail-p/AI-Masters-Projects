#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Template file for Assignment 1 -- Happy, Sad, Surprised, or Mad?

Last updated: 2026-03-23
"""
import os
import time
import random
import math

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras

SEED = 2026
class_map = {"Happy": 0,
             "Sad": 1,
             "Surprised": 2,
             "Mad": 3}
cache = True

# You will have to change these two
n_epochs = 30
batch_size = 16
learning_rate = 5e-4
model_output_path = "affectnet_emotion_cnn.keras"
best_model_output_path = "affectnet_emotion_cnn_best.keras"

# Directory where the data are stored
data_dir = "/import/course/5dv236/vt26/AffectNet/"
print(f"Loading data from {data_dir}")

# Set seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.optimizer.set_jit(False)

# Check GPU availability (it will be very slow without a GPU...)
gpus = tf.config.experimental.list_physical_devices("GPU")
print()
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print("GPU(s) available. Training will be lightning fast!")
else:
    print("No GPU(s) available. Training will be very slow ...")

# For pretty plots...
PLOT_SETTINGS = {"text.usetex": False,
                 "font.family": "serif",
                 "figure.figsize": (8.0, 6.0),
                 "font.size": 16,
                 "axes.labelsize": 16,
                 "legend.fontsize": 14,
                 "xtick.labelsize": 14,
                 "ytick.labelsize": 14,
                 "axes.titlesize": 24,
                 "lines.linewidth": 2.0,
                 }
plt.rcParams.update(PLOT_SETTINGS)


def preprocess_batch(images):
    """Convert a list of uint8 images to normalized float32 tensors."""
    batch = np.asarray(images, dtype=np.float32) / 255.0
    return batch


def labels_to_array(labels):
    """Convert labels to a NumPy array with a stable dtype."""
    return np.asarray(labels, dtype=np.int32)


def conv_block(x, filters, dropout_rate):
    """Apply two regularized convolution layers followed by pooling."""
    x = keras.layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    return x


def build_model(input_shape=(64, 64, 3), n_classes=4):
    """Build a stronger but still GPU-safe CNN for four-class classification."""
    inputs = keras.Input(shape=input_shape)

    augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.06),
        keras.layers.RandomZoom(0.08),
        keras.layers.RandomContrast(0.08),
    ], name="augmentation")

    x = augmentation(inputs)
    x = conv_block(x, 24, 0.10)
    x = conv_block(x, 48, 0.15)
    x = conv_block(x, 96, 0.20)
    x = conv_block(x, 128, 0.25)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.30)(x)
    outputs = keras.layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="affectnet_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def evaluate_dataset(model, dataset):
    """Evaluate a dataset batch-by-batch and return mean loss and accuracy."""
    loss, accuracy = model.evaluate(dataset, verbose=0)
    return {"loss": float(loss), "accuracy": float(accuracy)}


def per_class_accuracy(model, dataset):
    """Compute per-class accuracies for a dataset."""
    correct = {label: 0 for label in class_map}
    total = {label: 0 for label in class_map}
    inverse_class_map = {value: key for key, value in class_map.items()}

    for X, y in dataset:
        predictions = model.predict_on_batch(X)
        predicted_labels = np.argmax(predictions, axis=1)

        for true_label, predicted_label in zip(y, predicted_labels):
            class_name = inverse_class_map[int(true_label)]
            total[class_name] += 1
            if int(true_label) == int(predicted_label):
                correct[class_name] += 1

    return {
        class_name: (correct[class_name] / total[class_name])
        for class_name in class_map
    }


def plot_learning_curves(training_history, validation_history):
    """Plot loss and accuracy curves for training and validation."""
    epochs = np.arange(1, len(training_history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, training_history["loss"], label="Training")
    axes[0].plot(epochs, validation_history["loss"], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()

    axes[1].plot(epochs, training_history["accuracy"], label="Training")
    axes[1].plot(epochs, validation_history["accuracy"], label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curves")
    axes[1].legend()

    fig.tight_layout()
    return fig


class DataLoader(keras.utils.Sequence):
    """A simple data loader for Assignment 1.

    Note: We advice against making changes to the data loader!
    """

    def __init__(self,
                 data_path,
                 class_map,
                 batch_size=32,
                 cache=True,
                 random_state=None,
                 dtype=np.uint8,
                 ):

        self.data_path = data_path
        self.class_map = class_map
        self.batch_size = max(1, int(batch_size))
        self.cache = bool(cache)
        if random_state is None:
            self.random_state = np.random
        elif isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(random_state)
        self.dtype = dtype

        if self.data_path is None:
            raise ValueError('The data path is not defined.')

        if not os.path.isdir(self.data_path):
            raise ValueError('The data path is incorrectly defined.')

        if not isinstance(self.class_map, dict):
            raise ValueError('The folder map is not a dictionary.')

        # Read the files in all subfolders
        self._file_idx = 0
        self._images = []
        self._labels = []
        for folder in self.class_map:
            path = os.path.join(self.data_path, folder)
            for file in os.listdir(path):
                file_path = os.path.join(path, file)
                self._images.append(file_path)
                self._labels.append(folder)

        self._image_cache = dict()

        self.on_epoch_end()

    def __len__(self):
        """Get the number of mini-batches per epoch."""
        return int(math.ceil(len(self._images) / self.batch_size))

    def __getitem__(self, index):
        """Get one batch of data."""
        # Generate indices of the batch
        indices = self._indices[
            index * self.batch_size:(index + 1) * self.batch_size]

        # Find the next set of file indices
        minibatch_files = [self._images[k] for k in indices]
        minibatch_labels = [self.class_map[self._labels[k]] for k in indices]

        # Load up the corresponding minibatch
        minibatch = self.__load_minibatch(minibatch_files)

        return preprocess_batch(minibatch), labels_to_array(minibatch_labels)

    def on_epoch_end(self):
        """Update indices after each epoch."""
        self._indices = np.arange(len(self._images))
        self.random_state.shuffle(self._indices)

    def __load_image(self, file):
        """Load a single image from file."""
        im = Image.open(file)
        if im.mode != "RGB":
            im = im.convert("RGB")
        im = np.asarray(im, dtype=self.dtype)

        return im

    def __load_minibatch(self, minibatch_files):
        """Load the next minibatch of samples."""
        minibatch = [None] * len(minibatch_files)
        for i, file in enumerate(minibatch_files):
            if self.cache:
                if file in self._image_cache:
                    im = self._image_cache[file]
                else:
                    im = self.__load_image(file)
                    self._image_cache[file] = im
            else:
                im = self.__load_image(file)

            minibatch[i] = im

        return minibatch

# NOTE: We will use a clean version of the data loader, so you should not rely
#       on any changes to it. Simply, don't change the data loader!


# Create the data loaders
train_ds = DataLoader(os.path.join(data_dir, "train/"),
                      class_map=class_map,
                      batch_size=batch_size,
                      cache=cache,
                      )
val_ds = DataLoader(os.path.join(data_dir, "val/"),
                    class_map=class_map,
                    batch_size=batch_size,
                    cache=cache,
                    )
# Do not use the test data in any way until the very end, when you fill in the
# values in your report just before handing it in!
test_ds = DataLoader(os.path.join(data_dir, "test/"),
                     class_map=class_map,
                     batch_size=batch_size,
                     cache=cache,
                     )

# A quick summary of the data:
print(f"Number of training mini-batches: {len(train_ds)}")
print(f"Number of training images      : {len(train_ds._indices)}")
print(f"Number of validation images    : {len(val_ds._indices)}")
print(f"Number of test images          : {len(test_ds._indices)}")


# Plot a few of the training images
fig = plt.figure(figsize=(12, 5))
fig.subplots_adjust(top=0.995,
                    bottom=0.005,
                    left=0.025,
                    right=0.995,
                    wspace=0.05,
                    hspace=0.0125)
M, N = 4, 10
axs = []
for m in range(M):
    axs.append([])
    for n in range(N):
        ax = plt.subplot2grid((M, N), (m, n), rowspan=1, colspan=1)
        ax.xaxis.set_ticklabels([])
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks([])
        axs[m].append(ax)

imgs = []
lbls = []
for i in range(3):
    imgs.extend(train_ds[i][0])
    lbls.extend(train_ds[i][1])
indices = [0] * 4
for i in range(len(imgs)):
    y = lbls[i]
    if indices[y] < N:
        axs[y][indices[y]].imshow(imgs[i].astype(int))  # int [0,...,255]
        indices[y] += 1
for m in range(M):
    label = list(train_ds.class_map.keys())[
        list(train_ds.class_map.values()).index(m)]
    axs[m][0].set_ylabel(f"{label}")


# Define and compile your model here. Don't forget to use accuracy as a metric.
sample_batch, _ = train_ds[0]
input_shape = sample_batch.shape[1:]
model = build_model(input_shape=input_shape, n_classes=len(class_map))
model.summary()

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=best_model_output_path,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1,
)
reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
    monitor="val_accuracy",
    mode="max",
    factor=0.5,
    patience=4,
    min_lr=1e-5,
    verbose=1,
)
early_stopping_callback = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    patience=8,
    restore_best_weights=True,
    verbose=1,
)

class_weights = {
    class_map["Happy"]: 1.0,
    class_map["Sad"]: 1.4,
    class_map["Surprised"]: 1.0,
    class_map["Mad"]: 1.8,
}

time_ = time.time()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=n_epochs,
    callbacks=[checkpoint_callback,
               reduce_lr_callback,
               early_stopping_callback],
    class_weight=class_weights,
    verbose=2,
)

training_history = {
    "loss": [float(value) for value in history.history["loss"]],
    "accuracy": [float(value) for value in history.history["accuracy"]],
}
validation_history = {
    "loss": [float(value) for value in history.history["val_loss"]],
    "accuracy": [float(value) for value in history.history["val_accuracy"]],
}

time_passed = time.time() - time_
print(f"Training done in {int(time_passed // 60):.0f} minutes "
      f"{int(time_passed % 60):.0f} seconds")

validation_class_accuracy = per_class_accuracy(model, val_ds)
print("Validation per-class accuracy:")
for class_name, accuracy in validation_class_accuracy.items():
    print(f"  {class_name:10s}: {accuracy:.4f}")

# Note: We should never evaluate our model on the test data before we have
#       chosen a _final model_. This means you should not run the below code
#       until you are done and ready to hand in the assignment. It will be
#       tempting, but I repeat: Do not evaluate your model on the test data
#       until you are completely done and have a final model. It is this one
#       _final model_ that you evaluate with the test data. If you do anything
#       to your model and evaluate it again, there is no value in this
#       evaluation any more.
if False:
    test_metrics = evaluate_dataset(model, test_ds)
    test_class_accuracy = per_class_accuracy(model, test_ds)
    print(f"Final test results {test_metrics}")
    print(f"Final test per-class accuracy {test_class_accuracy}")


# Plot the training and validation curves
learning_curve_figure = plot_learning_curves(training_history,
                                             validation_history)
learning_curve_figure.savefig("learning_curves.png", dpi=200,
                              bbox_inches="tight")
plt.close(learning_curve_figure)

# Save the model to file using model.save(...)
model.save(model_output_path)
print(f"Saved model to {os.path.abspath(model_output_path)}")
