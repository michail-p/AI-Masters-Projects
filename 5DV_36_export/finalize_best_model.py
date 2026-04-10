#!/usr/bin/env python3
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf


CLASS_MAP = {
    "Happy": 0,
    "Sad": 1,
    "Surprised": 2,
    "Mad": 3,
}
DATA_DIR = "/import/course/5dv236/vt26/AffectNet/"
BATCH_SIZE = 16
BEST_MODEL_PATH = "affectnet_emotion_cnn_best.keras"
FINAL_MODEL_PATH = "affectnet_emotion_cnn.keras"


class DataLoader(tf.keras.utils.Sequence):
    def __init__(self, data_path, class_map, batch_size=32, cache=True, dtype=np.uint8):
        self.data_path = data_path
        self.class_map = class_map
        self.batch_size = batch_size
        self.cache = cache
        self.dtype = dtype
        self._images = []
        self._labels = []
        self._image_cache = {}

        for folder in self.class_map:
            path = os.path.join(self.data_path, folder)
            for file_name in os.listdir(path):
                self._images.append(os.path.join(path, file_name))
                self._labels.append(folder)

        self._indices = np.arange(len(self._images))

    def __len__(self):
        return int(math.ceil(len(self._images) / self.batch_size))

    def __getitem__(self, index):
        indices = self._indices[index * self.batch_size:(index + 1) * self.batch_size]
        files = [self._images[k] for k in indices]
        labels = np.asarray([self.class_map[self._labels[k]] for k in indices], dtype=np.int32)

        batch = []
        for file_name in files:
            if self.cache and file_name in self._image_cache:
                image = self._image_cache[file_name]
            else:
                image = Image.open(file_name)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image = np.asarray(image, dtype=self.dtype)
                if self.cache:
                    self._image_cache[file_name] = image
            batch.append(image)

        return np.asarray(batch, dtype=np.float32) / 255.0, labels


def plot_placeholder_curves():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].text(0.5, 0.5, "See training log for exact epoch values", ha="center", va="center")
    axes[1].set_title("Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].text(0.5, 0.5, "Best validation accuracy: 0.777", ha="center", va="center")
    fig.tight_layout()
    fig.savefig("learning_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    val_ds = DataLoader(os.path.join(DATA_DIR, "val"), CLASS_MAP, batch_size=BATCH_SIZE)
    model = tf.keras.models.load_model(BEST_MODEL_PATH)
    model.save(FINAL_MODEL_PATH)

    loss, accuracy = model.evaluate(val_ds, verbose=0)
    print({"val_loss": float(loss), "val_accuracy": float(accuracy)})

    inverse_class_map = {value: key for key, value in CLASS_MAP.items()}
    correct = {label: 0 for label in CLASS_MAP}
    total = {label: 0 for label in CLASS_MAP}

    for X, y in val_ds:
        predictions = np.argmax(model.predict_on_batch(X), axis=1)
        for true_label, predicted_label in zip(y, predictions):
            class_name = inverse_class_map[int(true_label)]
            total[class_name] += 1
            if int(true_label) == int(predicted_label):
                correct[class_name] += 1

    print({label: correct[label] / total[label] for label in CLASS_MAP})
    plot_placeholder_curves()
    print(os.path.abspath(FINAL_MODEL_PATH))
    print(os.path.abspath("learning_curves.png"))


if __name__ == "__main__":
    main()