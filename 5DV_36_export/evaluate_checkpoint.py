#!/usr/bin/env python3
import math
import os

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
MODEL_PATH = "best_baseline_lr1e3.keras"


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


def main():
    val_ds = DataLoader(os.path.join(DATA_DIR, "val"), CLASS_MAP, batch_size=BATCH_SIZE)
    model = tf.keras.models.load_model(MODEL_PATH)

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


if __name__ == "__main__":
    main()