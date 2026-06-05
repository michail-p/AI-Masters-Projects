#!/usr/bin/env python3
"""Evaluation script - loads trained model and generates all output plots."""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# Configuration
DATA_DIR = '/import/course/5dv236/vt26/AffectNet'
OUTPUT_DIR = os.path.expanduser('~/workspace/lab3_output')
CLASSES = ['Happy', 'Sad', 'Surprised', 'Mad']
NUM_CLASSES = len(CLASSES)
IMG_SIZE = 112
LATENT_DIM = 128


# Data loading
def load_split(split):
    images, labels = [], []
    for i, cls in enumerate(CLASSES):
        cls_dir = os.path.join(DATA_DIR, split, cls)
        for fname in sorted(os.listdir(cls_dir)):
            fpath = os.path.join(cls_dir, fname)
            if not os.path.isfile(fpath):
                continue
            images.append(np.array(Image.open(fpath).convert('RGB'), dtype=np.float32) / 255.0)
            labels.append(i)
    return np.array(images), np.array(labels)


class Sampling(keras.layers.Layer):
    def call(self, inputs):
        mu, log_var = inputs
        return mu + tf.exp(0.5 * log_var) * tf.random.normal(tf.shape(mu))


def build_encoder():
    img_in = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    label_in = keras.Input(shape=(NUM_CLASSES,))
    x = img_in
    for filters in [32, 64, 128, 256]:
        x = keras.layers.Conv2D(filters, 4, strides=2, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Concatenate()([x, label_in])
    x = keras.layers.Dense(512, activation='relu')(x)
    mu = keras.layers.Dense(LATENT_DIM, name='mu')(x)
    log_var = keras.layers.Dense(LATENT_DIM, name='log_var')(x)
    z = Sampling()([mu, log_var])
    return keras.Model([img_in, label_in], [mu, log_var, z], name='encoder')


def build_decoder():
    z_in = keras.Input(shape=(LATENT_DIM,))
    label_in = keras.Input(shape=(NUM_CLASSES,))
    x = keras.layers.Concatenate()([z_in, label_in])
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dense(7 * 7 * 256, activation='relu')(x)
    x = keras.layers.Reshape((7, 7, 256))(x)
    for filters in [128, 64, 32]:
        x = keras.layers.Conv2DTranspose(filters, 4, strides=2, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
    x = keras.layers.Conv2DTranspose(3, 4, strides=2, padding='same', activation='sigmoid')(x)
    return keras.Model([z_in, label_in], x, name='decoder')


print("Loading data...")
X_train, y_train = load_split('train')
X_val, y_val = load_split('val')
y_train_oh = tf.one_hot(y_train, NUM_CLASSES).numpy()
y_val_oh = tf.one_hot(y_val, NUM_CLASSES).numpy()
print(f"Train: {X_train.shape}, Val: {X_val.shape}")

encoder = build_encoder()
decoder = build_decoder()

print("Loading weights...")
encoder.load_weights(os.path.join(OUTPUT_DIR, 'encoder.weights.h5'))
decoder.load_weights(os.path.join(OUTPUT_DIR, 'decoder.weights.h5'))
print("Weights loaded.")

# Task 1: Reconstructions (4 per class = 16 total)
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
for j in range(4):
    axes[0, j].set_title('Original', fontsize=9)
    axes[0, j + 4].set_title('Reconstructed', fontsize=9)

for i, cls in enumerate(CLASSES):
    mask = y_val == i
    imgs = X_val[mask][:4]
    imgs_t = tf.constant(imgs)
    labs_t = tf.constant(y_val_oh[mask][:4])
    _, _, z = encoder([imgs_t, labs_t], training=False)
    recon = decoder([z, labs_t], training=False).numpy()
    for j in range(4):
        axes[i, j].imshow(imgs[j])
        axes[i, j].axis('off')
        axes[i, j + 4].imshow(np.clip(recon[j], 0, 1))
        axes[i, j + 4].axis('off')
    axes[i, 0].set_ylabel(cls, fontsize=11, rotation=0, labelpad=50)

plt.suptitle('Reconstructed Images (4 per class)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'reconstructions.png'), dpi=150)
plt.close()
print("Saved reconstructions.png")

# Task 2.1: Encode Happy, decode with Sad conditioning
happy_img = X_val[y_val == 0][:1]
happy_img_t = tf.constant(happy_img)
happy_lab = tf.constant(y_val_oh[y_val == 0][:1])
sad_lab = tf.one_hot([1], NUM_CLASSES)

_, _, z_h = encoder([happy_img_t, happy_lab], training=False)
recon_h = decoder([z_h, happy_lab], training=False).numpy()
recon_s = decoder([z_h, sad_lab], training=False).numpy()

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
titles = ['Original (Happy)', 'Recon (Happy cond.)', 'Same latent, Sad cond.']
for ax, img, t in zip(axes,
                       [happy_img[0], np.clip(recon_h[0], 0, 1), np.clip(recon_s[0], 0, 1)],
                       titles):
    ax.imshow(img)
    ax.set_title(t)
    ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'happy_to_sad_direct.png'), dpi=150)
plt.close()
print("Saved happy_to_sad_direct.png")

# Task 2.2: Mean-latent interpolation Happy to Sad
def mean_latent(cls_idx):
    mask = y_train == cls_idx
    mus = []
    for i in range(0, mask.sum(), 256):
        mu, _, _ = encoder([tf.constant(X_train[mask][i:i + 256]),
                            tf.constant(y_train_oh[mask][i:i + 256])],
                           training=False)
        mus.append(mu.numpy())
    return np.concatenate(mus).mean(axis=0)


happy_mean = mean_latent(0)
sad_mean = mean_latent(1)

fig, axes = plt.subplots(1, 10, figsize=(20, 2.5))
for idx, alpha in enumerate(np.linspace(0, 1, 10)):
    z_interp = tf.constant(((1 - alpha) * happy_mean + alpha * sad_mean)[np.newaxis, :])
    img = decoder([z_interp, sad_lab], training=False).numpy()[0]
    axes[idx].imshow(np.clip(img, 0, 1))
    axes[idx].set_title(f'α={alpha:.2f}', fontsize=9)
    axes[idx].axis('off')

plt.suptitle('Latent Interpolation: Happy mean → Sad mean (Sad conditioning)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'interpolation.png'), dpi=150)
plt.close()
print("Saved interpolation.png")

print(f"\nAll outputs saved to {OUTPUT_DIR}")
