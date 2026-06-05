import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# Configuration
DATA_DIR = "/import/course/5dv236/vt26/AffectNet"
OUTPUT_DIR = os.path.expanduser("~/workspace/lab3_output")
CLASSES = ["Happy", "Sad", "Surprised", "Mad"]
NUM_CLASSES = len(CLASSES)
IMG_SIZE = 112
LATENT_DIM = 128
BATCH_SIZE = 64
EPOCHS = 120
LR = 1e-3
LR_MIN = 1e-5
BETA_MAX = 0.15
BETA_WARMUP = 20

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_split(split):
    images, labels = [], []
    for i, cls in enumerate(CLASSES):
        cls_dir = os.path.join(DATA_DIR, split, cls)
        for fname in sorted(os.listdir(cls_dir)):
            fpath = os.path.join(cls_dir, fname)
            if not os.path.isfile(fpath):
                continue
            images.append(np.array(Image.open(fpath).convert("RGB"), dtype=np.float32) / 255.0)
            labels.append(i)
    return np.array(images), np.array(labels)


print("Loading data...")
X_train, y_train = load_split("train")
X_val, y_val = load_split("val")
print(f"Train: {X_train.shape}, Val: {X_val.shape}")

y_train_oh = tf.one_hot(y_train, NUM_CLASSES).numpy()
y_val_oh = tf.one_hot(y_val, NUM_CLASSES).numpy()


def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    return image, label


train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_oh)).shuffle(len(X_train)).map(augment).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val_oh)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


class Sampling(keras.layers.Layer):
    def call(self, inputs):
        mu, log_var = inputs
        return mu + tf.exp(0.5 * log_var) * tf.random.normal(tf.shape(mu))


def build_encoder():
    img_in = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    label_in = keras.Input(shape=(NUM_CLASSES,))

    label_map = keras.layers.Dense(IMG_SIZE * IMG_SIZE, activation="relu")(label_in)
    label_map = keras.layers.Reshape((IMG_SIZE, IMG_SIZE, 1))(label_map)
    x = keras.layers.Concatenate()([img_in, label_map])

    for filters in [32, 64, 128, 256]:
        x = keras.layers.Conv2D(filters, 4, strides=2, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Concatenate()([x, label_in])
    x = keras.layers.Dense(512, activation="relu")(x)

    mu = keras.layers.Dense(LATENT_DIM, name="mu")(x)
    log_var = keras.layers.Dense(LATENT_DIM, name="log_var")(x)
    z = Sampling()([mu, log_var])

    return keras.Model([img_in, label_in], [mu, log_var, z], name="encoder")


def build_decoder():
    z_in = keras.Input(shape=(LATENT_DIM,))
    label_in = keras.Input(shape=(NUM_CLASSES,))

    x = keras.layers.Concatenate()([z_in, label_in])
    x = keras.layers.Dense(512, activation="relu")(x)
    x = keras.layers.Dense(7 * 7 * 256, activation="relu")(x)
    x = keras.layers.Reshape((7, 7, 256))(x)

    for filters, sz in [(128, 14), (64, 28), (32, 56)]:
        x = keras.layers.Conv2DTranspose(filters, 4, strides=2, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        lm = keras.layers.Dense(sz * sz, activation="relu")(label_in)
        lm = keras.layers.Reshape((sz, sz, 1))(lm)
        x = keras.layers.Concatenate()([x, lm])

    x = keras.layers.Conv2DTranspose(3, 4, strides=2, padding="same", activation="sigmoid")(x)

    return keras.Model([z_in, label_in], x, name="decoder")


encoder = build_encoder()
decoder = build_decoder()
encoder.summary()
decoder.summary()

steps_per_epoch = len(X_train) // BATCH_SIZE + 1
lr_schedule = keras.optimizers.schedules.CosineDecay(initial_learning_rate=LR, decay_steps=steps_per_epoch * EPOCHS, alpha=LR_MIN / LR)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)


def compute_loss(images, labels, training):
    mu, log_var, z = encoder([images, labels], training=training)
    recon = decoder([z, labels], training=training)
    recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(images - recon), axis=[1, 2, 3]))
    kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1.0 + log_var - tf.square(mu) - tf.exp(log_var), axis=1))
    mse = tf.reduce_mean(tf.square(images - recon))
    return recon_loss, kl_loss, mse


@tf.function
def train_step(images, labels, beta):
    with tf.GradientTape() as tape:
        recon_loss, kl_loss, mse = compute_loss(images, labels, training=True)
        total = recon_loss + beta * kl_loss
    grads = tape.gradient(total, encoder.trainable_variables + decoder.trainable_variables)
    optimizer.apply_gradients(zip(grads, encoder.trainable_variables + decoder.trainable_variables))
    return mse, kl_loss


history = {"train_mse": [], "train_kl": [], "val_mse": [], "val_kl": []}
best_val_mse = float("inf")

for epoch in range(EPOCHS):
    beta = min(BETA_MAX, BETA_MAX * (epoch + 1) / BETA_WARMUP)
    beta_tf = tf.constant(beta, dtype=tf.float32)

    t_mse, t_kl, n = 0.0, 0.0, 0
    for imgs, labs in train_ds:
        mse, kl = train_step(imgs, labs, beta_tf)
        b = len(imgs)
        t_mse += mse.numpy() * b
        t_kl += kl.numpy() * b
        n += b

    v_mse, v_kl, nv = 0.0, 0.0, 0
    for imgs, labs in val_ds:
        _, kl, mse = compute_loss(imgs, labs, training=False)
        b = len(imgs)
        v_mse += mse.numpy() * b
        v_kl += kl.numpy() * b
        nv += b

    history["train_mse"].append(t_mse / n)
    history["train_kl"].append(t_kl / n)
    history["val_mse"].append(v_mse / nv)
    history["val_kl"].append(v_kl / nv)

    cur_val = v_mse / nv
    improved = cur_val < best_val_mse
    if improved:
        best_val_mse = cur_val
        encoder.save_weights(os.path.join(OUTPUT_DIR, "encoder.weights.h5"))
        decoder.save_weights(os.path.join(OUTPUT_DIR, "decoder.weights.h5"))

    print(f"Epoch {epoch+1:3d}/{EPOCHS} β={beta:.3f} | " f"Train MSE={history['train_mse'][-1]:.6f} KL={history['train_kl'][-1]:.2f} | " f"Val MSE={history['val_mse'][-1]:.6f} KL={history['val_kl'][-1]:.2f}" f"{'  *' if improved else ''}")

encoder.load_weights(os.path.join(OUTPUT_DIR, "encoder.weights.h5"))
decoder.load_weights(os.path.join(OUTPUT_DIR, "decoder.weights.h5"))
print(f"Loaded best model (val MSE={best_val_mse:.6f})")

# Learning curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history["train_mse"], label="Train")
ax1.plot(history["val_mse"], label="Validation")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("MSE")
ax1.set_title("Reconstruction Loss (MSE)")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(range(1, len(history["train_kl"])), history["train_kl"][1:], label="Train")
ax2.plot(range(1, len(history["val_kl"])), history["val_kl"][1:], label="Validation")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("KL Divergence")
ax2.set_title("Regularisation Loss (KL)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "learning_curves.png"), dpi=150)
plt.close()
print("Saved learning_curves.png")

# Reconstructions
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
for j in range(4):
    axes[0, j].set_title("Original", fontsize=9)
    axes[0, j + 4].set_title("Reconstructed", fontsize=9)

for i, cls in enumerate(CLASSES):
    mask = y_val == i
    imgs = X_val[mask][:4]
    imgs_t = tf.constant(imgs)
    labs_t = tf.constant(y_val_oh[mask][:4])
    _, _, z = encoder([imgs_t, labs_t], training=False)
    recon = decoder([z, labs_t], training=False).numpy()
    for j in range(4):
        axes[i, j].imshow(imgs[j])
        axes[i, j].axis("off")
        axes[i, j + 4].imshow(np.clip(recon[j], 0, 1))
        axes[i, j + 4].axis("off")
    axes[i, 0].set_ylabel(cls, fontsize=11, rotation=0, labelpad=50)

plt.suptitle("Reconstructed Images (4 per class)", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "reconstructions.png"), dpi=150)
plt.close()
print("Saved reconstructions.png")

# Task 2.1
happy_img = X_val[y_val == 0][:1]
happy_img_t = tf.constant(happy_img)
happy_lab = tf.constant(y_val_oh[y_val == 0][:1])
sad_lab = tf.one_hot([1], NUM_CLASSES)

_, _, z_h = encoder([happy_img_t, happy_lab], training=False)
recon_h = decoder([z_h, happy_lab], training=False).numpy()
recon_s = decoder([z_h, sad_lab], training=False).numpy()

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
titles = ["Original (Happy)", "Recon (Happy cond.)", "Same latent, Sad cond."]
for ax, img, t in zip(axes, [happy_img[0], np.clip(recon_h[0], 0, 1), np.clip(recon_s[0], 0, 1)], titles):
    ax.imshow(img)
    ax.set_title(t)
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "happy_to_sad_direct.png"), dpi=150)
plt.close()
print("Saved happy_to_sad_direct.png")


# Task 2.2
def mean_latent(cls_idx):
    mask = y_train == cls_idx
    mus = []
    for i in range(0, mask.sum(), 256):
        mu, _, _ = encoder([tf.constant(X_train[mask][i : i + 256]), tf.constant(y_train_oh[mask][i : i + 256])], training=False)
        mus.append(mu.numpy())
    return np.concatenate(mus).mean(axis=0)


happy_mean = mean_latent(0)
sad_mean = mean_latent(1)

fig, axes = plt.subplots(1, 10, figsize=(20, 2.5))
for idx, alpha in enumerate(np.linspace(0, 1, 10)):
    z_interp = tf.constant(((1 - alpha) * happy_mean + alpha * sad_mean)[np.newaxis, :], dtype=tf.float32)
    img = decoder([z_interp, sad_lab], training=False).numpy()[0]
    axes[idx].imshow(np.clip(img, 0, 1))
    axes[idx].set_title(f"α={alpha:.2f}", fontsize=9)
    axes[idx].axis("off")

plt.suptitle("Latent Interpolation: Happy mean → Sad mean (Sad conditioning)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "interpolation.png"), dpi=150)
plt.close()
print("Saved interpolation.png")

print(f"\nAll outputs saved to {OUTPUT_DIR}")
