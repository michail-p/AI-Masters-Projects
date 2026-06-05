#!/usr/bin/env python3
"""Generate the PDF report for Assignment 3."""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

# Configuration
DATA_DIR = '/import/course/5dv236/vt26/AffectNet'
OUTPUT_DIR = os.path.expanduser('~/workspace/lab3_output')
CLASSES = ['Happy', 'Sad', 'Surprised', 'Mad']
NUM_CLASSES = len(CLASSES)
IMG_SIZE = 112
LATENT_DIM = 128


# Model definition (must match training)
class Sampling(keras.layers.Layer):
    def call(self, inputs):
        mu, log_var = inputs
        return mu + tf.exp(0.5 * log_var) * tf.random.normal(tf.shape(mu))


def build_encoder():
    img_in = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    label_in = keras.Input(shape=(NUM_CLASSES,))
    label_map = keras.layers.Dense(IMG_SIZE * IMG_SIZE, activation='relu')(label_in)
    label_map = keras.layers.Reshape((IMG_SIZE, IMG_SIZE, 1))(label_map)
    x = keras.layers.Concatenate()([img_in, label_map])
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
    for filters, sz in [(128, 14), (64, 28), (32, 56)]:
        x = keras.layers.Conv2DTranspose(filters, 4, strides=2, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        lm = keras.layers.Dense(sz * sz, activation='relu')(label_in)
        lm = keras.layers.Reshape((sz, sz, 1))(lm)
        x = keras.layers.Concatenate()([x, lm])
    x = keras.layers.Conv2DTranspose(3, 4, strides=2, padding='same', activation='sigmoid')(x)
    return keras.Model([z_in, label_in], x, name='decoder')


# Load data and model
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


print("Loading data...")
X_train, y_train = load_split('train')
X_val, y_val = load_split('val')
y_train_oh = tf.one_hot(y_train, NUM_CLASSES).numpy()
y_val_oh = tf.one_hot(y_val, NUM_CLASSES).numpy()

encoder = build_encoder()
decoder = build_decoder()
encoder.load_weights(os.path.join(OUTPUT_DIR, 'encoder.weights.h5'))
decoder.load_weights(os.path.join(OUTPUT_DIR, 'decoder.weights.h5'))
print("Model loaded.")


# Generate report PDF
pdf_path = os.path.join(OUTPUT_DIR, 'report.pdf')

with PdfPages(pdf_path) as pdf:

    # Page 1: Title + Solution Description
    fig = plt.figure(figsize=(8.27, 11.69))  # A4
    fig.text(0.5, 0.92, 'Assignment 3: Report', ha='center', fontsize=20, fontweight='bold')
    fig.text(0.5, 0.88, 'Deep Generative Models: VAEs for Conditional\nFace Expression Generation',
             ha='center', fontsize=14)

    fig.text(0.12, 0.82, 'Student Name(s):  Michail Pettas', fontsize=11)
    fig.text(0.12, 0.79, 'CS Username(s):   mai25mps', fontsize=11)

    fig.text(0.5, 0.74, 'Solution Description', ha='center', fontsize=16, fontweight='bold')

    desc = (
        "We implement a conditional variational autoencoder (CVAE) using TensorFlow/Keras to "
        "generate face expression images conditioned on one of four classes: Happy, Sad, Surprised, "
        "and Mad.\n\n"
        "Architecture. The encoder receives a 112x112x3 image concatenated with a spatially "
        "broadcast learned label map (Dense projection of the one-hot label to 112x112x1), then "
        "passes through four convolutional layers (32, 64, 128, 256 filters, stride 2, 4x4 "
        "kernels) with batch normalisation and ReLU, reducing to 7x7. The flattened features are "
        "concatenated with the one-hot label again and mapped via a dense layer (512 units) to "
        "mean and log-variance vectors in a 128-dimensional latent space. Sampling uses the "
        "reparameterisation trick: z = mu + exp(0.5 * log_var) * epsilon, epsilon ~ N(0, I). "
        "The decoder concatenates z with the one-hot label, maps through dense layers to a "
        "7x7x256 feature map, then applies three transposed-convolution layers (128, 64, 32 "
        "filters) with a label injection at every spatial level: a Dense projection of the label "
        "to a sz x sz x 1 map is concatenated with the feature maps after each upsampling. A "
        "final transposed-convolution layer outputs the 112x112x3 image with sigmoid activation.\n\n"
        "Loss function. The total loss combines MSE reconstruction (summed over pixels per "
        "sample) with an analytical KL divergence: KL = -0.5 * sum(1 + log_var - mu^2 - "
        "exp(log_var)). We use beta = 0.15 with linear KL annealing over 20 epochs to prevent "
        "posterior collapse.\n\n"
        "Training. Adam with cosine learning rate decay (1e-3 to 1e-5) over 120 epochs, batch "
        "size 64 on GPU, with random horizontal flip augmentation. The best model (by validation "
        "MSE) is checkpointed. The reduced beta allows the latent space to encode richer "
        "class-specific information, improving interpolation quality.\n\n"
        "Challenges. Balancing reconstruction vs. latent regularity was key: beta=1.0 caused "
        "posterior collapse and washed-out interpolations. Multi-level label injection in the "
        "decoder significantly improved class-conditional generation. KL annealing with reduced "
        "beta resolved these issues."
    )

    ax = fig.add_axes([0.08, 0.05, 0.84, 0.65])
    ax.axis('off')
    ax.text(0, 1, desc, fontsize=9, verticalalignment='top', wrap=True,
            fontfamily='serif', transform=ax.transAxes)

    pdf.savefig(fig)
    plt.close()

    # Page 2: Reconstructed Images
    fig, axes = plt.subplots(4, 8, figsize=(8.27, 7))
    fig.suptitle('Sample Images: Original (left 4) vs Reconstructed (right 4)', fontsize=13,
                 fontweight='bold', y=0.98)
    for j in range(4):
        axes[0, j].set_title('Orig.', fontsize=7)
        axes[0, j + 4].set_title('Recon.', fontsize=7)

    for i, cls in enumerate(CLASSES):
        mask = y_val == i
        imgs = X_val[mask][:4]
        labs_t = tf.constant(y_val_oh[mask][:4])
        imgs_t = tf.constant(imgs)
        _, _, z = encoder([imgs_t, labs_t], training=False)
        recon = decoder([z, labs_t], training=False).numpy()
        for j in range(4):
            axes[i, j].imshow(imgs[j])
            axes[i, j].axis('off')
            axes[i, j + 4].imshow(np.clip(recon[j], 0, 1))
            axes[i, j + 4].axis('off')
        axes[i, 0].set_ylabel(cls, fontsize=10, rotation=0, labelpad=45)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig)
    plt.close()
    print("Page 2: Reconstructions done")

    # Page 3: Learning Curves
    # Re-read training history from log
    train_mse, train_kl, val_mse, val_kl = [], [], [], []
    log_path = os.path.expanduser('~/workspace/cvae_log7.txt')
    with open(log_path) as f:
        for line in f:
            if line.strip().startswith('Epoch'):
                # Format: Epoch N/80 β=X | Train MSE=X KL=X | Val MSE=X KL=X
                parts = line.strip().split('|')
                t_parts = parts[1].strip().split()  # ["Train", "MSE=X", "KL=X"]
                v_parts = parts[2].strip().split()   # ["Val", "MSE=X", "KL=X"]
                train_mse.append(float(t_parts[1].split('=')[1]))
                train_kl.append(float(t_parts[2].split('=')[1]))
                val_mse.append(float(v_parts[1].split('=')[1]))
                val_kl.append(float(v_parts[2].split('=')[1]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.27, 4))
    fig.suptitle('Learning Curves', fontsize=13, fontweight='bold')

    ax1.plot(train_mse, label='Train')
    ax1.plot(val_mse, label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE')
    ax1.set_title('Reconstruction Loss (MSE)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Skip first epoch for KL (has huge spike from annealing start)
    ax2.plot(range(1, len(train_kl)), train_kl[1:], label='Train')
    ax2.plot(range(1, len(val_kl)), val_kl[1:], label='Validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('KL Divergence')
    ax2.set_title('Regularisation Loss (KL)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()
    print("Page 3: Learning curves done")

    # Page 4: Task 2 - Interpolation
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.5, 0.95, 'Task 2: Latent Space Interpolation', ha='center',
             fontsize=16, fontweight='bold')

    # 2.1: Happy → Sad conditioning
    happy_img = X_val[y_val == 0][:1]
    happy_img_t = tf.constant(happy_img)
    happy_lab = tf.constant(y_val_oh[y_val == 0][:1])
    sad_lab = tf.one_hot([1], NUM_CLASSES)

    _, _, z_h = encoder([happy_img_t, happy_lab], training=False)
    recon_h = decoder([z_h, happy_lab], training=False).numpy()
    recon_s = decoder([z_h, sad_lab], training=False).numpy()

    ax_row = [fig.add_axes([0.08 + j * 0.30, 0.72, 0.25, 0.20]) for j in range(3)]
    titles_2_1 = ['Original (Happy)', 'Recon (Happy cond.)', 'Same latent, Sad cond.']
    imgs_2_1 = [happy_img[0], np.clip(recon_h[0], 0, 1), np.clip(recon_s[0], 0, 1)]
    for ax, img, t in zip(ax_row, imgs_2_1, titles_2_1):
        ax.imshow(img)
        ax.set_title(t, fontsize=8)
        ax.axis('off')

    # Explanation text
    interp_desc_1 = (
        "Task 2.1: A Happy image is encoded and its latent vector is decoded with the Sad "
        "class-conditioning signal. The generated image does not look like a sad version of the "
        "original person. The latent vector was encoded under Happy conditioning, so it carries "
        "features that belong to the Happy class. Decoding it with Sad conditioning makes the "
        "decoder treat that latent as if it came from Sad data, so the output looks like a Sad "
        "face but loses the original identity. Both the conditioning signal and the latent code "
        "affect the result, so swapping only the condition is not enough to keep the identity "
        "across classes."
    )
    ax_text1 = fig.add_axes([0.08, 0.58, 0.84, 0.12])
    ax_text1.axis('off')
    ax_text1.text(0, 1, interp_desc_1, fontsize=9, verticalalignment='top', wrap=True,
                  fontfamily='serif', transform=ax_text1.transAxes)

    # 2.2: Mean-latent interpolation
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

    interp_axes = [fig.add_axes([0.02 + j * 0.098, 0.35, 0.088, 0.18]) for j in range(10)]
    for idx, alpha in enumerate(np.linspace(0, 1, 10)):
        z_interp = tf.constant(
            ((1 - alpha) * happy_mean + alpha * sad_mean)[np.newaxis, :], dtype=tf.float32)
        img = decoder([z_interp, sad_lab], training=False).numpy()[0]
        interp_axes[idx].imshow(np.clip(img, 0, 1))
        interp_axes[idx].set_title(f'α={alpha:.1f}', fontsize=7)
        interp_axes[idx].axis('off')

    fig.text(0.5, 0.55, 'Latent Interpolation: Happy mean → Sad mean (Sad conditioning)',
             ha='center', fontsize=11, fontweight='bold')

    interp_desc_2 = (
        "Task 2.2: We compute the mean latent vector (mu) for all Happy training images and all "
        "Sad training images. We then linearly interpolate between these two mean vectors in 10 "
        "equally spaced steps (alpha = 0.0 to 1.0), decoding each interpolated vector with the "
        "Sad conditioning signal.\n\n"
        "The interpolations give plausible transitions. At alpha=0 (Happy mean) the generated "
        "face shows a slight smile, and at alpha=1 (Sad mean) the expression becomes more "
        "neutral or downturned. The change is gradual and there are no sudden jumps between "
        "steps. The faces stay blurry throughout, which is typical for a VAE trained with MSE "
        "loss since it tends to average over possible outputs instead of producing sharp images. "
        "The faces shift expression gradually rather than jumping between distinct classes, which "
        "suggests the CVAE has learned a fairly continuous latent space where the expression can "
        "be moved along a direction."
    )
    ax_text2 = fig.add_axes([0.08, 0.05, 0.84, 0.28])
    ax_text2.axis('off')
    ax_text2.text(0, 1, interp_desc_2, fontsize=9, verticalalignment='top', wrap=True,
                  fontfamily='serif', transform=ax_text2.transAxes)

    pdf.savefig(fig)
    plt.close()
    print("Page 4: Interpolation done")

print(f"\nReport saved to {pdf_path}")
