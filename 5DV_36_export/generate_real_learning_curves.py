#!/usr/bin/env python3
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOG_TEXT = r"""
Epoch 1: val_accuracy improved from -inf to 0.28500, saving model to affectnet_emotion_cnn_best.keras
500/500 - 43s - 86ms/step - accuracy: 0.2702 - loss: 1.8404 - val_accuracy: 0.2850 - val_loss: 1.4977 - learning_rate: 5.0000e-04
Epoch 2: val_accuracy improved from 0.28500 to 0.31100, saving model to affectnet_emotion_cnn_best.keras
500/500 - 36s - 72ms/step - accuracy: 0.2865 - loss: 1.7867 - val_accuracy: 0.3110 - val_loss: 1.4125 - learning_rate: 5.0000e-04
Epoch 3: val_accuracy improved from 0.31100 to 0.32400, saving model to affectnet_emotion_cnn_best.keras
500/500 - 36s - 72ms/step - accuracy: 0.3069 - loss: 1.7478 - val_accuracy: 0.3240 - val_loss: 1.4895 - learning_rate: 5.0000e-04
Epoch 4: val_accuracy did not improve from 0.32400
500/500 - 36s - 72ms/step - accuracy: 0.3904 - loss: 1.6471 - val_accuracy: 0.2690 - val_loss: 1.7048 - learning_rate: 5.0000e-04
Epoch 5: val_accuracy improved from 0.32400 to 0.35900, saving model to affectnet_emotion_cnn_best.keras
500/500 - 36s - 72ms/step - accuracy: 0.4821 - loss: 1.5060 - val_accuracy: 0.3590 - val_loss: 1.4798 - learning_rate: 5.0000e-04
Epoch 6: val_accuracy improved from 0.35900 to 0.54400, saving model to affectnet_emotion_cnn_best.keras
500/500 - 36s - 72ms/step - accuracy: 0.5270 - loss: 1.4287 - val_accuracy: 0.5440 - val_loss: 1.0860 - learning_rate: 5.0000e-04
Epoch 7: val_accuracy did not improve from 0.54400
500/500 - 36s - 72ms/step - accuracy: 0.5429 - loss: 1.3875 - val_accuracy: 0.5120 - val_loss: 1.2135 - learning_rate: 5.0000e-04
Epoch 8: val_accuracy did not improve from 0.54400
500/500 - 36s - 72ms/step - accuracy: 0.5674 - loss: 1.3380 - val_accuracy: 0.5180 - val_loss: 1.1837 - learning_rate: 5.0000e-04
Epoch 9: val_accuracy improved from 0.54400 to 0.63600, saving model to affectnet_emotion_cnn_best.keras
500/500 - 36s - 73ms/step - accuracy: 0.5881 - loss: 1.2964 - val_accuracy: 0.6360 - val_loss: 0.9589 - learning_rate: 5.0000e-04
Epoch 10: val_accuracy did not improve from 0.63600
500/500 - 36s - 72ms/step - accuracy: 0.6061 - loss: 1.2610 - val_accuracy: 0.4660 - val_loss: 1.5377 - learning_rate: 5.0000e-04
Epoch 11: val_accuracy did not improve from 0.63600
500/500 - 36s - 72ms/step - accuracy: 0.6204 - loss: 1.2459 - val_accuracy: 0.5880 - val_loss: 0.9938 - learning_rate: 5.0000e-04
Epoch 12: val_accuracy improved from 0.63600 to 0.67800, saving model to affectnet_emotion_cnn_best.keras
500/500 - 36s - 72ms/step - accuracy: 0.6364 - loss: 1.2057 - val_accuracy: 0.6780 - val_loss: 0.9053 - learning_rate: 5.0000e-04
Epoch 13: val_accuracy did not improve from 0.67800
500/500 - 36s - 72ms/step - accuracy: 0.6435 - loss: 1.1829 - val_accuracy: 0.5840 - val_loss: 1.1185 - learning_rate: 5.0000e-04
Epoch 14: val_accuracy did not improve from 0.67800
500/500 - 36s - 72ms/step - accuracy: 0.6510 - loss: 1.1689 - val_accuracy: 0.4450 - val_loss: 1.3903 - learning_rate: 5.0000e-04
Epoch 15: val_accuracy improved from 0.67800 to 0.68300, saving model to affectnet_emotion_cnn_best.keras
500/500 - 36s - 72ms/step - accuracy: 0.6586 - loss: 1.1437 - val_accuracy: 0.6830 - val_loss: 0.8747 - learning_rate: 5.0000e-04
Epoch 16: val_accuracy did not improve from 0.68300
500/500 - 36s - 72ms/step - accuracy: 0.6744 - loss: 1.1175 - val_accuracy: 0.5880 - val_loss: 1.0399 - learning_rate: 5.0000e-04
Epoch 17: val_accuracy did not improve from 0.68300
500/500 - 36s - 72ms/step - accuracy: 0.6800 - loss: 1.0977 - val_accuracy: 0.5860 - val_loss: 1.1374 - learning_rate: 5.0000e-04
Epoch 18: val_accuracy did not improve from 0.68300
500/500 - 36s - 72ms/step - accuracy: 0.6875 - loss: 1.0821 - val_accuracy: 0.6740 - val_loss: 0.9201 - learning_rate: 5.0000e-04
Epoch 19: val_accuracy did not improve from 0.68300
500/500 - 36s - 72ms/step - accuracy: 0.6940 - loss: 1.0656 - val_accuracy: 0.6570 - val_loss: 0.9522 - learning_rate: 5.0000e-04
Epoch 20: val_accuracy improved from 0.68300 to 0.74700, saving model to affectnet_emotion_cnn_best.keras
500/500 - 36s - 72ms/step - accuracy: 0.7109 - loss: 1.0190 - val_accuracy: 0.7470 - val_loss: 0.7925 - learning_rate: 2.5000e-04
Epoch 21: val_accuracy improved from 0.74700 to 0.76400, saving model to affectnet_emotion_cnn_best.keras
500/500 - 36s - 72ms/step - accuracy: 0.7205 - loss: 0.9829 - val_accuracy: 0.7640 - val_loss: 0.7059 - learning_rate: 2.5000e-04
Epoch 22: val_accuracy did not improve from 0.76400
500/500 - 36s - 72ms/step - accuracy: 0.7194 - loss: 0.9887 - val_accuracy: 0.7400 - val_loss: 0.7702 - learning_rate: 2.5000e-04
Epoch 23: val_accuracy did not improve from 0.76400
500/500 - 36s - 72ms/step - accuracy: 0.7260 - loss: 0.9774 - val_accuracy: 0.6420 - val_loss: 0.9289 - learning_rate: 2.5000e-04
Epoch 24: val_accuracy did not improve from 0.76400
500/500 - 36s - 72ms/step - accuracy: 0.7327 - loss: 0.9558 - val_accuracy: 0.7120 - val_loss: 0.8239 - learning_rate: 2.5000e-04
Epoch 25: val_accuracy did not improve from 0.76400
500/500 - 36s - 72ms/step - accuracy: 0.7361 - loss: 0.9376 - val_accuracy: 0.6810 - val_loss: 0.9542 - learning_rate: 2.5000e-04
Epoch 26: val_accuracy did not improve from 0.76400
500/500 - 36s - 72ms/step - accuracy: 0.7466 - loss: 0.9155 - val_accuracy: 0.7100 - val_loss: 0.8120 - learning_rate: 1.2500e-04
Epoch 27: val_accuracy improved from 0.76400 to 0.76800, saving model to affectnet_emotion_cnn_best.keras
500/500 - 36s - 72ms/step - accuracy: 0.7610 - loss: 0.8765 - val_accuracy: 0.7680 - val_loss: 0.6945 - learning_rate: 1.2500e-04
Epoch 28: val_accuracy did not improve from 0.76800
500/500 - 36s - 72ms/step - accuracy: 0.7624 - loss: 0.8768 - val_accuracy: 0.7590 - val_loss: 0.7230 - learning_rate: 1.2500e-04
Epoch 29: val_accuracy did not improve from 0.76800
500/500 - 36s - 72ms/step - accuracy: 0.7634 - loss: 0.8850 - val_accuracy: 0.7680 - val_loss: 0.7089 - learning_rate: 1.2500e-04
Epoch 30: val_accuracy improved from 0.76800 to 0.77700, saving model to affectnet_emotion_cnn_best.keras
500/500 - 36s - 73ms/step - accuracy: 0.7678 - loss: 0.8694 - val_accuracy: 0.7770 - val_loss: 0.6738 - learning_rate: 1.2500e-04
"""


def main():
    pattern = re.compile(
        r"accuracy: ([0-9.]+) - loss: ([0-9.]+) - val_accuracy: ([0-9.]+) - val_loss: ([0-9.]+)"
    )
    matches = pattern.findall(LOG_TEXT)

    train_acc = [float(match[0]) for match in matches]
    train_loss = [float(match[1]) for match in matches]
    val_acc = [float(match[2]) for match in matches]
    val_loss = [float(match[3]) for match in matches]
    epochs = list(range(1, len(matches) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, train_loss, label="Training")
    axes[0].plot(epochs, val_loss, label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="Training")
    axes[1].plot(epochs, val_acc, label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curves")
    axes[1].legend()

    fig.tight_layout()
    output_path = Path(__file__).resolve().parent / "learning_curves.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(output_path)


if __name__ == "__main__":
    main()