# Deep Learning 1

Assignment 1 for the **Deep Learning course (5DV236)**.

This project focuses on **facial emotion recognition** with a convolutional neural network trained on a subset of the AffectNet dataset. The implementation is contained in a single training script and a course environment file.

## What This Project Covers

- Four-class emotion classification: `Happy`, `Sad`, `Surprised`, and `Mad`
- A custom CNN built with TensorFlow / Keras
- Image augmentation, batch normalization, dropout, and L2 regularization
- Training, validation, and test evaluation
- Per-class accuracy reporting and learning-curve plotting
- Saved Keras model checkpoints for later reuse

## Files

- `Lab_1--Skeleton_Template.py` - Main training and evaluation script for the assignment
- `environment_5dv236vt26.yml` - Conda environment used for the course setup
- `affectnet_emotion_cnn.keras` - Saved trained model

## Model Overview

The CNN uses repeated convolution blocks with:

- `Conv2D` layers
- Batch normalization
- ReLU activations
- Max pooling
- Dropout regularization

The network ends with global average pooling and dense classification layers for the four emotion classes.

## Tech Stack

- Python 3
- TensorFlow / Keras
- NumPy
- Pillow
- matplotlib

## How to Run

1. Create or activate an environment from `environment_5dv236vt26.yml`.
2. Update the dataset path in `Lab_1--Skeleton_Template.py` if your AffectNet files are stored elsewhere.
3. Run the script:
   - `python Lab_1--Skeleton_Template.py`

## Notes

- The script expects access to the course AffectNet directory and is much faster with a GPU.
- The saved `.keras` model in this folder is an output artifact from training.
- This is course assignment work intended for academic use.