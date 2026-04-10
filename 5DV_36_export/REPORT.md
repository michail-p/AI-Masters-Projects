# Assignment 1 Report

Student Name: Michail Pettas

CS username: mai25mps

## Solution Description

I implemented a convolutional neural network in TensorFlow/Keras for four-class facial expression classification on the AffectNet subset with the classes Happy, Sad, Surprised, and Mad. The final model uses four convolutional blocks with increasing channel depth `24 -> 48 -> 96 -> 128`. Each block contains two `3 x 3` convolution layers followed by batch normalization, ReLU activations, max pooling, and dropout. After the convolutional feature extractor, the model uses global average pooling, a dense layer with 128 hidden units, dropout, and a final softmax output layer for multiclass classification.

I chose a CNN because the task is image classification and convolutional layers are well suited for learning local spatial patterns such as facial features, edges, and expression-specific structures. Batch normalization was used to stabilize optimization, while dropout was used to reduce overfitting. I also used data augmentation during training with horizontal flips, small rotations, zoom, and contrast changes to improve generalization. The final training setup used the Adam optimizer with an initial learning rate of `5e-4`, `ReduceLROnPlateau` to lower the learning rate when validation accuracy stopped improving, and early stopping to restore the best weights. To improve weak classes, I used class weighting with higher weights for Sad and Mad.

The input images were normalized from integer pixel values in `[0, 255]` to floating-point values in `[0, 1]`. The final model was selected based on the highest validation accuracy during training. The best validation accuracy reached `0.777`, which is above the suggested threshold in the assignment.

## Validation and Test Accuracies

- Happy: `0.856` (Validation), `0.884` (Test)
- Sad: `0.736` (Validation), `0.636` (Test)
- Surprised: `0.700` (Validation), `0.716` (Test)
- Mad: `0.816` (Validation), `0.744` (Test)

Overall:

- Validation accuracy: `0.7770`
- Validation loss: `0.6756`
- Test accuracy: `0.7450`
- Test loss: `0.7320`

## Learning Curves and Model Architecture

Use the figure in:

- `learning_curves.png`

For the architecture figure, you can use the model summary from the training output or create a simple diagram based on the following structure:

- Input `(112, 112, 3)`
- Augmentation
- Conv block 1: `24` filters
- Conv block 2: `48` filters
- Conv block 3: `96` filters
- Conv block 4: `128` filters
- Global average pooling
- Dense `128`
- Softmax `4`

## Model Path

Path to the model on the CS computer systems:

- `~/workspace/5DV_36_export/affectnet_emotion_cnn.keras`
