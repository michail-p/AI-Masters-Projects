#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skeleton file for Assignment 2 in the course Deep Learning (5DV236).

Your task is to fill in the missing parts.

Created on Tue Feb  6 10:07:12 2024

Copyright (c) 2024-2026, Tommy Löfstedt. All rights reserved.

@author:  Tommy Löfstedt
@email:   tommy.lofstedt@umu.se
@license: BSD 3-clause.
"""
import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

import jaxon

SEED = 42
EPOCHS = 200

PLOT_SETTINGS = {"text.usetex": False,
                 "font.family": "serif",
                 "figure.figsize": (10.0, 5.0),
                 "font.size": 16,
                 "axes.labelsize": 16,
                 "legend.fontsize": 14,
                 "xtick.labelsize": 14,
                 "ytick.labelsize": 14,
                 "axes.titlesize": 24,
                 "lines.linewidth": 2.0,
                 "axes.formatter.limits": [-5, 5],
                 }
plt.rcParams.update(PLOT_SETTINGS)


def compose_input_gradient(upstream_grad, input_jacobian):
    """Compose dL/dout with dout/din to obtain dL/din."""
    output_axes = tuple(range(upstream_grad.ndim))
    result = jnp.tensordot(upstream_grad, input_jacobian, axes=(output_axes, output_axes))
    return result


def compose_param_gradient(upstream_grad, param_jacobian):
    """Compose dL/dout with dout/dparam to obtain dL/dparam."""
    output_axes = tuple(range(upstream_grad.ndim))
    result = jnp.tensordot(upstream_grad, param_jacobian, axes=(output_axes, output_axes))
    return result


def backpropagate_model(model, X, Y):
    """Backpropagate the loss gradient through all layers in the hypothesis."""
    jacobians = model.hypothesis.backward(X)

    grad = {}
    grad["loss"] = model.loss.backward(Y, model.output["hypothesis"])

    upstream_grad = grad["loss"]["inputs"][1]
    layers = model.hypothesis.layers

    for index in range(len(layers) - 1, -1, -1):
        layer_name, _ = layers[index]
        layer_jacobians = jacobians[layer_name]

        layer_grad = {"inputs": []}
        layer_grad["inputs"].append(
            compose_input_gradient(upstream_grad, layer_jacobians["inputs"][0])
        )

        for param_name, param_jacobian in layer_jacobians.items():
            if param_name == "inputs":
                continue
            layer_grad[param_name] = compose_param_gradient(upstream_grad,
                                                            param_jacobian)

        grad[layer_name] = layer_grad
        upstream_grad = layer_grad["inputs"][0]

    return grad


def gradient_descent_step(model, grad, learning_rate):
    """Update all trainable layer parameters using one gradient descent step."""
    for layer_name, _ in model.hypothesis.layers:
        params = model.params[layer_name]
        for param_name in params:
            params[param_name] = params[param_name] - learning_rate * grad[layer_name][param_name]


def train_model(model, X, Y, epochs, learning_rate):
    """Train a model using full-batch gradient descent."""
    training_loss = []
    for _ in range(epochs):
        loss_value = model(X, Y)
        grad = model.backward(X, Y)
        gradient_descent_step(model, grad, learning_rate)
        training_loss.append(float(loss_value))

    return training_loss


# Example 1: A linear regression model.

data = jnp.load("regression_data.npz")
X = data["X"]
y = data["y"]

n, p = X.shape
c = y.shape[1]

key = jaxon.utils.prng_key(SEED)  # Generate PRNG key
key, subkey = jax.random.split(key)  # Use the split template for new rng keys
affine_hypothesis = jaxon.hypotheses.AffineFunction(key=subkey)
loss = jaxon.losses.MeanSquaredError()


class LinearRegressionModel(jaxon.models.Model):
    """A linear regression model."""

    @jaxon.utils.handle_backward(num_inputs=2)
    def backward(self, *inputs):
        """Implement the backwards pass, compute all derivatives."""
        X, y = inputs  # The training data
        return backpropagate_model(self, X, y)


model = LinearRegressionModel(loss, affine_hypothesis)
yhat = affine_hypothesis(X)
loss_value = model(X, y)

# Test the computed derivatives
try:
    jaxon.test.model_backward(model, X, y)
    print(f"Great! The derivatives provided by '{type(model).__name__}' "
          "seem correct!")
except Exception:
    print(f"Oops! The derivatives provided by '{type(model).__name__}' seem "
          "to be wrong!")


# Here you need to implement gradient descent using your computed gradients
learning_rate = 0.01
training_loss = train_model(model, X, y,
                            epochs=EPOCHS,
                            learning_rate=learning_rate)
print(f"Linear regression final loss: {training_loss[-1]:.6f}")

# Plot the training loss
fig = plt.figure()
fig.subplots_adjust(left=0.0575,
                    right=0.985,
                    top=0.93,
                    bottom=0.1075,
                    wspace=0.00,
                    hspace=0.12)
plt.plot(range(1, len(training_loss) + 1), training_loss)
plt.xlabel("Epoch")
plt.ylabel("Loss value")
plt.title("Gradient Descent on a Linear Regression Model")
plt.xlim([1, 200])
plt.savefig("linear_regression_learning_curve.png", dpi=200)


# Example 2: A deep neural network model for classification.

data = jnp.load("classification_data.npz")
X = data["X"]
Y = data["y"]  # Note: Captial Y here!

n, p = X.shape
c = Y.shape[1]

key = jaxon.utils.prng_key(SEED)  # Generate PRNG key

key, subkey = jax.random.split(key)
nn = jaxon.hypotheses.FeedForwardNeuralNetwork(key=subkey)

# Implement your neural network here using nn.add(...)
key, subkey = jax.random.split(key)
nn.add("dense1", jaxon.layers.Dense(16, key=subkey))
nn.add("relu1", jaxon.layers.ReLU())
key, subkey = jax.random.split(key)
nn.add("dense2", jaxon.layers.Dense(c, key=subkey))
nn.add("softmax", jaxon.layers.Softmax())

loss = jaxon.losses.CategoricalCrossentropy()


class NeuralNetworkModel(jaxon.models.Model):
    """A neural network model for multi-class classification."""

    @jaxon.utils.handle_backward(num_inputs=2)
    def backward(self, *inputs):
        """Implement the backwards pass, compute all derivatives."""
        X, Y = inputs  # The training data
        return backpropagate_model(self, X, Y)


model = NeuralNetworkModel(loss, nn)
Yhat = nn(X)
loss_value = model(X, Y)

# Test the computed derivatives
try:
    jaxon.test.model_backward(model, X, Y)
    print(f"Great! The derivatives provided by '{type(model).__name__}' "
          "seem correct!")
except Exception:
    print(f"Oops! The derivatives provided by '{type(model).__name__}' seem "
          "to be wrong!")


# Here you need to implement gradient descent using your partial derivatives
learning_rate = 0.1
training_loss = train_model(model, X, Y,
                            epochs=EPOCHS,
                            learning_rate=learning_rate)
print(f"Neural network final loss: {training_loss[-1]:.6f}")

# Plot the training loss
fig = plt.figure()
fig.subplots_adjust(left=0.0575,
                    right=0.985,
                    top=0.93,
                    bottom=0.1075,
                    wspace=0.00,
                    hspace=0.12)
plt.plot(range(1, len(training_loss) + 1), training_loss)
plt.xlabel("Epoch")
plt.ylabel("Loss value")
plt.title("Gradient Descent on a Neural Network for Multiclass Classification")
plt.xlim([1, 200])
plt.savefig("classification_learning_curve.png", dpi=200)
plt.show()
