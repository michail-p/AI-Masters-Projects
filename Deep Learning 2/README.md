# Deep Learning 2

Assignment 2 for the **Deep Learning course (5DV236)**.

This project implements the **backpropagation algorithm manually** for small models built with the course `jaxon` library on top of JAX. The assignment demonstrates how layer-wise Jacobians can be combined into full loss gradients and then used for gradient descent.

## What This Project Covers

- Manual backpropagation through a model's hypothesis layers
- Jacobian-vector style gradient composition for inputs and parameters
- Full-batch gradient descent updates
- A linear regression example with mean squared error
- A feed-forward neural network example for multiclass classification
- Learning-curve generation for both training runs

## Files

- `Lab_2--Skeleton_Template.py` - Main assignment solution with backpropagation and training code
- `Lab_2--The_Backpropagation_Algorithm.txt` - Assignment description and report instructions
- `environment_5dv236vt26.yml` - Course environment definition
- `Lab_2_Report.pdf` - Submitted report with solution summary and plots

## Implementation Overview

The script defines helper functions to:

- Compose upstream gradients with layer input Jacobians
- Compose upstream gradients with parameter Jacobians
- Traverse model layers in reverse order to build the gradient dictionary
- Apply gradient descent updates to all trainable parameters
- Train both example models and save their loss curves

## Tech Stack

- Python 3
- JAX
- JAX NumPy
- matplotlib
- `jaxon` course library

## How to Run

1. Create or activate the environment from `environment_5dv236vt26.yml`.
2. Ensure the required course files are available, including `jaxon`, `regression_data.npz`, and `classification_data.npz`.
3. Run the script:
   - `python Lab_2--Skeleton_Template.py`

## Notes

- This assignment is centered on implementing backpropagation manually rather than using autodiff helpers such as `jax.grad`.
- Some required inputs come from the course distribution and may not be committed in this repository.
- This is course assignment work intended for academic use.