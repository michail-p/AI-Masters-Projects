# Machine Learning 2

Assignment 2 for the **Machine Learning course (5DV238)**.

This notebook-based project contains the second course assignment and related machine learning experiments.

## Datasets Used

- **California Housing** (`fetch_california_housing`) for regression experiments
- **Wine** (`load_wine`) for PCA analysis and clustering
- **MNIST-784** (`fetch_openml('mnist_784')`) for PCA dimensionality reduction and reconstruction

## Methods and Models

### Supervised Learning (Neural Networks)

- Dense **Sequential** regression network (Keras/TensorFlow)
- Tuned dense network with **L2 regularization** and **Dropout**
- Optimizer/layer-depth experiments (including Adam/SGD/RMSprop comparisons in assignment tasks)
- **LSTM** model variant for comparison
- **Conv1D** model variant for comparison
- Loss/metrics used: `mean_squared_error`, `mse`, `mae`, `mape`

### Unsupervised Learning

- **PCA** on Wine (2D projection and explained variance analysis)
- **PCA** on MNIST (component selection and reconstruction)
- Custom implementation of **K-Means** on Wine features
- `sklearn.cluster.KMeans` baseline comparison
- Cluster evaluation with confusion matrix, ARI, and NMI

## Files

- `Michail_Pettas.ipynb` – Main notebook with all tasks, code, and answers

## Tech Stack

- Python 3
- Jupyter Notebook
- pandas, numpy
- scikit-learn
- TensorFlow / Keras
- matplotlib, seaborn

## How to Run

1. Open `Michail_Pettas.ipynb` in Jupyter Notebook or VS Code.
2. Install missing dependencies if needed:
   - `pip install pandas numpy scikit-learn matplotlib seaborn`
3. Run cells from top to bottom.

## Notes

This is course assignment work and intended for academic use.
