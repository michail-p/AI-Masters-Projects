# ML – Fashion Article Classification

**Course:** Machine Learning  
**Author:** Michail Pettas

---

## 📋 Overview

This project implements classification of fashion articles from the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) using two different approaches:

1. **Custom k-Nearest Neighbors (k-NN)** classifier built from scratch
2. **Dense Neural Network** (Multilayer Perceptron)

---

## 🗂️ Project Structure

```
ML - Fashion Article Classification/
├── assignment-classification-5.ipynb   # Main Jupyter notebook
└── report.tex                          # LaTeX report
```

---

## 📊 Dataset

**Fashion-MNIST** contains 70,000 grayscale images (28×28 pixels) of fashion items across 10 categories:

| Label | Description |
|-------|-------------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

### Data Split

- **Training:** 52,500 samples (75%)
- **Validation:** 8,750 samples (12.5%)
- **Test:** 8,750 samples (12.5%)

---

## 🧠 Implemented Models

### 1. Custom k-NN Classifier

A from-scratch implementation featuring:

- **Euclidean distance** calculation
- **Lazy learning** approach (stores all training data)
- **Majority voting** for classification
- **Vectorized operations** for performance optimization

**Hyperparameter Search:**

- Tested k values: [1, 3, 5, 7, 9, 11, 13, 15]
- Selection based on validation accuracy

### 2. Dense Neural Network (MLP)

Multilayer perceptron with:

- Input layer (784 features)
- Hidden layers with ReLU activation
- Output layer with softmax (10 classes)
- Hyperparameter tuning for architecture and learning rate

---

## 🚀 Usage

### Running the Notebook

1. Open `assignment-classification-5.ipynb` in Jupyter/VS Code
2. Select appropriate Python kernel
3. Run cells sequentially

### Setting Up Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install numpy scikit-learn matplotlib jupyter

# Register kernel for Jupyter
pip install ipykernel
python -m ipykernel install --user --name=venv
```

---

## 📈 Data Preprocessing

- **Normalization:** MinMaxScaler to range [-1, 1]
- **Fitted on training data only** to prevent data leakage
- Test data transformed only at final evaluation

---

## 📦 Requirements

- Python 3.10+
- NumPy
- scikit-learn
- Matplotlib
- Jupyter Notebook

```bash
pip install numpy scikit-learn matplotlib jupyter
```

---

## 📊 Evaluation Metrics

- **Accuracy:** Overall classification accuracy
- **Confusion Matrix:** Per-class performance analysis
- **Training/Validation curves:** For neural network training

---

## 📚 References

- [Fashion-MNIST GitHub](https://github.com/zalandoresearch/fashion-mnist)
- Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms
