# Data Privacy 3

Assignment 3 for the **Data Privacy course (5DV241)** at Umeå University.

This project focuses on **Differential Privacy (DP)** and **Generative Tabular Synthetic Data**, evaluating how formal privacy guarantees and generative models impact downstream machine learning utility across multiple architectures and privacy budgets.

## What This Project Covers

- **Differential Privacy Evaluation:**
  - Applying differential privacy mechanisms across multiple privacy budgets ($\epsilon$).
  - Measuring downstream classification performance (Logistic Regression, Random Forest, Multi-Layer Perceptron).
- **Synthetic Data Generation with CTGAN:**
  - Training Conditional GANs (CTGAN via SDV) on tabular data across varying training dataset sizes ($N \in \{10, 100, 1000, 5000, 16512\}$) and random seeds.
  - Benchmarking generator fit time and scaling properties.
  - Assessing synthetic data fidelity, marginal distributions, and correlation preservation.
- **Combined DP-Synthetic Data Pipelines:**
  - Evaluating compound privacy degradation when combining DP mechanisms with generative synthetic models.
  - Comparison of real-data baselines vs pure DP vs synthetic vs DP-synthetic pipelines.
- **Empirical Reporting:**
  - Statistical aggregation across multiple random seeds with confidence intervals.
  - Detailed visualizations and publication-quality report.

## Files

- `Assignment3_DP_Synthetic.ipynb` - Main notebook containing experimental pipeline, training loops, and evaluations
- `Report_Assignment3.pdf` - Full written assignment report
- `requirements.txt` - Python package requirements
- `outputs/figures/` - Generated analytical plots:
  - `fig0_scaling.png` - CTGAN training time scaling
  - `fig1_dp_utility.png` - Classification utility across DP budget $\epsilon$
  - `fig2_synthetic_utility.png` - Utility of models trained on synthetic data
  - `fig3_comparison.png` - Comparative utility matrix (Real vs DP vs Synthetic)
  - `fig4_dp_synthetic_utility.png` - DP-synthetic pipeline degradation
- `outputs/results/` - CSV benchmark logs and summary tables
- `outputs/synthetic/` - Generated CSV synthetic datasets across configurations

## Tech Stack

- Python 3.11
- PyTorch
- SDV (Synthetic Data Vault) / CTGAN
- scikit-learn
- pandas, NumPy
- matplotlib, seaborn

## How to Run

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\pip install -r requirements.txt
   ```
2. Open and run `Assignment3_DP_Synthetic.ipynb` in VS Code or Jupyter Notebook.

## Notes

This is course assignment work intended for academic use.
