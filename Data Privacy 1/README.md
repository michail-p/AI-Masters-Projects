# Data Privacy 1

Assignment 1 for the **Data Privacy course (5DV241)** at Umeå University.

This project investigates empirical data masking techniques and evaluates the fundamental tradeoff between privacy (disclosure risk) and data utility on the UCI Adult Census income dataset.

## What This Project Covers

- **Dataset Preprocessing:** Cleaning and structuring the Adult dataset (~32,561 records) across numerical and categorical features.
- **Data Protection / Masking Methods:**
  - **Generalization:** Interval rounding on quasi-identifiers.
  - **Rank Swapping:** Swapping attribute values within rank proximity windows to preserve marginal distributions while disrupting record linkage.
  - **Noise Addition:** Adding zero-mean Gaussian/uniform perturbations scaled to attribute variance.
- **Privacy Evaluation (Disclosure Risk):** Distance-based record linkage calculating re-identification rates when matching masked records against the original dataset.
- **Utility / Information Loss Evaluation:**
  - Generic information loss metrics (preservation of mean, variance, covariance, and correlations).
  - Specific downstream task utility: Random Forest classification predicting whether an individual earns `> $50K/year`.
- **Privacy–Utility Tradeoff Analysis:** Comparative visualization across 4 parameter settings per masking technique.

## Files

- `assignment1.ipynb` - Jupyter notebook containing all implementations, experiments, and visualizations
- `report.pdf` - Written assignment report with methodology, results, and discussion
- `results_comparison.png` - Tradeoff plot comparing disclosure risk vs classification utility
- `adult.data`, `adult.names` - UCI Adult Census dataset and its description

## Tech Stack

- Python 3
- Jupyter Notebook
- pandas, NumPy
- scikit-learn
- matplotlib, seaborn

## How to Run

1. Open `assignment1.ipynb` in VS Code or Jupyter Notebook.
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```
3. Run all cells sequentially.

## Notes

This is course assignment work intended for academic use.
