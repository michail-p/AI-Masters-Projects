# Machine Learning 3

Assignment 3 for the **Machine Learning course (5DV238)**.

This project focuses on **human activity recognition (HAR)** using the UCI Human Activity Recognition Using Smartphones dataset. The work compares classical machine learning on engineered features with sequence-based modeling on raw inertial signals.

## What This Project Covers

- Classification of six human activities from smartphone sensor data
- Subject-aware validation using `GroupKFold`
- Comparison of a majority baseline, Logistic Regression, k-Nearest Neighbors, and Decision Tree
- Analysis of feature importance for the best logistic-regression model
- Raw inertial-signal experiments using flattened 1-NN and a compact 1D CNN

## Files

- `Michail_Pettas_Sergio_Costa.ipynb` - Main notebook with the full assignment pipeline, experiments, and analysis
- `assignment_3_report_template_vt26.tex` - LaTeX source for the written report
- `report.pdf` - Final report in PDF format

## Tech Stack

- Python 3
- Jupyter Notebook
- pandas, numpy
- scikit-learn
- TensorFlow / Keras
- matplotlib, seaborn

## How to Run

1. Open `Michail_Pettas_Sergio_Costa.ipynb` in Jupyter Notebook or VS Code.
2. Install missing dependencies if needed:
   - `pip install pandas numpy scikit-learn matplotlib seaborn tensorflow`
3. Run the notebook cells from top to bottom.

## Notes

- The notebook downloads the UCI HAR dataset automatically if it is not already present.
- The classical feature-based pipeline is the main solution, while the raw-signal CNN is included as an extended comparison.
- This is course assignment work intended for academic use.
