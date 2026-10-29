# Natural Language Processing 2

Lab 2 for the **Natural Language Processing course (5DV218 Språkteknologi)** at Umeå University.

This project implements an end-to-end text classification benchmark on the **20 Newsgroups** dataset, systematically comparing text preprocessing strategies, feature extraction representations, and supervised learning algorithms.

## What This Project Covers

- **Dataset Benchmark:** Full train/test evaluation on the 20 Newsgroups corpus across 20 distinct discussion categories.
- **Preprocessing Pipeline:** Comparing four text normalization pipelines using NLTK:
  1. Default raw text
  2. Lowercased with punctuation and number removal
  3. Stopword removal (NLTK English stopwords)
  4. Snowball stemming
- **Feature Engineering:**
  - Bag-of-Words representation (`CountVectorizer`)
  - Term Frequency-Inverse Document Frequency (`TfidfVectorizer`) with sublinear term-frequency scaling
- **Supervised Classifiers:**
  - **Multinomial Naive Bayes** (`MultinomialNB`)
  - **Linear Support Vector Classification** (`LinearSVC`)
  - **Logistic Regression** (Multinomial with L2 regularization)
- **Evaluation & Diagnostics:**
  - Overall accuracy, macro-averaged, micro-averaged, and weighted F1-scores
  - Per-category precision and recall breakdowns
  - Confusion matrix generation and heatmap visualization (`confusion_matrix.png`) identifying common misclassification clusters
  - Structured JSON export of all experiment metrics (`output/results.json`)

## Files

- `lab2.py` - Complete benchmarking and training script
- `Lab2_report.pdf` - Written lab report with empirical comparisons and discussions
- `output/confusion_matrix.png` - Normalized confusion matrix plot for the best model
- `output/results.json` - Serialized performance metrics across all model and preprocessing configurations

## Tech Stack

- Python 3
- scikit-learn
- NLTK (punkt, stopwords, SnowballStemmer)
- NumPy
- matplotlib, seaborn

## How to Run

1. Install dependencies:
   ```bash
   pip install scikit-learn nltk numpy matplotlib seaborn
   ```
2. Run the benchmarking script:
   ```bash
   python lab2.py
   ```

## Notes

This is course assignment work intended for academic use.
