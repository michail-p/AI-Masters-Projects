"""
5DV218 Språkteknologi - Lab 2: Text Classification
Author: Michail Pettas

This script implements:
1. Dataset preparation (20 Newsgroups train and test splits).
2. Text preprocessing with NLTK (lowercasing, punctuation/number removal, stop words, stemming).
3. Feature representations (CountVectorizer and TfidfVectorizer across 4 preprocessing approaches).
4. Supervised classification with MultinomialNB, LinearSVC, and LogisticRegression.
5. Model evaluation (accuracy, precision, recall, F1).
6. F1-score analysis with different averaging methods and label subsets.
7. Confusion matrix generation and visualization for the best performing model.
"""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Download required NLTK resources
for resource in ("stopwords", "punkt", "punkt_tab"):
    nltk.download(resource, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
STEMMER = SnowballStemmer("english")


def heading(title):
    print(f"\n{'=' * 75}\n{title}\n{'=' * 75}")


# -----------------------------------------------------------------------------
def run_lab():
    # 1. Dataset preparation
    # -----------------------------------------------------------------------------
    heading("1. Dataset Preparation")

    newsgroups_train = fetch_20newsgroups(subset="train")
    newsgroups_test = fetch_20newsgroups(subset="test")
    class_names = newsgroups_train.target_names
    y_train, y_test = newsgroups_train.target, newsgroups_test.target

    print(f"Training instances: {len(newsgroups_train.data)}")
    print(f"Test instances:     {len(newsgroups_test.data)}")
    print(f"Number of classes:  {len(class_names)}")
    print("\nClass distribution:")
    for i, name in enumerate(class_names):
        print(f"  [{i:2d}] {name:<26} Train: {np.sum(y_train == i):4d} | Test: {np.sum(y_test == i):4d}")


    # -----------------------------------------------------------------------------
    # 2. Text preprocessing
    # -----------------------------------------------------------------------------
    heading("2. Text Preprocessing")


    def preprocess_text(texts, apply_stemming=False):
        """
        Preprocess texts: lowercase, tokenize, filter non-letters and stop words,
        and optionally apply stemming.
        """
        preprocessed_texts = []
        for text in texts:
            tokens = word_tokenize(text.lower())
            # Keep only alphabetical tokens and filter out stopwords
            tokens = [t for t in tokens if t.isalpha() and t not in STOP_WORDS]
            if apply_stemming:
                tokens = [STEMMER.stem(t) for t in tokens]
            preprocessed_texts.append(" ".join(tokens))
        return preprocessed_texts


    print("Running text preprocessing...")
    start_time = time.time()
    newsgroups_train["preprocess_data"] = preprocess_text(newsgroups_train.data, apply_stemming=False)
    newsgroups_test["preprocess_data"] = preprocess_text(newsgroups_test.data, apply_stemming=False)

    newsgroups_train["preprocess_stem_data"] = preprocess_text(newsgroups_train.data, apply_stemming=True)
    newsgroups_test["preprocess_stem_data"] = preprocess_text(newsgroups_test.data, apply_stemming=True)
    print(f"Preprocessing completed in {time.time() - start_time:.1f}s")

    print("\nExample of first training document (first 250 characters):")
    print("Raw:")
    print(newsgroups_train.data[0][:250].strip())
    print("\nPreprocessed (no stemming):")
    print(newsgroups_train.preprocess_data[0][:250])
    print("\nPreprocessed (with stemming):")
    print(newsgroups_train.preprocess_stem_data[0][:250])


    # -----------------------------------------------------------------------------
    # 3. Feature representation
    # -----------------------------------------------------------------------------
    heading("3. Feature Representation")

    # For preprocessed texts, tokens are already lowercase and separated by space.
    # We pass tokenizer=str.split to keep the exact tokens from step 2.
    PRETOKENIZED_ARGS = {"tokenizer": str.split, "token_pattern": None, "lowercase": False}

    APPROACHES = {
        "Raw texts": ("data", {}),
        "Preprocessed (no stemming)": ("preprocess_data", PRETOKENIZED_ARGS),
        "Preprocessed (with stemming)": ("preprocess_stem_data", PRETOKENIZED_ARGS),
        "Preprocessed + bigrams": ("preprocess_stem_data", {**PRETOKENIZED_ARGS, "ngram_range": (1, 2)}),
    }

    VECTORIZERS = {
        "Count": CountVectorizer,
        "TF-IDF": TfidfVectorizer,
    }

    representations = {}
    vocab_sizes = {}

    for approach_name, (data_attr, extra_args) in APPROACHES.items():
        for vec_name, vec_class in VECTORIZERS.items():
            vectorizer = vec_class(**extra_args)
            X_train = vectorizer.fit_transform(newsgroups_train[data_attr])
            X_test = vectorizer.transform(newsgroups_test[data_attr])
            representations[(approach_name, vec_name)] = (X_train, X_test)
            if vec_name == "Count":
                vocab_sizes[approach_name] = len(vectorizer.vocabulary_)

    print("Vocabulary size (count vector dimension):")
    for approach, size in vocab_sizes.items():
        print(f"  {approach:<30} {size:>8,}")


    # -----------------------------------------------------------------------------
    # 4 & 5. Classification and evaluation
    # -----------------------------------------------------------------------------
    heading("4 & 5. Classification and Evaluation")

    # We test three supervised algorithms. max_iter is increased so solvers converge cleanly.
    CLASSIFIERS = {
        "Multinomial NB": lambda: MultinomialNB(),
        "Linear SVM": lambda: LinearSVC(max_iter=20000, random_state=0),
        "Logistic Regression": lambda: LogisticRegression(max_iter=3000),
    }

    results = []
    predictions = {}

    header = f"{'Approach':<30} {'Vector':<8} {'Classifier':<20} {'Acc':>6} {'P(macro)':>8} {'R(macro)':>8} {'F1(macro)':>9} {'Time':>6}"
    print(header)
    print("-" * len(header))

    for (approach_name, vec_name), (X_train, X_test) in representations.items():
        for clf_name, clf_factory in CLASSIFIERS.items():
            clf = clf_factory()
            t0 = time.time()
            clf.fit(X_train, y_train)
            fit_time = time.time() - t0

            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
            f1_w = f1_score(y_test, y_pred, average="weighted")

            row = {
                "approach": approach_name,
                "vectorizer": vec_name,
                "classifier": clf_name,
                "accuracy": acc,
                "precision_macro": p,
                "recall_macro": r,
                "f1_macro": f1,
                "f1_weighted": f1_w,
                "fit_seconds": fit_time,
            }
            results.append(row)
            predictions[(approach_name, vec_name, clf_name)] = y_pred

            print(
                f"{approach_name:<30} {vec_name:<8} {clf_name:<20} {acc:6.4f} "
                f"{p:8.4f} {r:8.4f} {f1:9.4f} {fit_time:5.1f}s"
            )

    # Identify best performing experiment
    best_exp = max(results, key=lambda x: x["accuracy"])
    best_key = (best_exp["approach"], best_exp["vectorizer"], best_exp["classifier"])
    y_best = predictions[best_key]

    print(f"\nBest Experiment:")
    print(f"  Classifier:     {best_exp['classifier']}")
    print(f"  Representation: {best_exp['approach']} + {best_exp['vectorizer']}")
    print(f"  Accuracy:       {best_exp['accuracy']:.4f}")
    print(f"  F1 (macro):     {best_exp['f1_macro']:.4f}")

    print("\nBest accuracy per classifier:")
    for clf_name in CLASSIFIERS:
        best_clf_row = max((r for r in results if r["classifier"] == clf_name), key=lambda r: r["accuracy"])
        print(f"  {clf_name:<20} {best_clf_row['accuracy']:.4f} ({best_clf_row['approach']}, {best_clf_row['vectorizer']})")

    print("\nMean accuracy across classifiers by representation:")
    mean_accuracy = {}
    for approach in APPROACHES:
        for vec_name in VECTORIZERS:
            accs = [r["accuracy"] for r in results if r["approach"] == approach and r["vectorizer"] == vec_name]
            mean_acc = float(np.mean(accs))
            mean_accuracy[f"{approach} | {vec_name}"] = mean_acc
            print(f"  {approach:<30} {vec_name:<8} {mean_acc:.4f}")


    # -----------------------------------------------------------------------------
    # 6. F1-Score Analysis (best experiment)
    # -----------------------------------------------------------------------------
    heading("6. F1-Score Analysis (Best Experiment)")

    f1_per_class = f1_score(y_test, y_best, average=None)
    print("F1-score per class (average=None):")
    for name, score in zip(class_names, f1_per_class):
        print(f"  {name:<28} {score:.4f}")

    f1_micro = f1_score(y_test, y_best, average="micro")
    f1_macro = f1_score(y_test, y_best, average="macro")
    f1_weighted = f1_score(y_test, y_best, average="weighted")

    print("\nAveraging methods across all 20 classes:")
    print(f"  micro:    {f1_micro:.4f} (equals accuracy: {accuracy_score(y_test, y_best):.4f})")
    print(f"  macro:    {f1_macro:.4f}")
    print(f"  weighted: {f1_weighted:.4f}")

    # Subsets of labels
    comp_indices = [i for i, n in enumerate(class_names) if n.startswith("comp.")]
    worst_5_indices = sorted(np.argsort(f1_per_class)[:5].tolist())
    best_5_indices = sorted(np.argsort(f1_per_class)[-5:].tolist())

    label_subsets = {
        "comp.* classes": comp_indices,
        "5 lowest-F1 classes": worst_5_indices,
        "5 highest-F1 classes": best_5_indices,
    }

    f1_subsets = {}
    print("\nF1-score on label subsets:")
    for subset_name, indices in label_subsets.items():
        print(f"\n  Subset: {subset_name} ({[class_names[i] for i in indices]})")
        f1_subsets[subset_name] = {"labels": [class_names[i] for i in indices]}
        for avg in ("micro", "macro", "weighted"):
            score = f1_score(y_test, y_best, labels=indices, average=avg)
            f1_subsets[subset_name][avg] = score
            print(f"    average={avg:<9} {score:.4f}")


    # -----------------------------------------------------------------------------
    # 7 & 8. Confusion Matrix
    # -----------------------------------------------------------------------------
    heading("7 & 8. Confusion Matrix")

    cm = confusion_matrix(y_test, y_best)
    class_precisions = cm.diagonal() / cm.sum(axis=0)
    class_recalls = cm.diagonal() / cm.sum(axis=1)

    print("Classes sorted by F1-score (lowest first):")
    print(f"  {'Class':<28} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Errors':>8}")
    for i in np.argsort(f1_per_class):
        err = cm[i].sum() - cm[i, i]
        print(f"  {class_names[i]:<28} {class_precisions[i]:10.4f} {class_recalls[i]:10.4f} {f1_per_class[i]:10.4f} {err:8d}")

    # Find most frequent off-diagonal confusions
    cm_off_diag = cm.copy()
    np.fill_diagonal(cm_off_diag, 0)
    top_confusions = []

    print("\nTop 10 most frequent classification mistakes:")
    for flat_idx in np.argsort(-cm_off_diag, axis=None)[:10]:
        true_idx, pred_idx = np.unravel_index(flat_idx, cm.shape)
        count = int(cm_off_diag[true_idx, pred_idx])
        pct_of_true = count / cm[true_idx].sum()
        top_confusions.append({
            "true": class_names[true_idx],
            "predicted": class_names[pred_idx],
            "count": count,
            "share_of_true": pct_of_true,
        })
        print(f"  {class_names[true_idx]:<26} -> {class_names[pred_idx]:<26} {count:3d} errors ({pct_of_true:.1%} of true class)")

    # Plot and save confusion matrix
    fig, ax = plt.subplots(figsize=(14, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation="vertical", colorbar=True)
    ax.set_title(
        f"Confusion Matrix: {best_exp['classifier']} ({best_exp['approach']} + {best_exp['vectorizer']})\n"
        f"Test Accuracy: {best_exp['accuracy']:.4f}",
        fontsize=12,
    )
    fig.tight_layout()
    cm_plot_path = OUTPUT_DIR / "confusion_matrix.png"
    fig.savefig(cm_plot_path, dpi=150)
    plt.close(fig)
    print(f"\nConfusion matrix saved to {cm_plot_path}")

    # Save results dictionary for report generation
    summary = {
        "train_size": len(newsgroups_train.data),
        "test_size": len(newsgroups_test.data),
        "class_names": class_names,
        "train_class_counts": np.bincount(y_train).tolist(),
        "test_class_counts": np.bincount(y_test).tolist(),
        "dimensions": vocab_sizes,
        "results": results,
        "mean_accuracy": mean_accuracy,
        "best": best_exp,
        "f1_per_class": dict(zip(class_names, f1_per_class.tolist())),
        "precision_per_class": dict(zip(class_names, class_precisions.tolist())),
        "recall_per_class": dict(zip(class_names, class_recalls.tolist())),
        "f1_all_labels": {
            "micro": f1_micro,
            "macro": f1_macro,
            "weighted": f1_weighted,
        },
        "f1_label_subsets": f1_subsets,
        "confusion_matrix": cm.tolist(),
        "top_confusions": top_confusions,
    }

    results_path = OUTPUT_DIR / "results.json"
    results_path.write_text(json.dumps(summary, indent=2))
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    run_lab()
