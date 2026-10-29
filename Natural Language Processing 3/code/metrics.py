"""Accuracy, macro-F1 and a document-level bootstrap.

Instances from the same paragraph or story are not independent (same entities, same text), so
the bootstrap resamples whole documents. All systems are scored on the same resamples (same
seed), which makes differences between two systems paired.
"""

import numpy as np
from sklearn.metrics import f1_score

N_BOOTSTRAP = 1000


def accuracy(gold, pred):
    gold, pred = np.asarray(gold), np.asarray(pred)
    return float((gold == pred).mean()) if len(gold) else float("nan")


def macro_f1(gold, pred):
    """Macro-F1 over the classes that occur in the gold labels or the predictions."""
    if len(gold) == 0:
        return float("nan")
    return float(f1_score(gold, pred, average="macro", zero_division=0))


class Bootstrap:
    """Bootstrap replicates of accuracy and macro-F1 for a fixed set of instances."""

    def __init__(self, docs, n=N_BOOTSTRAP, seed=0):
        unique, self.doc_of = np.unique(np.asarray(docs), return_inverse=True)
        draws = np.random.default_rng(seed).integers(0, len(unique), size=(n, len(unique)))
        # weights[r, d] = how often document d is drawn in replicate r
        self.weights = np.zeros((n, len(unique)))
        np.add.at(self.weights, (np.arange(n)[:, None], draws), 1)

    def per_doc(self, values):
        return np.bincount(self.doc_of, weights=values, minlength=self.weights.shape[1])

    def accuracy(self, correct, mask):
        """Replicates of the mean of `correct` (booleans, or fractions when averaging seeds) over
        the instances selected by `mask`. Every subset is drawn from the same resampled documents,
        so differences between subsets are paired too."""
        correct, mask = np.asarray(correct, dtype=float), np.asarray(mask, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (self.weights @ self.per_doc(correct * mask)) / (self.weights @ self.per_doc(mask))

    def macro_f1(self, gold, pred, n_classes, mask):
        """Replicates of macro-F1 over the classes present in gold or predictions, as macro_f1()."""
        cell = np.where(mask, np.asarray(gold) * n_classes + np.asarray(pred), -1)
        per_doc = np.stack([self.per_doc((cell == c).astype(float)) for c in range(n_classes ** 2)], 1)
        confusion = (self.weights @ per_doc).reshape(-1, n_classes, n_classes)
        tp = np.diagonal(confusion, axis1=1, axis2=2)
        support = confusion.sum(2) + confusion.sum(1)  # gold count + predicted count per class
        with np.errstate(invalid="ignore", divide="ignore"):
            f1 = np.where(support > 0, 2 * tp / support, np.nan)
        return np.nanmean(f1, axis=1)


    def auc(self, positive, scores, mask):
        """Replicates of ROC-AUC (probability that a positive outscores a negative, ties count
        half) over the instances selected by `mask`, with each instance weighted by how often its
        document is drawn."""
        order = np.argsort(scores, kind="stable")
        s, y = np.asarray(scores)[order], np.asarray(positive, dtype=float)[order]
        w = self.weights[:, self.doc_of[order]] * np.asarray(mask, dtype=float)[order]  # [replicates, n]
        wp, wn = w * y, w * (1 - y)
        below = np.concatenate([np.zeros((len(w), 1)), np.cumsum(wn, 1)], 1)  # negatives before each position
        starts = np.searchsorted(s, s, side="left")  # first position of each instance's tie group
        ends = np.searchsorted(s, s, side="right")
        under = below[:, starts] + 0.5 * (below[:, ends] - below[:, starts])
        with np.errstate(invalid="ignore", divide="ignore"):
            return (wp * under).sum(1) / (wp.sum(1) * wn.sum(1))


def auc(positive, scores):
    from sklearn.metrics import roc_auc_score
    positive = np.asarray(positive)
    return float(roc_auc_score(positive, scores)) if 0 < positive.sum() < len(positive) else float("nan")


def interval(replicates):
    """95% percentile interval."""
    return [float(x) for x in np.nanpercentile(replicates, [2.5, 97.5])]
