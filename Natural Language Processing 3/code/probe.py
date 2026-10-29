"""Linear probes on frozen hidden states, one per layer, plus a control task.

Three probes are trained per layer, all logistic regression with an L2 penalty:
  - change:    4-way (NONE/CREATE/MOVE/DESTROY) on the query states " The <entity> is" after and
               before the step, concatenated
  - existence: 3-way (does not exist / unknown / known place) on the query state after the step
  - location:  a linear score for each candidate statement " The <entity> is in the <place>.",
               normalised with a softmax over the candidates of the instance (a conditional logit),
               trained on the instances whose gold location is a known place
The location prediction combines the two: '-' or '?' if the existence probe says so, otherwise
the highest-scoring candidate.

Control task (Hewitt and Liang, 2019): the same probes are trained on labels that depend only on
names. Every entity name gets a random change label (drawn from the training label
distribution), and every location name a random rank; the control location of an instance is its
candidate with the highest rank. A probe can learn both only by memorising names seen in
training, so a probe that does well on them is memorising names rather than reading the state,
and selectivity = task - control score.
The control is run on ProPara only: bAbI has three object names and six places, so any name-based
control is trivially learnable there and says nothing about the probe. Both are scored with the
task's metric (macro-F1 for change, accuracy for location).

The penalty strength and the layer are chosen on the development split; test results are saved
for every layer so they can be plotted.

Usage: python code/probe.py --model pythia-410m --dataset propara
Output: results/probe/{dataset}/{model}.npz (predictions per layer) and .json (dev/test scores)
"""

import argparse
import hashlib

import numpy as np
import torch
import torch.nn.functional as F

from common import CHANGES, DATA, EXISTENCE, FEATURES, RESULTS, load_json, save_json, seed_everything
from metrics import macro_f1

PENALTIES = [1e-4, 1e-3, 1e-2, 1e-1]  # L2 strengths tried; the best on the dev split is kept
DEVICE = "cuda"


# ---------------------------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------------------------

def standardise(train, *others):
    """z-score every dimension with training statistics. Pythia's residual stream has a few
    dimensions with very large values that would otherwise dominate the L2 penalty."""
    mean, std = train.mean(0, keepdim=True), train.std(0, keepdim=True) + 1e-4
    return [(x - mean) / std for x in (train, *others)]


def fit_lbfgs(params, loss_fn, max_iter=300):
    opt = torch.optim.LBFGS(params, lr=1, max_iter=max_iter, history_size=20,
                            tolerance_grad=1e-6, tolerance_change=1e-9, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss

    opt.step(closure)


def fit_softmax(x, y, n_classes, penalty):
    """Multinomial logistic regression; returns (W, b)."""
    W = torch.zeros(x.shape[1], n_classes, device=DEVICE, requires_grad=True)
    b = torch.zeros(n_classes, device=DEVICE, requires_grad=True)
    fit_lbfgs([W, b], lambda: F.cross_entropy(x @ W + b, y) + penalty * (W ** 2).sum())
    return W.detach(), b.detach()


def group_softmax_loss(scores, group, gold):
    """Cross-entropy of a softmax over the candidates of each instance. `group` gives the instance
    of each statement (consecutive ids from 0) and `gold` marks the correct candidate."""
    n = int(group.max()) + 1
    peak = torch.zeros(n, device=DEVICE).scatter_reduce(0, group, scores.detach(), "amax", include_self=False)
    exp = torch.exp(scores - peak[group])
    log_norm = torch.log(torch.zeros(n, device=DEVICE).index_add(0, group, exp)) + peak
    return (log_norm - torch.zeros(n, device=DEVICE).index_add(0, group, scores * gold)).mean()


def fit_conditional_logit(x, group, gold, penalty):
    w = torch.zeros(x.shape[1], device=DEVICE, requires_grad=True)
    fit_lbfgs([w], lambda: group_softmax_loss(x @ w, group, gold) + penalty * (w ** 2).sum())
    return w.detach()


def pick_candidates(scores, group, n_groups):
    """Index (into the statements) of the highest-scoring candidate of each instance. Ties are
    broken at random: at the embedding layer every statement ends in the same '.' vector, and a
    fixed tie-break would turn the candidate order into a fake signal."""
    jitter = torch.rand(len(scores), generator=torch.Generator(device=DEVICE).manual_seed(0), device=DEVICE)
    scores = scores + 1e-6 * scores.abs().max().clamp(min=1.0) * jitter
    best = torch.full((n_groups,), -1, dtype=torch.long, device=DEVICE)
    peak = torch.full((n_groups,), -float("inf"), device=DEVICE).scatter_reduce(0, group, scores, "amax")
    is_peak = scores == peak[group]
    rows = torch.arange(len(scores), device=DEVICE)
    best.scatter_reduce_(0, group[is_peak], rows[is_peak], "amin", include_self=False)
    return best


# ---------------------------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------------------------

def stable_hash(text):
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)


def control_labels(data, statements):
    """Name-only control labels for the change task (from the entity name) and the location task
    (from the location names)."""
    docs = {d["id"]: d for d in data["docs"]}
    instances = data["instances"]
    names = [docs[x["doc"]]["entities"][x["entity"]]["name"].lower() for x in instances]
    train_changes = [CHANGES.index(x["change"]) for x in instances if x["split"] == "train"]
    rng = np.random.default_rng(0)
    change_of_name = {n: int(rng.choice(train_changes)) for n in sorted(set(names))}
    change = np.array([change_of_name[n] for n in names])
    # Location: the candidate whose name has the highest random rank (a hash of the name). A
    # statement's score can learn this from the place it names, as the real task's score can.
    gold = np.zeros(len(statements["instance"]), dtype=bool)
    for s, (i, c) in enumerate(zip(statements["instance"], statements["candidate"])):
        candidates = docs[instances[i]["doc"]]["candidates"]
        gold[s] = candidates[c] == max(candidates, key=stable_hash)
    return change, gold


def load_layer(path, layer, rows=None):
    feats = np.load(path, mmap_mode="r")[layer]
    return torch.tensor(np.asarray(feats if rows is None else feats[rows]), dtype=torch.float32, device=DEVICE)


# ---------------------------------------------------------------------------------------------
# One layer
# ---------------------------------------------------------------------------------------------

def run_layer(layer, feat_dir, split, previous, labels, stmt, controls):
    """Train all probes on one layer. Returns dev/test scores and test-split predictions."""
    q = load_layer(feat_dir / "queries.npy", layer)
    prev = torch.cat([q, load_layer(feat_dir / "initial.npy", layer)])[previous]
    names = ("train", "dev", "test")
    now = dict(zip(names, standardise(*(q[split[k]] for k in names))))
    before = dict(zip(names, standardise(*(prev[split[k]] for k in names))))
    del q, prev
    # Existence is a state: the query after the step. A change relates two states: the query
    # states after and before the step, side by side.
    q_feats = now
    pair_feats = {k: torch.cat([now[k], before[k]], 1) for k in names}
    s = load_layer(feat_dir / "statements.npy", layer)
    s_feats = dict(zip(names, standardise(*(s[stmt[k]["rows"]] for k in names))))
    del s
    scores, preds = {}, {}

    def classify(name, feats, y, n_classes, metric):
        """Fit for every penalty, keep the one with the best dev score. For the change task the
        test probability of a change, 1 - P(NONE), is kept as well: it shows whether changes are
        separable from non-changes even where the prior keeps the predicted class at NONE."""
        best = None
        for penalty in PENALTIES:
            W, b = fit_softmax(feats["train"], y["train"], n_classes, penalty)
            logits = {k: feats[k] @ W + b for k in ("dev", "test")}
            pred = {k: v.argmax(1).cpu().numpy() for k, v in logits.items()}
            changed = 1 - torch.softmax(logits["test"], 1)[:, CHANGES.index("NONE")].cpu().numpy()
            dev = metric(y["dev"].cpu().numpy(), pred["dev"])
            if best is None or dev > best[0]:
                best = (dev, penalty, pred, changed)
        dev, penalty, pred, changed = best
        scores[name] = {"dev": dev, "test": metric(y["test"].cpu().numpy(), pred["test"]), "penalty": penalty}
        preds[name] = pred["test"]
        if name == "change":
            preds["change_score"] = changed.astype(np.float32)

    def locate(name, gold):
        """Location probe: pick the best candidate per instance."""
        best = None
        for penalty in PENALTIES:
            w = fit_conditional_logit(s_feats["train"], stmt["train"]["group"], gold["train"], penalty)
            picked = {}
            for k in ("dev", "test"):
                g = stmt[k]["group"]
                rows = pick_candidates(s_feats[k] @ w, g, int(g.max()) + 1)
                picked[k] = rows.cpu().numpy()
            dev = gold["dev"].cpu().numpy()[picked["dev"]].mean()
            if best is None or dev > best[0]:
                best = (dev, penalty, picked)
        dev, penalty, picked = best
        test_acc = gold["test"].cpu().numpy()[picked["test"]].mean()
        scores[name] = {"dev": float(dev), "test": float(test_acc), "penalty": penalty}
        preds[name] = picked["test"]  # statement row (within the test statements) of the chosen candidate

    classify("change", pair_feats, labels["change"], len(CHANGES), macro_f1)
    classify("existence", q_feats, labels["existence"], len(EXISTENCE), macro_f1)
    locate("location", {k: v["gold"] for k, v in stmt.items()})
    if controls is not None:
        classify("control_change", pair_feats, controls["change"], len(CHANGES), macro_f1)
        locate("control_location", controls["location"])
    return scores, preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True, choices=["propara", "babi"])
    args = parser.parse_args()
    seed_everything(0)

    data = load_json(DATA / f"{args.dataset}.json")
    feat_dir = FEATURES / args.dataset / args.model
    meta = load_json(feat_dir / "meta.json")
    index = dict(np.load(feat_dir / "index.npz"))
    previous = torch.tensor(index["previous"], device=DEVICE)
    instances = data["instances"]
    splits = np.array([x["split"] for x in instances])
    split = {k: np.flatnonzero(splits == k) for k in ("train", "dev", "test")}

    def as_tensor(values, rows):
        return torch.tensor(np.asarray(values)[rows], device=DEVICE)

    change = np.array([CHANGES.index(x["change"]) for x in instances])
    existence = np.array([EXISTENCE.index(x["existence"]) for x in instances])
    labels = {"change": {k: as_tensor(change, r) for k, r in split.items()},
              "existence": {k: as_tensor(existence, r) for k, r in split.items()}}

    # Statements of each split, with their instance renumbered 0..n-1 within the split.
    stmt = {}
    for k in ("train", "dev", "test"):
        rows = np.flatnonzero(splits[index["instance"]] == k)
        _, group = np.unique(index["instance"][rows], return_inverse=True)
        stmt[k] = {"rows": rows, "group": torch.tensor(group, device=DEVICE),
                   "gold": torch.tensor(index["gold"][rows], dtype=torch.float32, device=DEVICE)}

    controls = None
    if args.dataset == "propara":
        control_change, control_gold = control_labels(data, index)
        controls = {"change": {k: as_tensor(control_change, r) for k, r in split.items()},
                    "location": {k: torch.tensor(control_gold[stmt[k]["rows"]], dtype=torch.float32, device=DEVICE)
                                 for k in stmt}}

    per_layer, predictions = [], {}
    for layer in range(meta["layers"]):
        scores, preds = run_layer(layer, feat_dir, split, previous, labels, stmt, controls)
        per_layer.append(scores)
        for name, p in preds.items():
            predictions.setdefault(name, []).append(p)
        print(f"layer {layer:2d}: " + "  ".join(f"{n} {s['dev']:.3f}/{s['test']:.3f}" for n, s in scores.items()),
              flush=True)

    out = RESULTS / "probe" / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{args.model}.npz", test_rows=split["test"], test_statement_rows=stmt["test"]["rows"],
             **{name: np.stack(p) for name, p in predictions.items()})
    save_json({"model": args.model, "layers": per_layer}, out / f"{args.model}.json")


if __name__ == "__main__":
    main()
