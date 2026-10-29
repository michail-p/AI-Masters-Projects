"""Score every system on the test split; write the numbers and figures used in the report.

Systems: the three baselines, the probes of every model (each task at the layer that is best on
the development split) and the fine-tuned models (averaged over seeds).
Subsets of the test instances:
  all, explicit (entity named in the step), implicit (not named),
  changed-explicit / changed-implicit (the state changes in this step),
  ProPara: implicit changes split by the manual verdict (genuine / reference / doubtful),
  bAbI:    "carrier moves": the object is not named, someone moves in this step and the object's
           location is known. It moved if its carrier is the one moving, otherwise it stayed.
           Keeping one's location and taking the mover's location are each right for one half,
           so only a system that tracks who carries what gets both.
Metrics: location accuracy; state-change accuracy and macro-F1; and change AUC, the ROC-AUC of
the predicted probability of a change against whether the state changed. Few implicit instances
change, so a well-calibrated classifier rarely predicts a change for them even when it has
information about it; the AUC measures that information without depending on the threshold.
Every number comes with a 95% document-level bootstrap interval.

Usage: python code/evaluate.py
Output: results/summary.json, results/tables.md, results/figures/*.png
"""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from baselines import baseline_predictions
from common import (CHANGES, DATASETS, EXISTENCE, MODEL_PARAMS, MODELS, RESULTS, SHORT, UNTRAINED,
                    existence_of, load_json, save_json)
from data import MOVE_RE
from extract import build_plan
from metrics import Bootstrap, accuracy, auc, interval, macro_f1

PLACE = "<place>"  # a predicted known location for an instance whose gold location is not a place
FINETUNED = ["pythia-160m", "pythia-410m"]  # smallest first; scored where results exist


# ---------------------------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------------------------

class Context:
    """Test-split instances, their gold labels and the location statements."""

    def __init__(self, dataset):
        self.dataset = dataset
        data, _, statements = build_plan(dataset)
        self.data = data
        self.docs = {d["id"]: d for d in data["docs"]}
        instances = data["instances"]
        splits = np.array([x["split"] for x in instances])
        self.test = np.flatnonzero(splits == "test")
        self.inst = [instances[i] for i in self.test]
        self.stmt_rows = np.flatnonzero(splits[statements["instance"]] == "test")
        self.statements = statements
        self.gold_change = np.array([x["change"] for x in self.inst])
        self.gold_location = np.array([x["loc"] for x in self.inst])
        self.doc_ids = np.array([x["doc"] for x in self.inst])

    def location_strings(self, existence, picked):
        """Combine the existence prediction with the chosen candidate into a location label."""
        out, k = [], 0
        for n, x in enumerate(self.inst):
            chosen = PLACE
            if x["existence"] == "known":
                row = self.stmt_rows[picked[k]]
                assert self.statements["instance"][row] == self.test[n]
                chosen = self.docs[x["doc"]]["candidates"][self.statements["candidate"][row]]
                k += 1
            kind = EXISTENCE[existence[n]]
            out.append("-" if kind == "nonexistent" else "?" if kind == "unknown" else chosen)
        assert k == len(picked)
        return np.array(out)


def best_layers(per_layer):
    return {task: int(np.argmax([layer[task]["dev"] for layer in per_layer])) for task in per_layer[0]}


def probe_predictions(ctx, model):
    scores = load_json(RESULTS / "probe" / ctx.dataset / f"{model}.json")["layers"]
    preds = np.load(RESULTS / "probe" / ctx.dataset / f"{model}.npz")
    assert (preds["test_rows"] == ctx.test).all()
    layer = best_layers(scores)
    return {"change": np.array(CHANGES)[preds["change"][layer["change"]]],
            "change_score": preds["change_score"][layer["change"]],
            "location": ctx.location_strings(preds["existence"][layer["existence"]],
                                             preds["location"][layer["location"]])}, layer


def finetune_predictions(ctx, model):
    sets = []
    for path in sorted((RESULTS / "finetune" / ctx.dataset).glob(f"{model}-seed*.npz")):
        p = np.load(path)
        sets.append({"change": np.array(CHANGES)[p["change"]], "change_score": p["change_score"],
                     "location": ctx.location_strings(p["existence"], p["location"])})
    return sets


def all_systems(ctx):
    """{system name: list of prediction sets (one per seed)}, and the probes' chosen layers."""
    systems, layers = {}, {}
    base = baseline_predictions(ctx.data)
    for name, p in base.items():
        systems[name] = [{task: np.array(v)[ctx.test] for task, v in p.items() if task != "existence"}]
    for model in [*MODELS, *UNTRAINED]:
        if (RESULTS / "probe" / ctx.dataset / f"{model}.json").exists():
            pred, layers[model] = probe_predictions(ctx, model)
            systems[f"probe {SHORT[model]}"] = [pred]
    for model in FINETUNED:
        sets = finetune_predictions(ctx, model)
        if sets:
            systems[f"fine-tuned {SHORT[model]}"] = sets
    return systems, layers


# ---------------------------------------------------------------------------------------------
# Subsets and scores
# ---------------------------------------------------------------------------------------------

def subsets(ctx):
    mentioned = np.array([x["mentioned"] for x in ctx.inst])
    changed = ctx.gold_change != "NONE"
    out = {"all": np.ones(len(ctx.inst), bool), "explicit": mentioned, "implicit": ~mentioned,
           "changed-explicit": changed & mentioned, "changed-implicit": changed & ~mentioned,
           "unchanged-implicit": ~changed & ~mentioned}
    if ctx.dataset == "propara":
        with open(RESULTS / "implicit_test_verdicts.csv", encoding="utf-8") as f:
            verdict = {(r["doc"], int(r["step"]), int(r["entity"])): r["verdict"] for r in csv.DictReader(f)}
        v = np.array([verdict.get((x["doc"], x["step"], x["entity"]), "") for x in ctx.inst])
        assert (v != "")[changed & ~mentioned].all()
        for kind in ("genuine", "reference", "doubtful"):
            out[f"changed-implicit {kind}"] = v == kind
        # The same changes together with every unchanged implicit instance, so that the change AUC
        # (which needs both classes) can be computed for each kind of implicit change.
        for kind in ("genuine", "reference", "doubtful"):
            out[f"implicit, {kind} changes"] = (v == kind) | (~changed & ~mentioned)
    else:
        mover_step = np.array([bool(MOVE_RE.match(ctx.docs[x["doc"]]["sentences"][x["step"] - 1]))
                               for x in ctx.inst])
        known_before = np.array([existence_of(x["prev"]) == "known" for x in ctx.inst])
        carrier = mover_step & known_before & ~mentioned
        out["carrier-moves"] = carrier
        out["carrier-moves moved"] = carrier & changed
        out["carrier-moves stayed"] = carrier & ~changed
    return out


def score_system(pred_sets, ctx, masks, boot):
    """Point estimates (mean over seeds) and bootstrap replicates for every subset."""
    result, replicates = {}, {}
    change_index = {c: i for i, c in enumerate(CHANGES)}
    gold_change = np.array([change_index[c] for c in ctx.gold_change])
    for subset, mask in masks.items():
        row, reps = {"n": int(mask.sum())}, {}
        loc_correct = np.mean([p["location"] == ctx.gold_location for p in pred_sets], 0)
        row["location_acc"] = float(loc_correct[mask].mean()) if mask.any() else float("nan")
        reps["location_acc"] = boot.accuracy(loc_correct, mask)
        if "change" in pred_sets[0]:
            pred_change = [np.array([change_index[c] for c in p["change"]]) for p in pred_sets]
            row["change_acc"] = float(np.mean([accuracy(gold_change[mask], p[mask]) for p in pred_change]))
            row["change_f1"] = float(np.mean([macro_f1(gold_change[mask], p[mask]) for p in pred_change]))
            reps["change_acc"] = boot.accuracy(np.mean([p == gold_change for p in pred_change], 0), mask)
            reps["change_f1"] = np.mean([boot.macro_f1(gold_change, p, len(CHANGES), mask) for p in pred_change], 0)
        if "change_score" in pred_sets[0]:
            # Can the change probability separate changed from unchanged instances in the subset?
            positive = gold_change != change_index["NONE"]
            row["change_auc"] = float(np.mean([auc(positive[mask], p["change_score"][mask]) for p in pred_sets]))
            reps["change_auc"] = np.mean([boot.auc(positive, p["change_score"], mask) for p in pred_sets], 0)
        if len(pred_sets) > 1:  # spread over seeds
            row["location_acc_seeds"] = [float((p["location"] == ctx.gold_location)[mask].mean()) for p in pred_sets]
        for metric, r in reps.items():
            row[f"{metric}_ci"] = interval(r)
        result[subset], replicates[subset] = row, reps
    return result, replicates


def evaluate(dataset):
    ctx = Context(dataset)
    masks = subsets(ctx)
    boot = Bootstrap(ctx.doc_ids)
    systems, layers = all_systems(ctx)
    scores, reps = {}, {}
    for name, pred_sets in systems.items():
        scores[name], reps[name] = score_system(pred_sets, ctx, masks, boot)

    # Paired differences used in the text.
    diffs = {}

    def diff(label, a, b, subset_a, subset_b, metric):
        if a in reps and b in reps and metric in reps[a][subset_a] and metric in reps[b][subset_b]:
            d = reps[a][subset_a][metric] - reps[b][subset_b][metric]
            diffs[label] = {"estimate": scores[a][subset_a][metric] - scores[b][subset_b][metric],
                            "ci": interval(d)}

    for name in systems:
        # RQ2: drop from explicit to implicit changes, for the same system
        for metric in ("location_acc", "change_acc"):
            diff(f"{name}: changed-explicit - changed-implicit, {metric}", name, name,
                 "changed-explicit", "changed-implicit", metric)
        diff(f"{name}: explicit - implicit, change_auc", name, name, "explicit", "implicit", "change_auc")
    for name in [s for s in systems if s.startswith(("probe", "fine-tuned"))]:
        for subset in ("implicit", "changed-implicit", "carrier-moves"):
            if subset not in masks:
                continue
            diff(f"{name} - previous: {subset}, location_acc", name, "previous", subset, subset, "location_acc")
            diff(f"{name} - recent: {subset}, location_acc", name, "recent", subset, subset, "location_acc")
            diff(f"{name} - majority: {subset}, location_acc", name, "majority", subset, subset, "location_acc")
            diff(f"{name} - majority: {subset}, change_f1", name, "majority", subset, subset, "change_f1")
    rq3 = [("all", "location_acc"), ("all", "change_f1"), ("all", "change_auc"),
           ("implicit", "location_acc"), ("implicit", "change_f1"), ("implicit", "change_auc"),
           ("changed-implicit", "location_acc"), ("changed-implicit", "change_f1")]
    for model in FINETUNED:  # RQ3: probe vs fine-tuning at the same size
        p, f = f"probe {SHORT[model]}", f"fine-tuned {SHORT[model]}"
        for subset, metric in rq3:
            diff(f"{f} - {p}: {subset}, {metric}", f, p, subset, subset, metric)
    # RQ3: does the gap between fine-tuning and the probe shrink from the smaller to the larger
    # model? Negative = the larger model's gap is smaller.
    finetuned = [m for m in FINETUNED if f"fine-tuned {SHORT[m]}" in reps]
    for small, large in zip(finetuned, finetuned[1:]):
        for subset, metric in rq3:
            gap = {m: (scores[f"fine-tuned {SHORT[m]}"][subset][metric] - scores[f"probe {SHORT[m]}"][subset][metric],
                       reps[f"fine-tuned {SHORT[m]}"][subset][metric] - reps[f"probe {SHORT[m]}"][subset][metric])
                   for m in (small, large)}
            diffs[f"gap {SHORT[large]} - gap {SHORT[small]}: {subset}, {metric}"] = {
                "estimate": gap[large][0] - gap[small][0], "ci": interval(gap[large][1] - gap[small][1])}

    probe_layers = {}
    for model, chosen in layers.items():
        per_layer = load_json(RESULTS / "probe" / dataset / f"{model}.json")["layers"]
        probe_layers[model] = {
            "layers": len(per_layer),
            "chosen": chosen,
            "test_at_chosen": {t: per_layer[chosen[t]][t]["test"] for t in chosen},
            "curves": {t: [layer[t]["test"] for layer in per_layer] for t in per_layer[0]},
        }
        # Change AUC per layer, within explicit and within implicit instances.
        change_scores = np.load(RESULTS / "probe" / dataset / f"{model}.npz")["change_score"]
        positive = ctx.gold_change != "NONE"
        for subset in ("explicit", "implicit"):
            m = masks[subset]
            probe_layers[model]["curves"][f"change_auc_{subset}"] = [auc(positive[m], c[m]) for c in change_scores]
        if "control_change" in chosen:
            probe_layers[model]["selectivity"] = {
                "change": per_layer[chosen["change"]]["change"]["test"]
                - per_layer[chosen["change"]]["control_change"]["test"],
                "location": per_layer[chosen["location"]]["location"]["test"]
                - per_layer[chosen["location"]]["control_location"]["test"],
            }
    n_candidates = [len(ctx.docs[x["doc"]]["candidates"]) for x in ctx.inst if x["existence"] == "known"]
    return {"scores": scores, "differences": diffs, "probe_layers": probe_layers,
            "candidate_chance": float(np.mean([1 / n for n in n_candidates])),
            "finetune": {m: runs for m in FINETUNED
                         if (runs := [load_json(p) for p in sorted((RESULTS / "finetune" / dataset).glob(f"{m}-seed*.json"))])}}


# ---------------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------------

def fmt(row, metric):
    if metric not in row or row["n"] == 0:
        return "-"
    lo, hi = row[f"{metric}_ci"]
    return f"{100 * row[metric]:.1f} [{100 * lo:.0f}, {100 * hi:.0f}]"


def tables_markdown(summary):
    lines = []
    for dataset, result in summary.items():
        scores = result["scores"]
        subset_names = list(next(iter(scores.values())).keys())
        for metric in ("location_acc", "change_acc", "change_f1", "change_auc"):
            lines += [f"\n## {dataset}: {metric} (test, 95% CI)\n",
                      "| system | " + " | ".join(f"{s} (n={scores['previous'][s]['n']})" for s in subset_names) + " |",
                      "|---" * (len(subset_names) + 1) + "|"]
            for name, rows in scores.items():
                lines.append(f"| {name} | " + " | ".join(fmt(rows[s], metric) for s in subset_names) + " |")
        lines += [f"\n## {dataset}: paired differences\n"]
        for label, d in result["differences"].items():
            lines.append(f"- {label}: {100 * d['estimate']:+.1f} [{100 * d['ci'][0]:+.1f}, {100 * d['ci'][1]:+.1f}]")
        lines += [f"\n## {dataset}: probe layers (dev-chosen) and selectivity\n"]
        for model, info in result["probe_layers"].items():
            lines.append(f"- {model}: {info['layers']} layers, chosen {info['chosen']}, "
                         f"test {({k: round(v, 3) for k, v in info['test_at_chosen'].items()})}, "
                         f"selectivity {({k: round(v, 3) for k, v in info.get('selectivity', {}).items()})}")
        lines.append(f"- chance of a random candidate: {result['candidate_chance']:.3f}")
        for model, runs in result["finetune"].items():
            lines.append(f"- fine-tuned {model}: seeds {[r['seed'] for r in runs]}, "
                         f"best epochs {[r['best_epoch'] for r in runs]} of {[r['args']['epochs'] for r in runs]}, "
                         f"dev {[round(r['history'][r['best_epoch'] - 1]['score'], 3) for r in runs]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# Figures (palette and mark specs from the dataviz reference palette)
# ---------------------------------------------------------------------------------------------

BLUE, ORANGE = "#2a78d6", "#eb6834"
SIZE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # ordinal blue, small -> large
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NAMES = {"propara": "ProPara", "babi": "bAbI task 2"}


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=8)


def figure_layers(summary, path):
    """Test score per layer (x: relative depth) for every model size: change AUC on implicit
    instances, and accuracy of the location probe at picking the right candidate."""
    tasks = [("change_auc_implicit", "change AUC, implicit instances"),
             ("location", "location probe, candidate accuracy")]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6), sharex=True)
    for row, dataset in enumerate(DATASETS):
        info = summary[dataset]["probe_layers"]
        for col, (task, title) in enumerate(tasks):
            ax = axes[row, col]
            for model, color in [*zip(MODELS, SIZE_RAMP), ("pythia-410m-step0", "#a3a29c")]:
                if model not in info:
                    continue
                curve = np.array(info[model]["curves"][task])
                depth = np.arange(len(curve)) / (len(curve) - 1)
                ax.plot(depth, curve, color=color, linewidth=1.6, label=SHORT[model], solid_capstyle="round")
                chosen = info[model]["chosen"]["change" if task.startswith("change") else task]
                ax.plot(depth[chosen], curve[chosen], "o", color=color, markersize=5,
                        markeredgecolor="white", markeredgewidth=1.0)
            chance = 0.5 if task.startswith("change") else summary[dataset]["candidate_chance"]
            ax.axhline(chance, color=INK_2, linewidth=0.8)
            ax.text(1.0, chance, "chance", color=INK_2, fontsize=7, ha="right", va="bottom")
            style(ax)
            ax.set_title(f"{NAMES[dataset]}: {title}", fontsize=9, color=INK, loc="left")
            if row == 1:
                ax.set_xlabel("relative layer depth (0 = embeddings)", fontsize=8, color=INK_2)
    axes[0, 0].legend(title="probe on Pythia", fontsize=7, title_fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def figure_explicit_implicit(summary, path):
    """Location accuracy on changed instances, explicit vs implicit, per system."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        scores = summary[dataset]["scores"]
        names = [n for n in scores if n not in ("majority", "previous")]  # previous is 0 by definition
        x = np.arange(len(names))
        for offset, subset, color, label in ((-0.19, "changed-explicit", BLUE, "named in the step"),
                                             (0.19, "changed-implicit", ORANGE, "not named (implicit)")):
            values = np.array([scores[n][subset]["location_acc"] for n in names])
            ci = np.array([scores[n][subset]["location_acc_ci"] for n in names])
            ax.bar(x + offset, values, width=0.36, color=color, label=label)
            ax.errorbar(x + offset, values, yerr=[values - ci[:, 0], ci[:, 1] - values], fmt="none",
                        ecolor=INK_2, elinewidth=0.8, capsize=0)
        style(ax)
        ax.set_xticks(x, [n.replace("fine-tuned", "FT").replace("probe ", "") for n in names], rotation=45,
                      ha="right", fontsize=7)
        ax.set_title(NAMES[dataset], fontsize=9, color=INK, loc="left")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("location accuracy, changed instances", fontsize=8, color=INK_2)
    axes[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def figure_size(summary, path):
    """Probe score against model size on implicit changes, with fine-tuned models and baselines."""
    panels = [("implicit", "change_auc", "change AUC, implicit instances"),
              ("changed-implicit", "location_acc", "location accuracy, implicit changes")]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6))
    for row, dataset in enumerate(DATASETS):
        scores = summary[dataset]["scores"]
        for col, (subset, metric, title) in enumerate(panels):
            ax = axes[row, col]
            models = [m for m in MODELS if f"probe {SHORT[m]}" in scores]
            params = np.array([MODEL_PARAMS[m] for m in models])
            vals = np.array([scores[f"probe {SHORT[m]}"][subset][metric] for m in models])
            ci = np.array([scores[f"probe {SHORT[m]}"][subset][f"{metric}_ci"] for m in models])
            ax.fill_between(params, ci[:, 0], ci[:, 1], color=BLUE, alpha=0.12, linewidth=0)
            ax.plot(params, vals, "-o", color=BLUE, linewidth=1.6, markersize=5, markeredgecolor="white",
                    markeredgewidth=1.0, label="frozen probe")
            ft = [m for m in FINETUNED if f"fine-tuned {SHORT[m]}" in scores]
            if ft:
                fv = np.array([scores[f"fine-tuned {SHORT[m]}"][subset][metric] for m in ft])
                fci = np.array([scores[f"fine-tuned {SHORT[m]}"][subset][f"{metric}_ci"] for m in ft])
                ax.errorbar([MODEL_PARAMS[m] * 1.12 for m in ft], fv, yerr=[fv - fci[:, 0], fci[:, 1] - fv], fmt="s",
                            color=ORANGE, markersize=5, elinewidth=0.8, capsize=0, label="fine-tuned")
            references = {name: scores[name][subset][metric] for name in ("previous", "recent", "majority")
                          if metric in scores[name][subset]}
            if metric == "change_auc":
                references = {"chance": 0.5}
            merged = {}  # baselines with the same value share one line and one label
            for name, v in references.items():
                merged.setdefault(round(v, 3), []).append(name)
            for v, names in merged.items():
                ax.axhline(v, color=INK_2, linewidth=0.8)
                ax.text(params[-1] * 1.25, v + 0.015, ", ".join(names), color=INK_2, fontsize=7, va="bottom")
            ax.set_xscale("log")
            ax.set_xticks(params, [SHORT[m] for m in models])
            ax.minorticks_off()
            ax.set_xlim(params[0] / 1.3, params[-1] * 2.2)
            ax.set_ylim(0, 1.05)
            style(ax)
            ax.set_title(f"{NAMES[dataset]}: {title}", fontsize=9, color=INK, loc="left")
    axes[0, 0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    summary = {dataset: evaluate(dataset) for dataset in DATASETS
               if (RESULTS / "probe" / dataset).exists()}
    save_json(summary, RESULTS / "summary.json")
    (RESULTS / "tables.md").write_text(tables_markdown(summary), encoding="utf-8")
    figures = RESULTS / "figures"
    figures.mkdir(exist_ok=True)
    if all(d in summary for d in DATASETS):
        figure_layers(summary, figures / "layers.png")
        figure_explicit_implicit(summary, figures / "explicit_implicit.png")
        figure_size(summary, figures / "size.png")
    print((RESULTS / "tables.md").read_text(encoding="utf-8"))
