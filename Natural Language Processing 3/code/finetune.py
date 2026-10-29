"""Fine-tune a Pythia model on the same three tasks the probes solve.

The model reads exactly what the probes read (same prefixes, queries and statements, see
extract.py) and the same three linear heads sit on the last-token state of the final layer:
  change (4 classes) on the query states after and before the step, existence (3 classes) on the
  query after the step, and a score per candidate statement for location. The difference to the probes is that all weights of the language model are trained
  too, so comparing the two shows how much fine-tuning adds beyond reading out what is there.

Training is multi-task: the loss is the sum of the three cross-entropies (the location one is a
softmax over the candidates of each instance, as in the probe). The epoch with the best
development score (mean of change macro-F1 and location accuracy) is kept.

Weights and computation are float32, like the probes' features (see extract.py on why half
precision is avoided). The optimizer keeps its moment estimates in 8 bits (bitsandbytes
AdamW8bit) so that full fine-tuning of the 410M model fits in 8 GB of GPU memory.

Usage: python code/finetune.py --model pythia-410m --dataset propara [--seed 0]
Output: results/finetune/{dataset}/{model}-seed{seed}.npz (test predictions) and .json (scores)
"""

import argparse
import math
import random
import time

import bitsandbytes as bnb
import numpy as np
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from common import CHANGES, EXISTENCE, MODELS, RESULTS, save_json, seed_everything
from extract import PrefixTokenizer, build_plan, read_group
from metrics import macro_f1
from probe import group_softmax_loss

DEVICE = "cuda"


class Heads(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.change = nn.Linear(2 * dim, len(CHANGES))  # query states after and before the step
        self.existence = nn.Linear(dim, len(EXISTENCE))
        self.location = nn.Linear(dim, 1)


def group_tensors(group, instances, statements):
    """Labels of one (document, step) group, and each statement's instance numbered within the group."""
    q_rows = [i for i, _ in group["queries"]]
    s_rows = [s for s, _ in group["statements"]]
    change = torch.tensor([CHANGES.index(instances[i]["change"]) for i in q_rows], device=DEVICE)
    existence = torch.tensor([EXISTENCE.index(instances[i]["existence"]) for i in q_rows], device=DEVICE)
    local = {i: n for n, i in enumerate(dict.fromkeys(statements["instance"][s_rows].tolist()))}
    s_group = torch.tensor([local[i] for i in statements["instance"][s_rows]], device=DEVICE, dtype=torch.long)
    s_gold = torch.tensor(statements["gold"][s_rows], device=DEVICE, dtype=torch.float32)
    return change, existence, s_group, s_gold


def forward(model, heads, ptok, group, before, grad):
    """Logits for one group. `before` is the group of the previous step (step 0 for the first),
    whose queries give the state before the step for the change head."""
    last = read_group(model, ptok, group, DEVICE, grad=grad)[-1]
    queries_only = {"doc": before["doc"], "step": before["step"], "queries": before["queries"], "statements": []}
    prev = read_group(model, ptok, queries_only, DEVICE, grad=grad)[-1]
    nq = len(group["queries"])
    return (heads.change(torch.cat([last[:nq], prev], 1)), heads.existence(last[:nq]),
            heads.location(last[nq:]).squeeze(-1))


@torch.no_grad()
def predict(model, heads, ptok, groups, before_of, statements):
    """Predictions for a list of groups, in the format probe.py saves for one layer: change and
    existence class per instance, and for location the statement row (counted within these
    groups' statements) of the chosen candidate of each instance with a known gold location,
    plus the probability of a change, 1 - P(NONE), per instance."""
    change, existence, picked, change_score = [], [], [], []
    offset = 0
    for group in groups:
        c, e, s = forward(model, heads, ptok, group, before_of(group), grad=False)
        change += c.argmax(1).tolist()
        change_score += (1 - torch.softmax(c, 1)[:, CHANGES.index("NONE")]).tolist()
        existence += e.argmax(1).tolist()
        s_rows = [r for r, _ in group["statements"]]
        owners = statements["instance"][s_rows]
        for inst in dict.fromkeys(owners.tolist()):
            rows = np.flatnonzero(owners == inst)
            picked.append(offset + rows[int(s[torch.tensor(rows, device=DEVICE)].argmax())])
        offset += len(s_rows)
    return np.array(change), np.array(existence), np.array(picked), np.array(change_score, dtype=np.float32)


def score(preds, groups, instances, statements):
    """Development score: mean of change macro-F1 and combined location accuracy."""
    change, existence, picked, _ = preds
    rows = [i for g in groups for i, _ in g["queries"]]
    gold_change = [CHANGES.index(instances[i]["change"]) for i in rows]
    s_rows = np.array([s for g in groups for s, _ in g["statements"]], dtype=int)
    picked_gold = statements["gold"][s_rows[picked]] if len(picked) else np.array([])
    known = EXISTENCE.index("known")
    correct, k = 0, 0
    for n, i in enumerate(rows):
        gold_exist = EXISTENCE.index(instances[i]["existence"])
        if gold_exist == known:
            correct += existence[n] == known and bool(picked_gold[k])
            k += 1
        else:
            correct += existence[n] == gold_exist
    location_acc = correct / len(rows)
    change_f1 = macro_f1(gold_change, change)
    return {"change_f1": change_f1, "location_acc": location_acc, "score": (change_f1 + location_acc) / 2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--dataset", required=True, choices=["propara", "babi"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate of the language model")
    parser.add_argument("--head-lr", type=float, default=1e-3, help="learning rate of the new linear heads")
    parser.add_argument("--accumulate", type=int, default=8, help="groups per optimizer step")
    parser.add_argument("--train-groups", type=int, default=None, help="only for testing the script: "
                        "use this many training groups")
    args = parser.parse_args()
    seed_everything(args.seed)

    data, groups, statements = build_plan(args.dataset)
    instances = data["instances"]
    by_step = {(g["doc"]["id"], g["step"]): g for g in groups}

    def before_of(group):
        return by_step[group["doc"]["id"], group["step"] - 1]

    by_split = {k: [g for g in groups if g["doc"]["split"] == k and not g["initial"]]
                for k in ("train", "dev", "test")}
    by_split["train"] = by_split["train"][:args.train_groups]
    ptok = PrefixTokenizer(AutoTokenizer.from_pretrained(MODELS[args.model]))
    model = AutoModel.from_pretrained(MODELS[args.model], dtype=torch.float32).to(DEVICE)
    heads = Heads(model.config.hidden_size).to(DEVICE)
    optimizer = bnb.optim.AdamW8bit([{"params": model.parameters(), "lr": args.lr},
                                     {"params": heads.parameters(), "lr": args.head_lr}], weight_decay=0.01)
    total_steps = math.ceil(len(by_split["train"]) / args.accumulate) * args.epochs
    warmup = max(1, int(0.06 * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / warmup, max(0.0, (total_steps - step) / (total_steps - warmup))))
    ce = nn.CrossEntropyLoss()

    history, best = [], None
    rng = random.Random(args.seed)
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = by_split["train"][:]
        rng.shuffle(order)
        running = 0.0
        for n, group in enumerate(order, start=1):
            change, existence, s_group, s_gold = group_tensors(group, instances, statements)
            c, e, s = forward(model, heads, ptok, group, before_of(group), grad=True)
            loss = ce(c, change) + ce(e, existence)
            if len(s):
                loss = loss + group_softmax_loss(s, s_group, s_gold)
            (loss / args.accumulate).backward()
            running += loss.item()
            if n % args.accumulate == 0 or n == len(order):
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(heads.parameters()), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if n % 500 == 0:
                print(f"epoch {epoch} group {n}/{len(order)} loss {running / 500:.3f} "
                      f"{time.time() - start:.0f}s", flush=True)
                running = 0.0
        model.eval()
        dev = score(predict(model, heads, ptok, by_split["dev"], before_of, statements), by_split["dev"],
                    instances, statements)
        history.append({"epoch": epoch, **dev})
        print(f"epoch {epoch} dev {dev}", flush=True)
        if best is None or dev["score"] > best[0]:
            best = (dev["score"], epoch, predict(model, heads, ptok, by_split["test"], before_of, statements))

    _, best_epoch, (change, existence, picked, change_score) = best
    out = RESULTS / "finetune" / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{args.model}-seed{args.seed}.npz", change=change, existence=existence, location=picked,
             change_score=change_score)
    save_json({"model": args.model, "seed": args.seed, "best_epoch": best_epoch, "history": history,
               "args": vars(args), "seconds": round(time.time() - start)},
              out / f"{args.model}-seed{args.seed}.json")


if __name__ == "__main__":
    main()
