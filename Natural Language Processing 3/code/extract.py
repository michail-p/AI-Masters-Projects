"""Extract frozen hidden states from the Pythia models for every instance.

For every document and step t, the model reads the first t sentences (the *prefix*). After the
prefix it reads short *continuations*, each in its own batch row:
  - a query for each entity:            " The water is"
  - for each entity whose location is known, one statement per candidate location:
                                        " The water is in the soil."
The hidden state of the last token of each continuation is saved at every layer. The query
state feeds the existence probe and, together with the query state one step earlier, the
state-change probe; the statement states feed the location probe, which scores each candidate
statement and picks the highest. Each entity also gets a query after an empty text (step 0), its
state before the first step.

The prefix is run once per step and its key/value cache is shared by all continuations of that
step, which gives the same result as running every continuation with the full prefix (checked
by `--check`) at a fraction of the cost.

All computation is in float32. Pythia's checkpoints are stored in float16, and running the models
in float16 or bfloat16 changes the middle-layer states by 5-30% (relative norm, measured on
pythia-160m), which would add noise to the probes. The weights stay in float16 on the GPU (so
2.8B fits in 8 GB) and are cast to float32 layer by layer, which is exact because float16
values are representable in float32.

Usage: python code/extract.py --model pythia-410m --dataset propara [--check]
Output: work/features/{dataset}/{model}/{queries,initial,statements}.npy with shape
        [layers+1, N, dim], float16, layer-major so that one layer can be loaded at a time, and
        index.npz (see build_plan).
"""

import argparse
import copy
import time
import types

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer

from common import DATA, FEATURES, MODELS, UNTRAINED, load_json, model_source, save_json
from data import query_text, statement_text


def build_plan(dataset):
    """Group the instances by (document, step) and list the continuations to read for each group.

    Returns the prepared data, the groups, and an index with
      - the location statements: for each, its instance, candidate index and whether the candidate
        is the gold location
      - "previous": for each instance, the row of the query state one step earlier, counted in
        the query states followed by the initial states. The state-change probe reads both, since
        a change is a relation between the state before and after a step.
    Each document also gets a step-0 group whose queries follow an empty text (the initial
    states); its query rows count the initial states.
    """
    data = load_json(DATA / f"{dataset}.json")
    instances = data["instances"]
    groups = []
    statements = {"instance": [], "candidate": [], "gold": []}
    initial_of = {}  # (doc id, entity) -> row among the initial states
    for doc in data["docs"]:
        queries = []
        for e, entity in enumerate(doc["entities"]):
            initial_of[doc["id"], e] = len(initial_of)
            queries.append((initial_of[doc["id"], e], query_text(entity["name"].lower())))
        groups.append({"doc": doc, "step": 0, "queries": queries, "statements": [], "initial": True})
    previous = []
    i = 0
    for doc in data["docs"]:
        n_entities = len(doc["entities"])
        for step in range(1, len(doc["sentences"]) + 1):
            group = {"doc": doc, "step": step, "queries": [], "statements": [], "initial": False}
            for e, entity in enumerate(doc["entities"]):
                inst = instances[i]
                assert (inst["doc"], inst["step"], inst["entity"]) == (doc["id"], step, e)
                previous.append(i - n_entities if step > 1 else len(instances) + initial_of[doc["id"], e])
                name = entity["name"].lower()
                group["queries"].append((i, query_text(name)))
                # Statements are only needed where the gold location is a known place: when it is
                # '-' or '?', the existence probe alone decides whether a prediction is right.
                if inst["existence"] == "known":
                    for c, candidate in enumerate(doc["candidates"]):
                        group["statements"].append((len(statements["instance"]), statement_text(name, candidate)))
                        statements["instance"].append(i)
                        statements["candidate"].append(c)
                        statements["gold"].append(candidate == inst["loc"])
                i += 1
            groups.append(group)
    assert i == len(instances)
    index = {k: np.array(v) for k, v in statements.items()}
    index["previous"] = np.array(previous)
    index["n_initial"] = np.array(len(initial_of))
    return data, groups, index


class PrefixTokenizer:
    """Token ids of a document prefix: an end-of-text token (Pythia's document separator during
    pretraining) followed by the sentences, each but the first with a leading space."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.cache = {}

    def sentence_ids(self, doc):
        if doc["id"] not in self.cache:
            self.cache[doc["id"]] = [self.tok(s if i == 0 else " " + s)["input_ids"]
                                     for i, s in enumerate(doc["sentences"])]
        return self.cache[doc["id"]]

    def prefix(self, doc, step):
        ids = [self.tok.eos_token_id]
        for sentence in self.sentence_ids(doc)[:step]:
            ids += sentence
        return ids

    def continuations(self, texts, device):
        """Right-padded batch of continuation ids and the position of each one's last token.
        Right padding needs no attention mask: with causal attention, real tokens never attend to
        the padding that follows them."""
        ids = [self.tok(t)["input_ids"] for t in texts]
        width = max(len(x) for x in ids)
        batch = torch.full((len(ids), width), self.tok.eos_token_id, dtype=torch.long)
        for r, x in enumerate(ids):
            batch[r, :len(x)] = torch.tensor(x)
        last = torch.tensor([len(x) - 1 for x in ids])
        return batch.to(device), last.to(device)


def _linear_fp32(self, x):
    return F.linear(x.float(), self.weight.float(), None if self.bias is None else self.bias.float())


def _layer_norm_fp32(self, x):
    return F.layer_norm(x.float(), self.normalized_shape, self.weight.float(), self.bias.float(), self.eps)


def _embedding_fp32(self, ids):
    return F.embedding(ids, self.weight).float()


def load_frozen_model(name, device, revision=None):
    """A Pythia model with float16 weights that computes in float32 (see the module docstring)."""
    model = AutoModel.from_pretrained(name, revision=revision, dtype=torch.float16).to(device).eval()
    patches = {nn.Linear: _linear_fp32, nn.LayerNorm: _layer_norm_fp32, nn.Embedding: _embedding_fp32}
    for module in model.modules():
        for cls, forward in patches.items():
            if type(module) is cls:
                module.forward = types.MethodType(forward, module)
    return model


def read_group(model, ptok, group, device, grad=False, max_batch=None):
    """Run one (document, step) group. Returns the last-token states of all continuations
    (queries first, then statements) as a tuple over layers of [batch, dim] tensors.
    With `max_batch`, the continuations are run in chunks, each with its own copy of the prefix
    cache, to bound memory."""
    texts = [t for _, t in group["queries"]] + [t for _, t in group["statements"]]
    prefix = torch.tensor([ptok.prefix(group["doc"], group["step"])], device=device)
    chunk = max_batch or len(texts)
    outputs = []
    with torch.set_grad_enabled(grad):
        prefix_cache = model(input_ids=prefix, use_cache=True).past_key_values
        for start in range(0, len(texts), chunk):
            part = texts[start:start + chunk]
            conts, last = ptok.continuations(part, device)
            cache = prefix_cache if chunk >= len(texts) else copy.deepcopy(prefix_cache)
            cache.batch_repeat_interleave(len(part))
            out = model(input_ids=conts, past_key_values=cache, output_hidden_states=True, use_cache=False)
            rows = torch.arange(len(part), device=device)
            outputs.append([h[rows, last] for h in out.hidden_states])
    return tuple(torch.cat(layer) for layer in zip(*outputs))


def check_against_full_forward(model, reference, ptok, groups, device, n_groups=5):
    """The cached, chunked computation must match a plain float32 model reading prefix +
    continuation as one sequence."""
    worst = 0.0
    for group in [g for g in groups if g["statements"] and g["step"] > 1][:n_groups]:
        cached = read_group(model, ptok, group, device, max_batch=2)
        texts = [t for _, t in group["queries"]] + [t for _, t in group["statements"]]
        for r, text in enumerate(texts):
            ids = ptok.prefix(group["doc"], group["step"]) + ptok.tok(text)["input_ids"]
            with torch.no_grad():
                full = reference(input_ids=torch.tensor([ids], device=device), output_hidden_states=True)
            for layer, h in enumerate(full.hidden_states):
                a, b = cached[layer][r], h[0, -1]
                worst = max(worst, ((a - b).norm() / b.norm()).item())
    print(f"largest relative difference between cached and full forward: {worst:.2e}")
    return worst


class RowWriter:
    """Writes the rows of a [layers, rows, dim] array in order, a block at a time. Writing each
    group's few rows straight into the memory-mapped file touches a small region of every layer;
    once the file outgrows the page cache that thrashes (Pythia-1B on bAbI ran 2.4 times slower).
    A block of rows is one contiguous write per layer."""

    def __init__(self, array, block_bytes=256 << 20):
        self.array = array
        self.block = max(1, block_bytes // (array.shape[0] * array.shape[2] * array.itemsize))
        self.start, self.pending, self.n_pending = 0, [], 0

    def add(self, first_row, states):
        """states: [layers, rows, dim], the rows that start at `first_row`."""
        assert first_row == self.start + self.n_pending, "rows must be added in order"
        self.pending.append(states)
        self.n_pending += states.shape[1]
        if self.n_pending >= self.block:
            self.flush()

    def flush(self):
        if self.pending:
            self.array[:, self.start:self.start + self.n_pending] = np.concatenate(self.pending, 1)
            self.array.flush()
            self.start += self.n_pending
            self.pending, self.n_pending = [], 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS) + list(UNTRAINED))
    parser.add_argument("--dataset", required=True, choices=["propara", "babi"])
    parser.add_argument("--max-batch", type=int, default=8,
                        help="continuations per forward pass; all of a step at once is about 20%% faster but needs "
                             "more than 8 GB on the longest bAbI stories with 1B and on ProPara with 2.8B")
    parser.add_argument("--check", action="store_true", help="only compare against full forward passes")
    args = parser.parse_args()

    device = "cuda"
    repo, revision = model_source(args.model)
    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = load_frozen_model(repo, device, revision)
    ptok = PrefixTokenizer(tokenizer)
    data, groups, index = build_plan(args.dataset)

    if args.check:
        reference = AutoModel.from_pretrained(repo, revision=revision, dtype=torch.float32).to(device).eval()
        # float32 rounding through the layers gives ~1e-4; float16 would give ~5e-2.
        assert check_against_full_forward(model, reference, ptok, groups, device) < 1e-3
        return

    n_layers = model.config.num_hidden_layers + 1  # the embedding output plus every block
    dim = model.config.hidden_size
    out_dir = FEATURES / args.dataset / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    # States are computed in float32 and stored in float16 to halve the disk space; that rounds
    # each value once (relative error ~1e-3) and does not accumulate through the layers.
    def memmap(name, rows):
        return np.lib.format.open_memmap(out_dir / name, mode="w+", dtype=np.float16, shape=(n_layers, rows, dim))

    queries = RowWriter(memmap("queries.npy", len(data["instances"])))
    initial = RowWriter(memmap("initial.npy", int(index["n_initial"])))
    stmts = RowWriter(memmap("statements.npy", len(index["instance"])))
    start = time.time()
    for g, group in enumerate(groups):
        states = torch.stack(read_group(model, ptok, group, device, max_batch=args.max_batch))
        states = states.cpu().numpy().astype(np.float16)  # [layers, batch, dim]
        assert np.isfinite(states).all()
        nq = len(group["queries"])
        # Each group's query and statement rows are consecutive and follow the previous group's.
        (initial if group["initial"] else queries).add(group["queries"][0][0], states[:, :nq])
        if group["statements"]:
            stmts.add(group["statements"][0][0], states[:, nq:])
        if g % 500 == 0:
            print(f"{g}/{len(groups)} groups, {time.time() - start:.0f}s", flush=True)
    for writer in (queries, initial, stmts):
        writer.flush()
    np.savez(out_dir / "index.npz", **index)
    save_json({"model": repo, "revision": revision, "layers": n_layers, "dim": dim,
               "instances": len(data["instances"]), "statements": len(index["instance"]),
               "seconds": round(time.time() - start)}, out_dir / "meta.json")
    print(f"done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
