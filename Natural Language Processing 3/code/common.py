"""Shared paths, model list and small helpers used by every script."""

import json
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
DATA = ROOT / "data" / "prepared"      # one JSON file per dataset (documents + instances)
FEATURES = ROOT / "work" / "features"  # hidden states, large, not submitted
RESULTS = ROOT / "results"             # predictions, metrics, figures

DATASETS = ["propara", "babi"]

# The Pythia suite: same data, same order, only the size differs (Biderman et al., 2023).
MODELS = {
    "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-1b": "EleutherAI/pythia-1b",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
    "pythia-2.8b": "EleutherAI/pythia-2.8b",
}
# The same 410M network before any training (Pythia's "step0" checkpoint). Probes on it show what
# a linear probe can get from the input tokens alone, without anything learned in pretraining.
UNTRAINED = {"pythia-410m-step0": ("EleutherAI/pythia-410m", "step0")}


def model_source(name):
    """(Hugging Face repository, revision) of a model name used in this project."""
    return UNTRAINED[name] if name in UNTRAINED else (MODELS[name], None)


# Approximate parameter counts, used as the x axis of the size plots.
MODEL_PARAMS = {"pythia-160m": 160e6, "pythia-410m": 410e6, "pythia-1b": 1.0e9,
                "pythia-1.4b": 1.4e9, "pythia-2.8b": 2.8e9}
# Model names as they appear in tables and figures.
SHORT = {"pythia-160m": "160M", "pythia-410m": "410M", "pythia-1b": "1B", "pythia-1.4b": "1.4B",
         "pythia-2.8b": "2.8B", "pythia-410m-step0": "410M untrained"}

# Labels. '-' = the entity does not exist, '?' = it exists but its location is unknown.
NONEXISTENT, UNKNOWN = "-", "?"
CHANGES = ["NONE", "CREATE", "MOVE", "DESTROY"]
EXISTENCE = ["nonexistent", "unknown", "known"]  # coarse location type, see existence_of()


def existence_of(location):
    """Map a location label to its coarse type: does not exist / unknown / a known place."""
    if location == NONEXISTENT:
        return "nonexistent"
    if location == UNKNOWN:
        return "unknown"
    return "known"


def change_of(before, after):
    """State change between two consecutive locations, as defined for ProPara (Dalvi et al., 2018)."""
    if before == NONEXISTENT and after != NONEXISTENT:
        return "CREATE"
    if before != NONEXISTENT and after == NONEXISTENT:
        return "DESTROY"
    if before != NONEXISTENT and after != NONEXISTENT and before != after:
        return "MOVE"
    return "NONE"


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
