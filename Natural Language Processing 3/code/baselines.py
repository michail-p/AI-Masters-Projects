"""The three baselines from the proposal, as predictions for every instance.

  majority:  the most frequent training label. For location, the most frequent training label that
             is possible for the instance ('-', '?' or one of its candidates).
  previous:  nothing changes: the entity stays where it was before the step (change = NONE).
             It gets the gold previous location, so it is a strong baseline everywhere except where
             the state changes, where it is wrong by definition.
  recent:    the location mentioned most recently in the text so far, '?' if none was mentioned.
             It only predicts location.
"""

from collections import Counter

from common import UNKNOWN, existence_of
from data import mention_positions


def most_recent_location(doc, step):
    """The candidate mentioned last in sentences 1..step (latest sentence, then latest word)."""
    for sentence in reversed(doc["sentences"][:step]):
        best, best_pos = None, -1
        for candidate in doc["candidates"]:
            positions = mention_positions([candidate], sentence)
            if positions and positions[-1] > best_pos:
                best, best_pos = candidate, positions[-1]
        if best is not None:
            return best
    return UNKNOWN


def baseline_predictions(data):
    """{baseline: {"change": [...], "location": [...], "existence": [...]}} over all instances;
    a task a baseline does not predict is left out."""
    docs = {d["id"]: d for d in data["docs"]}
    instances = data["instances"]
    train = [x for x in instances if x["split"] == "train"]
    change_majority = Counter(x["change"] for x in train).most_common(1)[0][0]
    location_ranking = [loc for loc, _ in Counter(x["loc"] for x in train).most_common()]

    def majority_location(x):
        allowed = set(docs[x["doc"]]["candidates"])
        return next(loc for loc in location_ranking if existence_of(loc) != "known" or loc in allowed)

    majority = [majority_location(x) for x in instances]
    recent = [most_recent_location(docs[x["doc"]], x["step"]) for x in instances]
    return {
        "majority": {"change": [change_majority] * len(instances), "location": majority,
                     "existence": [existence_of(loc) for loc in majority]},
        "previous": {"change": ["NONE"] * len(instances), "location": [x["prev"] for x in instances],
                     "existence": [existence_of(x["prev"]) for x in instances]},
        "recent": {"location": recent, "existence": [existence_of(loc) for loc in recent]},
    }
