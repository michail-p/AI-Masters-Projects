"""Turn ProPara and bAbI task 2 into one common format.

A *document* is a list of sentences, a list of entities and, for every entity, its location
before the first sentence and after each sentence (the "grid").
An *instance* is one entity after one step (sentence) and is the unit that is classified:
  - change:    CREATE / MOVE / DESTROY / NONE, derived from the location before and after the step
  - location:  the location after the step: '-' (does not exist), '?' (unknown) or one of the
               document's candidate locations
  - mentioned: whether the entity is named in the sentence of that step. Instances where it is not
               are the "implicit" ones the study is about.

Run `python code/data.py` to write data/prepared/{propara,babi}.json and print statistics.
"""

import json
import re
import tarfile
from collections import Counter

from common import DATA, NONEXISTENT, RAW, UNKNOWN, change_of, existence_of, save_json

# ---------------------------------------------------------------------------------------------
# Matching entity and location names against a sentence
# ---------------------------------------------------------------------------------------------

# Words that can start a ProPara location phrase without needing "in the" in front of it.
PREPOSITIONAL = {"in", "on", "at", "inside", "outside", "under", "underground", "above", "below",
                 "into", "near", "deep", "through", "within", "around", "over", "between",
                 "behind", "back", "up", "down", "out", "off", "away", "everywhere", "elsewhere",
                 "nowhere", "here", "there", "beneath", "underneath", "onto", "along", "across",
                 "toward", "towards", "upward", "downward", "surface"}


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def word_forms(word):
    """A small set of singular/plural variants, enough to match 'leaves' with 'leaf'."""
    forms = {word}
    if len(word) > 3:
        if word.endswith("ies"):
            forms.add(word[:-3] + "y")
        if word.endswith("ves"):
            forms.update({word[:-3] + "f", word[:-3] + "fe"})
        if word.endswith("es"):
            forms.add(word[:-2])
        if word.endswith("s") and not word.endswith("ss"):
            forms.add(word[:-1])
    return forms


def head_word(phrase):
    """Head noun of a short noun phrase: the word before 'of' ('tiny parts of rocks' -> 'parts'),
    otherwise the last word ('new rock' -> 'rock')."""
    ws = words(phrase)
    if not ws:
        return ""
    if "of" in ws[1:]:
        return ws[ws.index("of", 1) - 1]
    return ws[-1]


def mention_positions(phrase_alternatives, sentence):
    """Word positions in `sentence` where any alternative of a name is mentioned, matched on its head
    noun with plural/singular variants. ProPara names can list alternatives: 'water; droplets'."""
    sent_words = words(sentence)
    positions = []
    for alt in phrase_alternatives:
        head = head_word(alt)
        if not head:
            continue
        head_forms = word_forms(head)
        positions += [i for i, w in enumerate(sent_words) if word_forms(w) & head_forms]
    return sorted(set(positions))


def is_mentioned(alternatives, sentence):
    return bool(mention_positions(alternatives, sentence))


# ---------------------------------------------------------------------------------------------
# Natural-language templates that the language models read
# ---------------------------------------------------------------------------------------------

def location_phrase(location):
    """'soil' -> 'in the soil', 'underground' -> 'underground', 'the ocean' -> 'in the ocean'."""
    loc = location.strip()
    if loc.lower().startswith("the "):
        loc = loc[4:]
    first = words(loc)[0] if words(loc) else ""
    return loc if first in PREPOSITIONAL else f"in the {loc}"


def query_text(entity):
    """Text appended after the steps read so far; its last token is read for state change and
    existence. It is also the beginning of every location statement below."""
    return f" The {entity} is"


def statement_text(entity, location):
    """A candidate statement; the last token ('.') is read to score whether it is true."""
    return f" The {entity} is {location_phrase(location)}."


# ---------------------------------------------------------------------------------------------
# ProPara (Dalvi et al., 2018)
# ---------------------------------------------------------------------------------------------

def load_propara():
    docs = []
    for split in ["train", "dev", "test"]:
        with open(RAW / "propara" / f"grids.v1.{split}.json", encoding="utf-8") as f:
            for line in f:
                raw = json.loads(line)
                entities = []
                for name in raw["participants"]:
                    alternatives = [a.strip() for a in name.split(";") if a.strip()]
                    entities.append({"name": alternatives[0], "alternatives": alternatives})
                # Locations are lower-cased annotator spans; '-' and '?' are kept as they are.
                grid = [[s.strip().lower() if s.strip() not in (NONEXISTENT, UNKNOWN) else s.strip()
                         for s in row] for row in raw["states"]]
                assert all(len(row) == len(raw["sentence_texts"]) + 1 for row in grid)
                docs.append({
                    "id": f"propara-{raw['para_id']}",
                    "split": split,
                    "sentences": [s.strip() for s in raw["sentence_texts"]],
                    "entities": entities,
                    "grid": grid,
                })
    return docs


# ---------------------------------------------------------------------------------------------
# bAbI task 2, "two supporting facts" (Weston et al., 2016)
# ---------------------------------------------------------------------------------------------

BABI_TAR = RAW / "babi" / "tasks_1-20_v1-2.tar.gz"
BABI_FILES = {"train": "tasks_1-20_v1-2/en/qa2_two-supporting-facts_train.txt",
              "test": "tasks_1-20_v1-2/en/qa2_two-supporting-facts_test.txt"}
BABI_DEV_STORIES = 20  # the last 20 training stories are held out as the development set

MOVE_RE = re.compile(r"^(\w+) (?:moved|went|went back|journeyed|travelled) to the (\w+)\.$")
TAKE_RE = re.compile(r"^(\w+) (?:got|grabbed|picked up|took) the (\w+) there\.$")
DROP_RE = re.compile(r"^(\w+) (?:dropped|discarded|put down|left) the (\w+)(?: there)?\.$")


def read_babi_stories(split):
    """Each story is a list of sentences plus the questions asked after some of them."""
    with tarfile.open(BABI_TAR) as tar:
        lines = tar.extractfile(BABI_FILES[split]).read().decode("utf-8").splitlines()
    stories = []
    for line in lines:
        number, text = line.split(" ", 1)
        if number == "1":
            stories.append({"sentences": [], "questions": []})
        story = stories[-1]
        if "\t" in text:  # a question line: 'Where is the milk? \tkitchen\t3 6'
            question, answer, _support = text.split("\t")
            obj = re.match(r"Where is the (\w+)\?", question.strip()).group(1)
            story["questions"].append({"after_step": len(story["sentences"]), "object": obj,
                                       "answer": answer})
        else:
            story["sentences"].append(text.strip())
    return stories


def simulate_babi(sentences):
    """Track who carries what and where everyone is, and return each object's location after
    every step. An object is at the location of whoever carries it, or where it was dropped;
    before anybody has taken it, its location is unknown."""
    objects = []
    for s in sentences:
        m = TAKE_RE.match(s) or DROP_RE.match(s)
        if m and m.group(2) not in objects:
            objects.append(m.group(2))
    person_location, holder = {}, {}
    location = {o: UNKNOWN for o in objects}
    grid = {o: [UNKNOWN] for o in objects}
    for s in sentences:
        if m := MOVE_RE.match(s):
            person, place = m.groups()
            person_location[person] = place
            for o in objects:
                if holder.get(o) == person:
                    location[o] = place  # the implicit change: the object is not mentioned
        elif m := TAKE_RE.match(s):
            person, o = m.groups()
            holder[o] = person
            location[o] = person_location.get(person, UNKNOWN)
        elif m := DROP_RE.match(s):
            person, o = m.groups()
            holder[o] = None
            location[o] = person_location.get(person, UNKNOWN)
        else:
            raise ValueError(f"unparsed bAbI sentence: {s}")
        for o in objects:
            grid[o].append(location[o])
    return objects, grid


def load_babi():
    docs = []
    for split in ["train", "test"]:
        stories = read_babi_stories(split)
        for i, story in enumerate(stories):
            objects, grid = simulate_babi(story["sentences"])
            # The simulation must agree with every answer in the dataset.
            for q in story["questions"]:
                assert grid[q["object"]][q["after_step"]] == q["answer"], (split, i, q)
            doc_split = "dev" if split == "train" and i >= len(stories) - BABI_DEV_STORIES else split
            docs.append({
                "id": f"babi-{split}-{i}",
                "split": doc_split,
                "sentences": story["sentences"],
                "entities": [{"name": o, "alternatives": [o]} for o in objects],
                "grid": [grid[o] for o in objects],
                # Every place someone goes to, not only the places objects end up in.
                "candidates": sorted({m.group(2) for s in story["sentences"] if (m := MOVE_RE.match(s))}),
            })
    return docs


# ---------------------------------------------------------------------------------------------
# Instances
# ---------------------------------------------------------------------------------------------

def candidates_of(doc):
    """Candidate locations: every known location annotated anywhere in the document."""
    return sorted({loc for row in doc["grid"] for loc in row if loc not in (NONEXISTENT, UNKNOWN)})


def make_instances(doc):
    """One instance per entity per step, ordered by step and then entity. Feature extraction
    relies on this order."""
    instances = []
    for step, sentence in enumerate(doc["sentences"], start=1):
        for e, entity in enumerate(doc["entities"]):
            before, after = doc["grid"][e][step - 1], doc["grid"][e][step]
            instances.append({
                "doc": doc["id"],
                "split": doc["split"],
                "step": step,
                "entity": e,
                "prev": before,
                "loc": after,
                "existence": existence_of(after),
                "change": change_of(before, after),
                "mentioned": is_mentioned(entity["alternatives"], sentence),
            })
    return instances


def prepare(dataset):
    docs = load_propara() if dataset == "propara" else load_babi()
    for doc in docs:
        doc.setdefault("candidates", candidates_of(doc))
        assert set(candidates_of(doc)) <= set(doc["candidates"])
    instances = [inst for doc in docs for inst in make_instances(doc)]
    save_json({"docs": docs, "instances": instances}, DATA / f"{dataset}.json")
    return docs, instances


def print_stats(dataset, docs, instances):
    print(f"\n=== {dataset}")
    for split in ["train", "dev", "test"]:
        d = [x for x in docs if x["split"] == split]
        ins = [x for x in instances if x["split"] == split]
        changed = [x for x in ins if x["change"] != "NONE"]
        implicit_changed = [x for x in changed if not x["mentioned"]]
        n_cand = sum(len(x["candidates"]) for x in d) / max(len(d), 1)
        print(f"{split:5s}: {len(d):4d} docs, {len(ins):6d} instances, "
              f"{sum(not x['mentioned'] for x in ins):6d} implicit, {len(changed):5d} changes, "
              f"{len(implicit_changed):5d} implicit changes, {n_cand:.1f} candidates/doc")
    print("change x mentioned:", sorted(Counter((x["change"], x["mentioned"]) for x in instances).items()))
    print("existence:", Counter(x["existence"] for x in instances))


if __name__ == "__main__":
    for name in ["propara", "babi"]:
        print_stats(name, *prepare(name))
