"""Write every implicit state change in the ProPara test split to a text file for manual reading.

The proposal takes the annotations as correct but promises to read a sample of them. Reading all
test-split implicit changes both estimates how noisy they are and gives a cleaned subset to
evaluate on. The manual verdicts are in results/implicit_test_verdicts.csv with one category each:
  genuine   - the entity is not referred to, and the change follows from the step
  reference - the entity is referred to by another expression (pronoun, ellipsis, synonym)
  doubtful  - the change does not follow from this step (wrong step, unsupported, or only the
              spelling of the location changed)
"""

from common import DATA, RESULTS, load_json

if __name__ == "__main__":
    data = load_json(DATA / "propara.json")
    docs = {d["id"]: d for d in data["docs"]}
    implicit_changes = [x for x in data["instances"]
                        if x["split"] == "test" and x["change"] != "NONE" and not x["mentioned"]]
    lines = []
    for x in implicit_changes:
        doc = docs[x["doc"]]
        entity = doc["entities"][x["entity"]]
        lines.append(f"{x['doc']},{x['step']},{x['entity']}  entity='{'; '.join(entity['alternatives'])}'  "
                     f"{x['change']}: {x['prev']} -> {x['loc']}")
        for i, sentence in enumerate(doc["sentences"], start=1):
            marker = ">>" if i == x["step"] else "  "
            lines.append(f"   {marker} {i}. {sentence}")
        lines.append("")
    path = RESULTS / "implicit_test_changes.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {len(implicit_changes)} implicit test changes to {path}")
