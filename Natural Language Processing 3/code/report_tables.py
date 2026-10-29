"""Fill the result tables of report.html from results/summary.json and the prepared data.

Each table sits between <!-- BEGIN:name --> and <!-- END:name --> in report.html and is replaced
on every run, so the report never holds a number that differs from the results.

Usage: python code/report_tables.py
"""

import re

from common import DATA, DATASETS, RESULTS, ROOT, SHORT, load_json

SYSTEMS = ["majority", "previous", "recent", "probe 160M", "probe 410M", "probe 1B", "probe 1.4B", "probe 2.8B",
           "probe 410M untrained", "fine-tuned 160M", "fine-tuned 410M"]
LABEL = {"majority": "Majority", "previous": "Previous location (gold)", "recent": "Most recent location",
         "probe 410M untrained": "Probe, 410M untrained", "fine-tuned 160M": "Fine-tuned 160M",
         "fine-tuned 410M": "Fine-tuned 410M"}
GROUP_START = {"probe 160M", "probe 410M untrained", "fine-tuned 160M"}  # rule above these rows


def cell(row, metric, ci=True):
    if row is None or metric not in row or row["n"] == 0 or row[metric] != row[metric]:
        return "&ndash;"
    value = f"{100 * row[metric]:.1f}"
    if not ci:
        return value
    lo, hi = row[f"{metric}_ci"]
    return f"{value} <span class=\"ci\">{100 * lo:.0f}&ndash;{100 * hi:.0f}</span>"


def label(name):
    return LABEL.get(name, name.replace("probe", "Probe"))


def rows_for(summary, columns):
    """columns: list of (dataset, subset, metric)."""
    out = []
    for name in SYSTEMS:
        if not any(name in summary[d]["scores"] for d, _, _ in columns):
            continue
        cls = ' class="sep"' if name in GROUP_START else ""
        cells = [cell(summary[d]["scores"].get(name, {}).get(s), m) for d, s, m in columns]
        out.append(f"<tr{cls}><td>{label(name)}</td>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    return "\n".join(out)


def table_data():
    rows = []
    for dataset, name in (("propara", "ProPara"), ("babi", "bAbI task 2")):
        data = load_json(DATA / f"{dataset}.json")
        for split in ("train", "dev", "test"):
            docs = [d for d in data["docs"] if d["split"] == split]
            ins = [x for x in data["instances"] if x["split"] == split]
            changed = [x for x in ins if x["change"] != "NONE"]
            cand = sum(len(d["candidates"]) for d in docs) / len(docs)
            cls = ' class="sep"' if split == "train" and dataset == "babi" else ""
            rows.append(f"<tr{cls}><td>{name if split == 'train' else ''}</td><td>{split}</td><td>{len(docs)}</td>"
                        f"<td>{len(ins):,}</td><td>{sum(not x['mentioned'] for x in ins):,}</td>"
                        f"<td>{len(changed):,}</td><td>{sum(not x['mentioned'] for x in changed):,}</td>"
                        f"<td>{cand:.1f}</td></tr>")
    return ("<table><thead><tr><th></th><th>split</th><th>documents</th><th>instances</th><th>implicit</th>"
            "<th>changes</th><th>implicit changes</th><th>candidates</th></tr></thead><tbody>\n"
            + "\n".join(rows) + "\n</tbody></table>")


def table_overall(summary):
    columns = [(d, "all", m) for d in DATASETS for m in ("location_acc", "change_f1", "change_auc")]
    head = ("<thead><tr><th></th><th colspan=\"3\">ProPara</th><th colspan=\"3\">bAbI task 2</th></tr>"
            "<tr><th></th>" + "<th>location acc.</th><th>change F1</th><th>change AUC</th>" * 2 + "</tr></thead>")
    return f"<table>{head}<tbody>\n{rows_for(summary, columns)}\n</tbody></table>"


def table_propara(summary):
    columns = [("propara", "changed-explicit", "location_acc"), ("propara", "changed-implicit", "location_acc"),
               ("propara", "changed-implicit genuine", "location_acc"),
               ("propara", "explicit", "change_auc"), ("propara", "implicit", "change_auc"),
               ("propara", "implicit, genuine changes", "change_auc")]
    n = {s: summary["propara"]["scores"]["previous"][s]["n"] for _, s, _ in columns}
    head = ("<thead><tr><th></th><th colspan=\"3\">location accuracy, changed instances</th>"
            "<th colspan=\"3\">change AUC</th></tr>"
            f"<tr><th></th><th>explicit ({n['changed-explicit']})</th><th>implicit ({n['changed-implicit']})</th>"
            f"<th>genuine ({n['changed-implicit genuine']})</th><th>explicit ({n['explicit']})</th>"
            f"<th>implicit ({n['implicit']})</th><th>genuine ({n['implicit, genuine changes']})</th></tr></thead>")
    return f"<table>{head}<tbody>\n{rows_for(summary, columns)}\n</tbody></table>"


def table_babi(summary):
    columns = [("babi", "changed-explicit", "location_acc"), ("babi", "changed-implicit", "location_acc"),
               ("babi", "carrier-moves moved", "location_acc"), ("babi", "carrier-moves stayed", "location_acc"),
               ("babi", "carrier-moves", "location_acc"), ("babi", "carrier-moves", "change_auc")]
    n = {s: summary["babi"]["scores"]["previous"][s]["n"] for _, s, _ in columns}
    head = ("<thead><tr><th></th><th colspan=\"2\">location acc., changed</th>"
            "<th colspan=\"3\">location acc., someone else moves</th><th></th></tr>"
            f"<tr><th></th><th>explicit ({n['changed-explicit']})</th><th>implicit ({n['changed-implicit']})</th>"
            f"<th>carried ({n['carrier-moves moved']})</th><th>not carried ({n['carrier-moves stayed']})</th>"
            f"<th>both ({n['carrier-moves']})</th><th>change AUC</th></tr></thead>")
    return f"<table>{head}<tbody>\n{rows_for(summary, columns)}\n</tbody></table>"


def table_controls(summary):
    rows = []
    info = summary["propara"]["probe_layers"]
    for model in SHORT:
        if model not in info:
            continue
        x = info[model]
        layers = x["layers"] - 1
        chosen = x["chosen"]
        sel = x.get("selectivity", {})
        curves = x["curves"]
        rows.append(
            f"<tr><td>{label('probe ' + SHORT[model])}</td>"
            f"<td>{chosen['change']}/{layers}</td><td>{100 * curves['change'][chosen['change']]:.1f}</td>"
            f"<td>{100 * curves['control_change'][chosen['change']]:.1f}</td><td>{100 * sel.get('change', float('nan')):+.1f}</td>"
            f"<td>{chosen['location']}/{layers}</td><td>{100 * curves['location'][chosen['location']]:.1f}</td>"
            f"<td>{100 * curves['control_location'][chosen['location']]:.1f}</td><td>{100 * sel.get('location', float('nan')):+.1f}</td></tr>")
    head = ("<thead><tr><th></th><th colspan=\"4\">state change (macro-F1)</th><th colspan=\"4\">location candidate (accuracy)</th></tr>"
            "<tr><th></th><th>layer</th><th>task</th><th>control</th><th>selectivity</th>"
            "<th>layer</th><th>task</th><th>control</th><th>selectivity</th></tr></thead>")
    return f"<table>{head}<tbody>\n" + "\n".join(rows) + "\n</tbody></table>"


def table_finetune(summary):
    """One row per fine-tuning run: the epoch kept, its development score, and test location
    accuracy of that run alone (the other tables average the seeds)."""
    rows = []
    for dataset, name in (("propara", "ProPara"), ("babi", "bAbI task 2")):
        scores = summary[dataset]["scores"]
        for m, (model, runs) in enumerate(summary[dataset]["finetune"].items()):
            row = scores[f"fine-tuned {SHORT[model]}"]
            per_seed = {s: row[s].get("location_acc_seeds", [row[s]["location_acc"]])
                        for s in ("all", "changed-implicit")}
            for k, r in enumerate(runs):
                first = k == 0
                new_dataset = first and m == 0
                cls = ' class="sep"' if new_dataset and rows else ""
                rows.append(f"<tr{cls}><td>{name if new_dataset else ''}</td><td>{SHORT[model] if first else ''}</td>"
                            f"<td>{r['seed']}</td><td>{r['best_epoch']} of {r['args']['epochs']}</td>"
                            f"<td>{100 * r['history'][r['best_epoch'] - 1]['score']:.1f}</td>"
                            f"<td>{100 * per_seed['all'][k]:.1f}</td><td>{100 * per_seed['changed-implicit'][k]:.1f}</td></tr>")
    head = ("<thead><tr><th></th><th>model</th><th>seed</th><th>epoch kept</th><th>dev score</th>"
            "<th>test location acc.</th><th>location acc., implicit changes</th></tr></thead>")
    return f"<table>{head}<tbody>\n" + "\n".join(rows) + "\n</tbody></table>"


if __name__ == "__main__":
    summary = load_json(RESULTS / "summary.json")
    tables = {"data": table_data(), "overall": table_overall(summary), "propara": table_propara(summary),
              "babi": table_babi(summary), "controls": table_controls(summary), "finetune": table_finetune(summary)}
    path = ROOT / "report.html"
    html = path.read_text(encoding="utf-8")
    for name, table in tables.items():
        pattern = re.compile(rf"(<!-- BEGIN:{name} -->).*?(<!-- END:{name} -->)", re.S)
        assert pattern.search(html), f"no markers for table {name}"
        html = pattern.sub(lambda m: f"{m.group(1)}\n{table}\n{m.group(2)}", html)
    path.write_text(html, encoding="utf-8")
    print("tables written:", ", ".join(tables))
