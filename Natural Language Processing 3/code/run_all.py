"""Run the whole study in order; steps whose output already exists are skipped.

  1. prepare the datasets                       (data.py)
  2. extract hidden states and train the probes (extract.py, probe.py): ProPara with every Pythia
     size and the untrained 410M network, bAbI with 160M to 1B
  3. fine-tune: Pythia-160M and 410M on ProPara, 160M on bAbI, two seeds each (finetune.py)
  4. compute all tables and figures             (evaluate.py)
  5. write the tables into report.html          (report_tables.py)
  6. compile Final_Report.pdf with page numbers (build_report.py)

Logs go to work/logs/.
"""

import subprocess
import sys

from common import DATA, DATASETS, FEATURES, RESULTS, ROOT

PROBED = {"propara": ["pythia-160m", "pythia-410m", "pythia-1b", "pythia-1.4b", "pythia-2.8b", "pythia-410m-step0"],
          "babi": ["pythia-160m", "pythia-410m", "pythia-1b"]}
FINETUNED = {"propara": ["pythia-160m", "pythia-410m"], "babi": ["pythia-160m"]}
SEEDS = [0, 1]
# An epoch on bAbI takes about twice as long as on ProPara, and bAbI is synthetic.
EPOCHS = {"propara": 5, "babi": 3}
LOGS = ROOT / "work" / "logs"


def run(log_name, *args):
    print(">>", " ".join(args), flush=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    code_dir = Path(__file__).resolve().parent
    with open(LOGS / f"{log_name}.log", "w", encoding="utf-8") as log:
        subprocess.run([sys.executable, "-u", *args], check=True, stdout=log, stderr=subprocess.STDOUT, cwd=code_dir)


if __name__ == "__main__":
    if not all((DATA / f"{d}.json").exists() for d in DATASETS):
        run("data", "data.py")
    for dataset in DATASETS:
        for model in PROBED[dataset]:
            if not (FEATURES / dataset / model / "meta.json").exists():
                run(f"extract-{dataset}-{model}", "extract.py", "--model", model, "--dataset", dataset)
            if not (RESULTS / "probe" / dataset / f"{model}.json").exists():
                run(f"probe-{dataset}-{model}", "probe.py", "--model", model, "--dataset", dataset)
    for dataset in DATASETS:
        for model in FINETUNED[dataset]:
            for seed in SEEDS:
                if not (RESULTS / "finetune" / dataset / f"{model}-seed{seed}.json").exists():
                    run(f"finetune-{dataset}-{model}-seed{seed}", "finetune.py", "--model", model,
                        "--dataset", dataset, "--seed", str(seed), "--epochs", str(EPOCHS[dataset]))
    run("evaluate", "evaluate.py")
    run("report_tables", "report_tables.py")
    run("build_report", "build_report.py")
