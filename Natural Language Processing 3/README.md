# Natural Language Processing 3

Final Project for the **Natural Language Processing course (5DV218 Språkteknologi)** at Umeå University.

## Implicit State Changes in Procedural Text: Probing vs Fine-Tuning

This research project investigates whether autoregressive language models implicitly track entity states in procedural text (e.g., tracking existence and locations of entities in scientific processes and multi-step narratives). It systematically compares linear probing of frozen layer representations against end-to-end causal fine-tuning across EleutherAI Pythia model scales (160M, 410M, 1B, 1.4B, and 2.8B).

## What This Project Covers

- **Benchmark Datasets:**
  - **ProPara:** Procedural text tracking physical state changes (creation, movement, destruction) of participants across science processes.
  - **bAbI (Task 2):** Two-supporting-facts reasoning tracking entity locations across sentence steps.
- **Representation Probing:**
  - Extracting hidden state activations across all transformer layers.
  - Training linear probes to test whether intermediate representations encode implicit entity state transitions.
- **Causal Fine-Tuning:**
  - Supervised sequence-to-sequence fine-tuning on procedural state changes across multiple model sizes and random seeds.
  - Comparing implicit state representation recovery between frozen and fine-tuned models.
- **Statistical Evaluation & Automated Reporting:**
  - Bootstrap confidence intervals for test metrics.
  - Automated HTML and PDF report compilation pipeline.

## Setup

Python 3.11 and an NVIDIA GPU with 8 GB of memory (developed on an RTX 4070 laptop GPU, Windows 11).

```bat
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

The raw data is included in `data/raw/`: the ProPara grids (`grids.v1.*.json` from github.com/allenai/propara) and the bAbI v1.2 archive. The Pythia models are downloaded from the Hugging Face hub on first use (about 13 GB).

## Running

From `code/`:

```bat
..\.venv\Scripts\python run_all.py
```

runs every step in order, skips steps whose output exists (allowing stopping and restarting), and ends by writing the result tables into `report.html`.

The PDF report is built from `report.html`:

```bat
..\.venv\Scripts\python build_report.py
```

which renders the HTML with headless Chrome, inserts page numbers, and sets publication metadata. You can also print directly from PowerShell:

```powershell
& "C:\Program Files\Google\Chrome\Application\chrome.exe" --headless --no-pdf-header-footer --print-to-pdf="$PWD\Final_Report.pdf" "$PWD\report.html"
```

## Outputs

- `data/prepared/` – both datasets in unified format (documents and instances)
- `results/probe/`, `results/finetune/` – test predictions and development/test scores per model and seed
- `results/summary.json`, `results/tables.md` – statistical metrics with bootstrap intervals
- `results/figures/` – generated report figures
- `results/implicit_test_changes.txt`, `results/implicit_test_verdicts.csv` – manual verdicts on implicit test changes

## Files

- `code/` - Implementation scripts:
  - `data.py` - Unified dataset loading and formatting
  - `extract.py` - Hidden state extraction across transformer layers
  - `probe.py` - Linear probing experiments
  - `finetune.py` - Causal fine-tuning training loop
  - `evaluate.py` - Multi-metric evaluation and testing
  - `report_tables.py` - Aggregates results into tables
  - `run_all.py` - Orchestration script executing the entire experimental pipeline
  - `build_report.py` - Headless Chrome PDF report compiler
- `Final_Report.pdf` - Complete publication-style final project report
- `Project_Proposal.pdf` - Initial project research proposal
- `requirements.txt` - Dependency specification

## Tech Stack

- Python 3.11
- PyTorch
- Hugging Face Transformers & Datasets
- EleutherAI Pythia LLMs (160M to 2.8B parameters)
- scikit-learn
- matplotlib, NumPy

## Notes

This is course assignment work intended for academic use.
