# RL (Bandit + Pong)

Small reinforcement-learning project with two parts:

- **Multi-armed bandit** (compare a custom bandit against a reference epsilon-greedy baseline)
- **Multi-agent Pong (ma-gym)** (train/evaluate a simple Q-learning agent in `PongDuel-v0`)

## Repo layout

- `bandit/` — bandit implementations + simulator + pytest benchmark
- `pong/` — PongDuel runner + agents
- `recordings/` — optional video outputs when running Pong with `--record`
- `REPORT.tex` — LaTeX report

## Prerequisites

- Python 3.x
- (Optional) FFmpeg if you want video recording via `--record`

The repo already contains `.venv/` and `.venv-pong/` folders in this workspace; you can reuse them or create fresh virtual environments.

## Setup (Windows / PowerShell)

### Bandit environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r bandit\requirements.txt
```

### Pong environment

```powershell
python -m venv .venv-pong
.\.venv-pong\Scripts\Activate.ps1
pip install -r pong\requirements.txt
```

Notes:

- `pong/pong.py` imports `PIL` (Pillow). If you see `ModuleNotFoundError: No module named 'PIL'`, run: `pip install Pillow`.
- It also imports `gym`. If your environment doesn’t already have it through `ma-gym`, install it: `pip install gym`.

## Bandit (tests/benchmark)

The bandit evaluation is implemented as a pytest test (`bandit/test_bandits.py`). It simulates your bandit versus the reference bandit across multiple random seeds.

Run:

```powershell
.\.venv\Scripts\Activate.ps1
pytest -q bandit\test_bandits.py
```

Key files:

- `bandit/MyBandit.py` — your bandit implementation (`Bandit`)
- `bandit/ReferenceBandit.py` — baseline epsilon-greedy bandit
- `bandit/simulator.py` — reward generation + simulation loop

## Pong (run the simulator)

Run from the `pong/` folder so imports resolve cleanly:

```powershell
.\.venv-pong\Scripts\Activate.ps1
cd pong
python .\pong.py --episodes 200
```

Common options:

- Render a window (slow):

```powershell
python .\pong.py --episodes 50 --render
```

- Record videos to `recordings/` (requires ffmpeg):

```powershell
python .\pong.py --episodes 50 --record
```

- Save a GIF of the last episode (no ffmpeg required, but needs Pillow):

```powershell
python .\pong.py --episodes 50 --gif --gif-path ..\pong.gif
```

Agents:

- `pong/Agent.py` — Q-learning style agent (tabular Q with discretized observations)
- `pong/RandomAgent.py` — random baseline

## Report (LaTeX)

If you have a LaTeX distribution installed, you can compile the report:

```powershell
pdflatex REPORT.tex
```

## Troubleshooting

- **Gym/ma-gym install issues**: `ma-gym` is installed from GitHub (`git+https://...`). If install fails, make sure you have `git` available on PATH.
- **No window / render issues**: run without `--render` (headless). Rendering can be unstable on some setups.
