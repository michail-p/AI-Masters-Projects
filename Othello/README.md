# Othello Game Engine

**Course:** Artificial Intelligence (5DV243)  
**Author:** Michail Pettas

---

## 📋 Overview

A competitive Othello (Reversi) game engine that uses **Alpha-Beta pruning** with **Iterative Deepening Search (IDS)** to select optimal moves within time constraints. The engine is designed to outperform a naive fixed-depth implementation.

---

## 🗂️ Project Structure

```
Othello/
├── Introduction.txt        # Assignment specifications
├── report.tex              # LaTeX report
├── Python/
│   ├── Othello.py          # Main entry point
│   ├── AlphaBeta.py        # Alpha-Beta search implementation
│   ├── OthelloPosition.py  # Game state representation
│   ├── OthelloAction.py    # Move representation
│   ├── OthelloAlgorithm.py # Algorithm interface
│   ├── OthelloEvaluator.py # Evaluator interface
│   ├── CountingEvaluator.py    # Basic piece-count heuristic
│   ├── ImprovedEvaluator.py    # Advanced position-weighted heuristic
│   ├── FastEvaluator.py        # Optimized evaluation
│   └── othello.sh          # Execution script
└── test_code/
    ├── othellostart.sh     # Game runner script
    └── othello_naive.sh    # Naive opponent
```

---

## 🎮 Game Rules

Othello is played on an 8×8 board:

- **White (O)** always starts (MAX player)
- **Black (X)** is the MIN player
- Players place pieces to capture opponent's pieces
- Game ends when no legal moves exist for both players
- Winner has the most pieces

---

## 🧠 Algorithm Features

### Alpha-Beta Pruning

- Minimax search with alpha-beta cutoffs
- Significantly reduces search space

### Iterative Deepening Search (IDS)

- Starts at depth 1, incrementally increases
- Returns best move from last completed depth
- Guarantees a move within time limit

### Position-Weighted Heuristics

```
Corner weights: 120 (highly valuable)
Edge weights:   20  (moderately valuable)
X-squares:     -20  (dangerous positions)
C-squares:     -40  (very dangerous)
```

### Optimizations

- **Transposition tables:** Cache evaluated positions
- **Move ordering:** Examine promising moves first
- **History heuristics:** Learn from successful moves

---

## 🚀 Usage

### Running the Engine

```bash
# Navigate to Python directory
cd Othello/Python

# Run with position string and time limit
python Othello.py "WEEEEEEEEEEEEEEEEEEEEEEEEEEEOXEEEEEEXOEEEEEEEEEEEEEEEEEEEEEEEEEEE" 5

# Using the shell script
bash othello.sh "WEEEEEEEEEEEEEEEEEEEEEEEEEEEOXEEEEEEXOEEEEEEEEEEEEEEEEEEEEEEEEEEE" 5 0
```

### Input Format

- **Position string (65 chars):**
  - First char: `W` (White to move) or `B` (Black to move)
  - Next 64 chars: Board state (`E`=empty, `O`=white, `X`=black)
  - Reading order: top-left to bottom-right, row by row

- **Time limit:** Seconds allowed for move computation

### Output Format

- Move in format `(row,column)` using 1-based indexing
- `pass` if no legal moves exist

---

## 🏆 Playing Against Opponents

```bash
# Play your engine (white) vs naive (black)
bash test_code/othellostart.sh Python/othello.sh test_code/othello_naive.sh 5

# Play naive (white) vs your engine (black)
bash test_code/othellostart.sh test_code/othello_naive.sh Python/othello.sh 5
```

---

## 📊 Performance Requirements

The engine must outperform the naive Alpha-Beta player:

- Time limits: 2-10 seconds
- Tested as both White and Black
- Naive uses fixed depth 7 with simple piece-count evaluation

---

## 📦 Requirements

- Python 3.10+
- No external dependencies (pure Python implementation)

---

## 🔧 Key Classes

| Class | Description |
|-------|-------------|
| `Othello` | Main program, handles I/O and time control |
| `AlphaBeta` | Alpha-Beta search with IDS |
| `OthelloPosition` | Board state, move generation |
| `OthelloAction` | Move representation |
| `ImprovedEvaluator` | Position-weighted heuristic |

---

## 📚 References

- Russell, S., & Norvig, P. (2020). Artificial Intelligence: A Modern Approach
- [Othello Rules (Wikipedia)](https://en.wikipedia.org/wiki/Reversi)
