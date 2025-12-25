# Answer Set Programming – Argumentation Frameworks

**Course:** AI-2 (5DV181) – Logic Programming  
**Author:** Michail Pettas (mai25mps)

---

## 📋 Overview

This project implements various argumentation semantics using Answer Set Programming (ASP) with Clingo. It explores both declarative ASP encodings and an imperative Python solver to compute and compare argumentation extensions.

---

## 🗂️ Project Structure

```
Answer Set Programming/
├── asp/                    # ASP encodings
│   ├── conflict_free.lp    # Conflict-free sets
│   ├── admissible.lp       # Admissible sets
│   ├── stable.lp           # Stable extensions
│   └── preferred.lp        # Preferred extensions
├── examples/               # Test frameworks
│   ├── example1-6.lp       # Lecture examples
│   └── random_*.lp         # Generated random frameworks
├── python/                 # Python implementations
│   ├── preferred.py        # Imperative preferred solver
│   ├── benchmark.py        # Performance comparison tool
│   └── generate_af.py      # Random framework generator
└── report/
    └── assignment_report.tex
```

---

## 🧠 Implemented Semantics

### 1. Conflict-Free Sets (`asp/conflict_free.lp`)

- **Guessing:** Choice rule over `in/1` predicate
- **Checking:** Integrity constraint forbidding internal attacks

### 2. Admissible Sets (`asp/admissible.lp`)

- Extends conflict-free with defense requirements
- Each included argument must be defended against all attackers

### 3. Stable Extensions (`asp/stable.lp`)

- Builds on admissibility
- Every outside argument must be attacked by the chosen set

### 4. Preferred Extensions (`asp/preferred.lp`)

- Maximal admissible sets
- Uses saturation-based maximality test (ASPARTIX-style)

---

## 🚀 Usage

### Running ASP Programs

```bash
# Enumerate conflict-free sets
clingo asp/conflict_free.lp examples/example1.lp 0

# Find admissible sets
clingo asp/admissible.lp examples/example2.lp 0

# Compute stable extensions
clingo asp/stable.lp examples/example3.lp 0

# Compute preferred extensions
clingo asp/preferred.lp examples/example4.lp 0
```

### Running Python Solver

```bash
# Compute preferred extensions using imperative solver
python python/preferred.py examples/example1.lp

# Run with benchmark mode
python python/preferred.py examples/example1.lp --benchmark
```

### Benchmarking

```bash
# Compare ASP vs Python performance
python python/benchmark.py --runs 5 --instances examples/example*.lp
```

---

## 📊 Performance Results

| Instance | #Args | #Attacks | ASP (ms) | Python (ms) |
|----------|-------|----------|----------|-------------|
| example1.lp | 1 | 1 | 7.25 | 61.06 |
| example2.lp | 2 | 2 | 6.44 | 56.96 |
| example3.lp | 3 | 2 | 6.95 | 57.85 |
| random_n10_a05 | 10 | 5 | 13.12 | 62.18 |
| random_n20_d025 | 20 | 112 | 7.75 | 56.45 |

*ASP consistently outperforms the imperative approach due to optimized constraint propagation.*

---

## 📦 Requirements

- **Clingo 5.4.0+** – ASP solver
- **Python 3.10+** – For imperative solver and benchmarking

### Installation

```bash
# Install Clingo (conda)
conda install -c potassco clingo

# Or via pip
pip install clingo
```

---

## 📚 References

- [ASPARTIX System](https://www.dbai.tuwien.ac.at/research/argumentation/aspartix/)
- Dung, P.M. (1995). On the acceptability of arguments and its fundamental role in nonmonotonic reasoning, logic programming and n-person games
