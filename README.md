# AI Masters Projects

A collection of academic projects completed during the AI Masters program at Umeå University.

---

## 📁 Projects Overview

| Project | Description | Technologies |
|---------|-------------|--------------|
| [Answer Set Programming](Answer%20Set%20Programming/) | Logic programming for argumentation frameworks | ASP, Clingo, Python |
| [Machine Learning 1](Machine%20Learning%201/) | Supervised learning assignment with classification and regression | Python, scikit-learn, Jupyter |
| [ML - Fashion Article Classification](Fashion%20Article%20Classification/) | Image classification using k-NN and neural networks | Python, scikit-learn, TensorFlow |
| [Othello](Othello/) | Game AI engine with Alpha-Beta pruning | Python, Minimax, IDS |
| [Reinforcement Learning](Reinforcement%20Learing/) | Bandit benchmarking and multi-agent Pong with Q-learning | Python, pytest, ma-gym |
| [Spin the Wheel](Spin%20the%20wheel/) | Full-stack web application prototype | Spring Boot, Angular, Firebase |
| [Statistics](Statistics/) | Statistical inference analysis | R, tidyverse |

---

## 📂 Project Details

### 🧠 [Answer Set Programming](Answer%20Set%20Programming/)

**Course:** AI-2 (5DV181) – Logic Programming

Implementation of argumentation semantics using Answer Set Programming (ASP). Includes conflict-free sets, admissible sets, stable extensions, and preferred extensions. Features both declarative ASP encodings and an imperative Python solver for performance comparison.

### 👕 [ML - Fashion Article Classification](Fashion%20Article%20Classification/)

**Course:** Machine Learning

Classification of fashion items from the Fashion-MNIST dataset using:

- Custom k-Nearest Neighbors (k-NN) implementation
- Dense Neural Network (Multilayer Perceptron)
- Hyperparameter tuning and model evaluation

### 📘 [Machine Learning 1](Machine%20Learning%201/)

**Course:** Machine Learning (5DV238)

Foundational supervised learning assignment covering:

- Classification on the Breast Cancer Wisconsin (Diagnostic) dataset
- Regression on the California Housing dataset
- Model training, evaluation, and comparison using scikit-learn

### ♟️ [Othello](Othello/)

**Course:** Artificial Intelligence (5DV243)

A competitive Othello (Reversi) game engine featuring:

- Alpha-Beta pruning with iterative deepening search
- Custom position-weighted heuristic evaluation
- Transposition tables and move ordering optimizations
- Time-controlled search within specified limits

### 🤖 [Reinforcement Learning](Reinforcement%20Learing/)

**Course:** Reinforcement Learning

Hands-on reinforcement learning project with two parts:

- Multi-armed bandit evaluation against an epsilon-greedy reference baseline
- Multi-agent Pong (`PongDuel-v0`) using a tabular Q-learning style agent
- Optional rendering, recording, and GIF export for episode analysis

### 🎰 [Spin the Wheel](Spin%20the%20wheel/)

**Course:** Software Engineering / Prototype Development

A full-stack "Spin the Wheel" application with:

- **Backend:** Spring Boot 3.x REST API with OpenAPI documentation
- **Frontend:** Angular 19 SPA with slot machine UI components
- Firebase integration for data persistence

### 📊 [Statistics](Statistics/)

**Course:** Statistics for Engineers

Statistical analysis investigating the impact of structured micro-breaks on workplace productivity. Includes hypothesis testing, paired t-tests, effect size calculations, and data visualization.

---

## 🛠️ General Requirements

- **Python 3.10+** (for ML, ASP, and Othello projects)
- **Java 17+** (for Spin the Wheel backend)
- **Node.js 18+** (for Angular frontend)
- **R 4.x** (for Statistics project)
- **Clingo 5.x** (for ASP project)
- **Gym/ma-gym + Pillow** (for RL Pong environment and GIF export)

---

## 👤 Author

**Michail Pettas**  
Umeå University – AI Masters Program

---

## 📄 License

These projects are academic assignments. Please respect academic integrity policies if referencing this work.
