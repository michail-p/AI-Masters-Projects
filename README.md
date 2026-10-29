# AI Masters Projects

A collection of academic projects completed during the AI Masters program at Umeå University.

---

## 📁 Projects Overview

| Project | Description | Technologies |
| ------- | ----------- | ------------ |
| [Answer Set Programming](Answer%20Set%20Programming/) | Logic programming for argumentation frameworks | ASP, Clingo, Python |
| [Data Privacy 1](Data%20Privacy%201/) | Masking methods (generalization, rank swapping, noise) and privacy-utility tradeoff on Adult Census | Python, scikit-learn, Jupyter |
| [Data Privacy 2](Data%20Privacy%202/) | Seminar on privacy risks in model explanations (Shokri 2021) and reconstruction attacks on unlearning (Bertran 2024) | Python, PowerPoint, Research |
| [Data Privacy 3](Data%20Privacy%203/) | Differential privacy and CTGAN tabular synthetic data generation with utility preservation benchmarks | Python, PyTorch, SDV, CTGAN |
| [Deep Learning 1](Deep%20Learning%201/) | CNN-based facial emotion recognition on AffectNet with training and evaluation scripts | Python, TensorFlow, Keras |
| [Deep Learning 2](Deep%20Learning%202/) | Manual backpropagation and gradient descent in JAX for regression and classification models | Python, JAX, matplotlib |
| [Deep Learning 3](Deep%20Learning%203/) | Conditional VAE for facial expression generation and latent-space interpolation | Python, TensorFlow, Keras |
| [Fashion Article Classification](Fashion%20Article%20Classification/) | Image classification using k-NN and neural networks | Python, scikit-learn, TensorFlow |
| [LLM Backend](LLM%20Backend/) | FastAPI backend for educational game with LLM-powered hints | Python, FastAPI, Hugging Face |
| [Machine Learning 1](Machine%20Learning%201/) | Supervised learning assignment with classification and regression | Python, scikit-learn, Jupyter |
| [Machine Learning 2](Machine%20Learning%202/) | Neural networks, PCA, and clustering across California Housing, Wine, and MNIST | Python, scikit-learn, TensorFlow, Jupyter |
| [Machine Learning 3](Machine%20Learning%203/) | Human activity recognition with classical models and raw-signal deep learning | Python, scikit-learn, TensorFlow, Jupyter |
| [MLOps-Pipeline](MLOps-Pipeline/) | End-to-end MLOps pipeline for customer churn prediction with CI/CD and monitoring | Python, XGBoost, MLflow, Docker, FastAPI |
| [Monocular Depth Estimation](Monocular%20Depth%20Estimation/) | YOLO26 + Depth Anything 3 fusion for real-time obstacle avoidance on an ESP32-C3 robot | Python, PyTorch, FastAPI, BLE |
| [Natural Language Processing 1](Natural%20Language%20Processing%201/) | spaCy pipeline exploration: tokenization, POS tagging, dependency parsing, NER, and multilingual Swedish NLP | Python, spaCy |
| [Natural Language Processing 2](Natural%20Language%20Processing%202/) | 20 Newsgroups text classification with NLTK preprocessing, TF-IDF, and supervised linear/NB models | Python, scikit-learn, NLTK |
| [Natural Language Processing 3](Natural%20Language%20Processing%203/) | Probing vs fine-tuning Pythia language models (160M–2.8B) for implicit state changes in procedural text (bAbI & ProPara) | Python, PyTorch, Transformers, Pythia |
| [Othello](Othello/) | Game AI engine with Alpha-Beta pruning | Python, Minimax, IDS |
| [Reinforcement Learning](Reinforcement%20Learning/) | Bandit benchmarking and multi-agent Pong with Q-learning | Python, pytest, ma-gym |
| [Spin the Wheel](Spin%20the%20wheel/) | Full-stack web application prototype | Spring Boot, Angular, Firebase |
| [Statistics](Statistics/) | Statistical inference analysis | R, tidyverse |

---

## 📂 Project Details

### 🧠 [Answer Set Programming](Answer%20Set%20Programming/)

**Course:** AI-2 (5DV181) – Logic Programming

Implementation of argumentation semantics using Answer Set Programming (ASP). Includes conflict-free sets, admissible sets, stable extensions, and preferred extensions. Features both declarative ASP encodings and an imperative Python solver for performance comparison.

### 🔒 [Data Privacy 1](Data%20Privacy%201/)

**Course:** Data Privacy (5DV241)

Empirical evaluation of data protection and anonymization methods applied to the UCI Adult Census income dataset (~32,000 records). Investigates the tradeoff between disclosure risk and downstream utility:

- Data cleaning, quasi-identifier identification, and preprocessing
- Masking techniques: Generalization (interval rounding), rank swapping, and additive continuous noise
- Disclosure risk quantification using distance-based record linkage
- Information loss assessment: Generic statistical preservation vs specific task utility (Random Forest classification predicting income > $50K)
- Comparative tradeoff curves across four parameter configurations per method

### 🔍 [Data Privacy 2](Data%20Privacy%202/)

**Course:** Data Privacy (5DV241)

Advanced seminar and critical analysis of privacy vulnerabilities in machine learning models and post-hoc explanations:

- Seminar presentation and report on Shokri et al. (IEEE S&P 2021), *"On the Privacy Risks of Model Explanations"*, investigating membership inference attacks exploiting feature explanations (SHAP, saliency maps)
- Evaluation of reconstruction attacks on machine unlearning based on Bertran et al. (NeurIPS 2024), auditing weight differences before and after deletion requests
- Technical report detailing attack models, privacy-preserving explanation mechanisms, and formal mitigation strategies
- Slide deck and structured discussion Q&A

### 🛡️ [Data Privacy 3](Data%20Privacy%203/)

**Course:** Data Privacy (5DV241)

Empirical investigation of Differential Privacy (DP) and generative synthetic data generation using Conditional GANs:

- Differential Privacy mechanisms across varying privacy budgets ($\epsilon$)
- Synthetic tabular data generation with CTGAN (SDV) across sample sizes ($N \in [10, 16512]$) and multiple random seeds
- Downstream utility evaluation on Logistic Regression, Random Forest, and MLP models
- Evaluation of hybrid DP-synthetic pipelines assessing compound accuracy degradation
- Scaling benchmarks and publication-ready visualization suite

### 😃 [Deep Learning 1](Deep%20Learning%201/)

**Course:** Deep Learning

Facial emotion recognition using a Convolutional Neural Network (CNN) trained on the AffectNet dataset. Includes:

- Custom CNN architecture implemented in Keras
- Four-class emotion recognition: Happy, Sad, Surprised, and Mad
- Training, validation, and test evaluation on AffectNet emotion data
- Visualization of loss and accuracy curves
- Provided environment YAML for reproducibility

Model file: `affectnet_emotion_cnn.keras`

### 🧮 [Deep Learning 2](Deep%20Learning%202/)

**Course:** Deep Learning (5DV236)

Backpropagation assignment implemented with JAX and the course `jaxon` library. The project focuses on composing layer Jacobians into full gradients and then training models with manual gradient descent.

- Manual backpropagation through model layers
- Gradient descent for a linear regression example
- Gradient descent for a feed-forward multiclass classifier
- Learning-curve generation and written report

### 🎭 [Deep Learning 3](Deep%20Learning%203/)

**Course:** Deep Learning (5DV236)

Conditional Variational Autoencoder (CVAE) for facial expression generation on AffectNet. The model learns a 128-dimensional latent space conditioned on four emotion classes (Happy, Sad, Surprised, Mad) and supports:

- Reconstruction of 112×112 face images with multi-level label injection in the decoder
- Expression transfer — encoding a Happy face and decoding with Sad conditioning
- Smooth latent interpolation between mean class embeddings
- β-VAE training (β = 0.15) with KL annealing to prevent posterior collapse

### 👕 [Fashion Article Classification](Fashion%20Article%20Classification/)

**Course:** Machine Learning

Classification of fashion items from the Fashion-MNIST dataset using:

- Custom k-Nearest Neighbors (k-NN) implementation
- Dense Neural Network (Multilayer Perceptron)
- Hyperparameter tuning and model evaluation

### 🤖 [LLM Backend](LLM%20Backend/)

**Standalone Project**

Backend API for an educational game platform, built with FastAPI. Provides endpoints for curriculum, game logic, user modeling, and AI-powered hint generation using large language models (LLMs) via Hugging Face.

- RESTful API for game and curriculum management
- Integration with Llama-3-8B-instruct via Hugging Face Inference API
- In-memory session storage for local development
- Docker and local run support

### 📘 [Machine Learning 1](Machine%20Learning%201/)

**Course:** Machine Learning (5DV238)

Foundational supervised learning assignment covering:

- Classification on the Breast Cancer Wisconsin (Diagnostic) dataset
- Regression on the California Housing dataset
- Model training, evaluation, and comparison using scikit-learn

### 📗 [Machine Learning 2](Machine%20Learning%202/)

**Course:** Machine Learning

Second machine learning assignment covering neural network regression on California Housing, PCA on Wine and MNIST, and custom/scikit-learn K-Means clustering analysis.

### 📙 [Machine Learning 3](Machine%20Learning%203/)

**Course:** Machine Learning (5DV238)

Third machine learning assignment focused on human activity recognition using the UCI HAR smartphone dataset. The project compares classical models on engineered features with raw inertial-signal experiments, including a 1D CNN for sequence modeling.

### ⚙️ [MLOps-Pipeline](MLOps-Pipeline/)

**Standalone Project**

End-to-end MLOps pipeline for customer churn prediction using XGBoost. Demonstrates production-grade ML engineering practices:

- Experiment tracking and model registry with MLflow
- Data version control with DVC
- Model serving via FastAPI REST endpoint
- Containerised services with Docker Compose (API, MLflow, Prometheus, Grafana)
- Data drift monitoring with Evidently AI
- CI/CD with GitHub Actions (linting, testing, Docker builds)

### 🚗 [Monocular Depth Estimation](Monocular%20Depth%20Estimation/)

**Course:** Deep Learning (5DV236)

Real-time obstacle avoidance system: a phone streams its camera to a laptop, which runs YOLO26-Seg detection and Depth Anything 3 monocular metric depth, fuses them into per-object distances, plans a motor command, and drives an ESP32-C3 car over BLE.

- End-to-end fusion server with FastAPI + WebSocket + HTTPS
- Calibrated metric depth via affine fit in disparity space
- Per-instance mask-based distance aggregation (with bbox fallback)
- Phone web app for camera streaming and overlay rendering
- Gradio dev UI and latency benchmarking modes

### 🔤 [Natural Language Processing 1](Natural%20Language%20Processing%201/)

**Course:** Natural Language Processing (5DV218 Språkteknologi)

Exploration of fundamental computational linguistics and modern NLP pipelines using spaCy across English and Swedish:

- Rule-based and statistical tokenization, lemmatization, and sentence boundary detection
- Coarse and fine-grained Part-of-Speech (POS) tagging with universal morphological features
- Syntactic dependency parsing and ambiguity visualization (*"time flies like an arrow"*)
- Named Entity Recognition (NER) for standard entity classes
- Swedish language pipeline using `sv_core_news_sm` for compound words and inflectional morphology
- Vector SVG parse tree visualization with displaCy

### 📰 [Natural Language Processing 2](Natural%20Language%20Processing%202/)

**Course:** Natural Language Processing (5DV218 Språkteknologi)

End-to-end text classification benchmarking on the 20 Newsgroups dataset:

- Text preprocessing pipeline with NLTK: case normalization, punctuation filtering, stopword pruning, and Snowball stemming
- Feature representations: Bag-of-Words (`CountVectorizer`) vs TF-IDF (`TfidfVectorizer`)
- Supervised classification using Multinomial Naive Bayes, Linear Support Vector Classifier (`LinearSVC`), and Logistic Regression
- Detailed performance diagnostics: accuracy, macro/micro/weighted F1-scores, precision, recall, and normalized confusion matrix visualizations

### 📖 [Natural Language Processing 3](Natural%20Language%20Processing%203/)

**Course:** Natural Language Processing (5DV218 Språkteknologi)

Research project: *Implicit state changes in procedural text: probing vs fine-tuning*. Investigates whether autoregressive language models implicitly maintain mental representations of world state changes during procedural text processing:

- Benchmarking on the ProPara (science processes) and bAbI Task 2 (two supporting facts) datasets
- Linear layer-wise probing of frozen internal representations across EleutherAI Pythia model scales (160M, 410M, 1B, 1.4B, 2.8B)
- Full causal fine-tuning across models and seeds to evaluate implicit vs explicit state tracking
- Automated statistical analysis with bootstrap confidence intervals
- Complete reproducible experimental pipeline with automated headless Chrome report generation

### ♟️ [Othello](Othello/)

**Course:** Artificial Intelligence (5DV243)

A competitive Othello (Reversi) game engine featuring:

- Alpha-Beta pruning with iterative deepening search
- Custom position-weighted heuristic evaluation
- Transposition tables and move ordering optimizations
- Time-controlled search within specified limits

### 🤖 [Reinforcement Learning](Reinforcement%20Learning/)

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

- **Python 3.10+ / 3.11+** (for ML, DL, NLP, Data Privacy, ASP, and Othello projects)
- **CUDA GPU (8 GB+ VRAM)** (for Monocular Depth Estimation, Deep Learning 3, and NLP 3 Pythia probing/fine-tuning)
- **Java 17+** (for Spin the Wheel backend)
- **Node.js 18+** (for Angular frontend)
- **R 4.x** (for Statistics project)
- **Clingo 5.x** (for ASP project)
- **Gym/ma-gym + Pillow** (for RL Pong environment and GIF export)
- **spaCy & NLTK** (with `en_core_web_sm` and `sv_core_news_sm` models for NLP 1 & 2)
- **SDV / CTGAN & PyTorch** (for Data Privacy 3 synthetic data generation)
- **Transformers & Accelerate** (for NLP 3 LLM probing and fine-tuning)

---

## 👤 Author

**Michail Pettas**  
Umeå University – AI Masters Program

---

## 📄 License

These projects are academic assignments. Please respect academic integrity policies if referencing this work.
