# End-to-End MLOps Pipeline: Customer Churn Prediction

This repository demonstrates a complete, production-ready machine learning pipeline designed to predict customer churn. It showcases practical skills in model training, experiment tracking, model serving, containerization, data drift monitoring, and CI/CD automation.

## 🚀 Key MLOps Principles Demonstrated

1. **Experiment Tracking & Model Registry:** Utilizing **MLflow** to rigorously track hyperparameters, evaluation metrics (Accuracy, F1, ROC-AUC), and artifact versioning.
2. **Data Version Control (DVC):** Ensuring data reproducibility by tracking the raw datasets.
3. **Model Serving:** Exposing the best performing model via a robust, asynchronous **FastAPI** REST endpoint.
4. **Containerization & Orchestration:** Fully dockerized services managed with **Docker Compose**, ensuring parity across development, testing, and production environments.
5. **Production Monitoring:**
   - **Prometheus & Grafana:** Capturing and visualizing live API traffic and latency metrics.
   - **Evidently AI:** Generating comprehensive HTML reports for detecting Data Drift between reference (training) and current (production) data.
6. **Continuous Integration (CI/CD):** Utilizing **GitHub Actions** for automated linting (flake8), unit testing (pytest), and Docker image builds on every push to the main branch.

## 🛠️ Technology Stack

- **Machine Learning:** Scikit-Learn, XGBoost, Pandas
- **Experiment Tracking:** MLflow
- **Data Versioning:** DVC
- **API Framework:** FastAPI, Uvicorn, Pydantic
- **Monitoring:** Evidently AI, Prometheus, Grafana
- **DevOps/Deployment:** Docker, Docker Compose, GitHub Actions

## 🏗️ Architecture

```text
                           +-------------------+
                           |                   |
                           |   GitHub Actions  | (CI/CD)
                           |                   |
                           +---------+---------+
                                     |
                                     v
+------------------+       +-------------------+       +------------------+
|                  |       |                   |       |                  |
|    DVC / Git     +------>+   src/train.py    +------>+  MLflow Server   |
|  (Data & Code)   |       |   (XGBoost)       |       |  (Registry & UI) |
|                  |       |                   |       |                  |
+------------------+       +-------------------+       +---------+--------+
                                                                 |
                                                                 v
+------------------+       +-------------------+       +------------------+
|                  |       |                   |       |                  |
|   Evidently AI   +<------+    FastAPI App    +<------+ Models & Artifacts|
|  (Drift Reports) |       |  (Prediction API) |       |                  |
|                  |       |                   |       |                  |
+------------------+       +---------+---------+       +------------------+
                                     |
                                     v
+------------------+       +-------------------+
|                  |       |                   |
|     Grafana      +<------+    Prometheus     |
|   (Dashboards)   |       |     (Metrics)     |
|                  |       |                   |
+------------------+       +-------------------+
```

## 🏁 Quick Start

### 1. Prerequisites

Ensure you have the following installed on your machine:

- Docker and Docker Compose
- Python 3.11+
- Git

### 2. Clone the Repository

```bash
git clone <your-repo-url>
cd mlops
```

### 3. Setup Python Environment & DVC

```bash
python -m venv venv
source venv/Scripts/activate  # On Windows
# source venv/bin/activate    # On macOS/Linux
pip install -r requirements.txt
dvc pull  # To fetch the dataset if remote is configured
```

### 4. Run the Full Pipeline using Docker Compose

This command spins up the MLflow tracking server, the FastAPI prediction application, Prometheus, and Grafana.

```bash
docker compose up -d --build
```

### 5. Train the Model

Once the MLflow server is running, train the model to register it in MLflow:

```bash
python src/train.py
```

After training, you can view the experiments at `http://localhost:5000`.

*Note: Since the API dynamically loads the latest model on startup, you must restart the API container after the first training run so it can fetch the artifact.*

```bash
docker compose restart api
```

### 6. Test the Prediction Endpoint

The FastAPI application will be running at `http://localhost:8000`.

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"age": 30, "tenure_months": 12, "monthly_charges": 70.0, "total_charges": 840.0, "contract_type": "Month-to-month", "internet_service": "Fiber optic", "tech_support": "No"}'
```

You should receive a JSON response with the churn prediction and probability.

### 7. Monitoring & Dashboards

- **Prometheus:** Available at `http://localhost:9090`
- **Grafana:** Available at `http://localhost:3000` (Default login: `admin` / `admin`). Configure Prometheus as a data source and create dashboards using the `prediction_requests_total` and `prediction_latency_seconds` metrics.
- **Data Drift (Evidently):** Generate a drift report by running `python src/monitor.py`. The resulting HTML report will be saved in `reports/data_drift.html`.

## 📂 Project Structure

```text
.
├── .github/workflows/      # CI/CD pipelines
├── data/raw/               # DVC tracked datasets
├── models/                 # Serialized label encoders (Joblib)
├── reports/                # Evidently AI HTML drift reports
├── src/
│   ├── api.py              # FastAPI application & Prometheus middleware
│   ├── data_generator.py   # Script to generate synthetic customer churn data
│   ├── monitor.py          # Evidently AI script for data drift
│   └── train.py            # XGBoost training pipeline & MLflow integration
├── tests/                  # Pytest unit tests
├── docker-compose.yml      # Multi-container orchestration
├── Dockerfile              # Container image definition for FastAPI
├── prometheus.yml          # Prometheus configuration
├── requirements.txt        # Python dependencies
└── README.md               # You are here!
```
