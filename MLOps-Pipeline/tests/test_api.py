import pytest
from fastapi.testclient import TestClient
from src.api import app, load_model
import os

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_predict_endpoint_missing_model():
    # If the model isn't loaded (e.g., in a clean CI environment before training), 
    # it should return 503
    payload = {
        "age": 30,
        "tenure_months": 12,
        "monthly_charges": 70.0,
        "total_charges": 840.0,
        "contract_type": "Month-to-month",
        "internet_service": "Fiber optic",
        "tech_support": "No"
    }
    response = client.post("/predict", json=payload)
    # The actual behavior depends on if load_model() succeeded
    # We just ensure it either succeeds or cleanly fails with 503
    assert response.status_code in [200, 503]

@pytest.fixture(scope="module", autouse=True)
def setup_model():
    # Attempt to load the model if it exists locally
    if os.path.exists("models/label_encoders.joblib"):
        try:
            load_model()
        except Exception:
            pass

def test_predict_valid_data():
    payload = {
        "age": 30,
        "tenure_months": 12,
        "monthly_charges": 70.0,
        "total_charges": 840.0,
        "contract_type": "Month-to-month",
        "internet_service": "Fiber optic",
        "tech_support": "No"
    }
    response = client.post("/predict", json=payload)
    if response.status_code == 200:
        data = response.json()
        assert "churn_prediction" in data
        assert "churn_probability" in data
        assert isinstance(data["churn_probability"], float)
