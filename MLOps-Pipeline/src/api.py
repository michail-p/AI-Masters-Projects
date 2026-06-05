from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import mlflow.sklearn
import os
import time
from prometheus_client import make_asgi_app, Counter, Histogram

app = FastAPI(title="Customer Churn Prediction API")

# Add Prometheus ASGI app
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Prometheus metrics
PREDICTION_REQUESTS = Counter(
    "prediction_requests_total",
    "Total prediction requests",
    ["churn_prediction"]
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Latency of prediction requests",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0]
)

# Configure MLflow
mlflow.set_tracking_uri("http://mlflow:5000")

# Global variables
model = None
le_dict = None

class CustomerData(BaseModel):
    age: int
    tenure_months: int
    monthly_charges: float
    total_charges: float
    contract_type: str
    internet_service: str
    tech_support: str

@app.on_event("startup")
def load_model():
    global model, le_dict
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    
    try:
        model_name = "CustomerChurnXGB"
        # Since we just re-initialized the DB, the model is version 1
        model_uri = f"models:/{model_name}/1"
        model = mlflow.sklearn.load_model(model_uri)
        
        le_path = "models/label_encoders.joblib"
        if os.path.exists(le_path):
            le_dict = joblib.load(le_path)
        else:
            raise FileNotFoundError(f"Label encoders not found at {le_path}")
            
        print(f"Model {model_name} and encoders loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.get("/")
def read_root():
    return {"message": "Customer Churn API with Prometheus Metrics"}

@app.post("/predict")
def predict(data: CustomerData):
    start_time = time.time()
    
    if model is None or le_dict is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        df = pd.DataFrame([data.dict()])
        for col, le in le_dict.items():
            if col in df.columns and col != "churn":
                df[col] = le.transform(df[col])
        
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]
        
        churn_label = le_dict["churn"].inverse_transform([prediction])[0]
        
        # Track metrics
        PREDICTION_REQUESTS.labels(churn_prediction=churn_label).inc()
        PREDICTION_LATENCY.observe(time.time() - start_time)
        
        return {
            "churn_prediction": churn_label,
            "churn_probability": float(probability)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
