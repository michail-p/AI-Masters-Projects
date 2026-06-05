import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import mlflow
import mlflow.sklearn
import os
import joblib

def train():
    # 1. Load data
    data_path = "data/raw/customer_churn.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at {data_path}. Run data generator first.")
    
    df = pd.DataFrame(pd.read_csv(data_path))
    
    # 2. Preprocessing
    # Drop customer_id as it's not a feature
    df = df.drop(columns=["customer_id"])
    
    # Encode categorical features
    cat_cols = ["contract_type", "internet_service", "tech_support", "churn"]
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
    
    # Save label encoders for inference later
    os.makedirs("models", exist_ok=True)
    joblib.dump(le_dict, "models/label_encoders.joblib")
    
    X = df.drop(columns=["churn"])
    y = df["churn"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. MLflow Tracking
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Customer_Churn_Prediction")
    
    with mlflow.start_run():
        # Hyperparameters
        params = {
            "objective": "binary:logistic",
            "max_depth": 5,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "random_state": 42
        }
        
        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_prob)
        }
        
        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.sklearn.log_model(model, "model", registered_model_name="CustomerChurnXGB")
        
        # Log encoders as artifact
        mlflow.log_artifact("models/label_encoders.joblib")
        
        print("Training complete. Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    train()
