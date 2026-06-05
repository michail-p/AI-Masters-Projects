import pandas as pd
import numpy as np
import os

def generate_data(num_samples=10000, output_path="data/raw/customer_churn.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    np.random.seed(42)
    
    # Generate synthetic features
    data = {
        "customer_id": range(1, num_samples + 1),
        "age": np.random.randint(18, 70, size=num_samples),
        "tenure_months": np.random.randint(1, 72, size=num_samples),
        "monthly_charges": np.random.uniform(20.0, 120.0, size=num_samples),
        "total_charges": np.zeros(num_samples), # calculated below
        "contract_type": np.random.choice(["Month-to-month", "One year", "Two year"], size=num_samples, p=[0.5, 0.25, 0.25]),
        "internet_service": np.random.choice(["DSL", "Fiber optic", "No"], size=num_samples, p=[0.3, 0.5, 0.2]),
        "tech_support": np.random.choice(["Yes", "No", "No internet service"], size=num_samples, p=[0.3, 0.5, 0.2])
    }
    
    df = pd.DataFrame(data)
    df["total_charges"] = df["tenure_months"] * df["monthly_charges"]
    
    # Generate target variable (churn) based on some heuristics to make it realistic
    # Higher chance of churn if month-to-month, high monthly charges, and no tech support
    churn_prob = np.zeros(num_samples)
    
    churn_prob += np.where(df["contract_type"] == "Month-to-month", 0.3, 0.0)
    churn_prob += np.where(df["monthly_charges"] > 80, 0.15, 0.0)
    churn_prob += np.where(df["tech_support"] == "No", 0.1, 0.0)
    churn_prob -= np.where(df["tenure_months"] > 24, 0.15, 0.0)
    
    # Clip probabilities between 0.05 and 0.95
    churn_prob = np.clip(churn_prob, 0.05, 0.95)
    
    df["churn"] = np.random.binomial(1, churn_prob)
    
    # Convert binary churn to string for better MLflow / Evidently clarity later if desired, or keep as int
    df["churn"] = df["churn"].map({1: "Yes", 0: "No"})
    
    df.to_csv(output_path, index=False)
    print(f"Dataset generated at {output_path} with {num_samples} samples.")

if __name__ == "__main__":
    generate_data()
