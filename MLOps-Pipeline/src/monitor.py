import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import os

def generate_drift_report(reference_path="data/raw/customer_churn.csv", output_path="reports/data_drift.html"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load reference data
    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Reference data not found at {reference_path}")
    
    ref_data = pd.read_csv(reference_path)
    
    # Simulate "current" data with some drift
    # Increase monthly charges, lower tenure, change contract types
    curr_data = ref_data.copy()
    
    # Simulate drift
    curr_data['monthly_charges'] = curr_data['monthly_charges'] * np.random.uniform(1.0, 1.3, size=len(curr_data))
    curr_data['tenure_months'] = curr_data['tenure_months'] * np.random.uniform(0.5, 1.0, size=len(curr_data))
    
    # Randomly switch some contracts to Month-to-month
    mask = np.random.rand(len(curr_data)) < 0.3
    curr_data.loc[mask, 'contract_type'] = "Month-to-month"
    
    print("Generating Evidently Data Drift Report...")
    # Initialize and run report
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=ref_data, current_data=curr_data)
    
    # Save report
    drift_report.save_html(output_path)
    print(f"Data Drift report saved to: {output_path}")

if __name__ == "__main__":
    generate_drift_report()
