import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.registration.register_model import register_model

# Ensure MLflow tracking URI is always set to the project folder
import mlflow
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
mlflow.set_tracking_uri(f"file:{os.path.join(project_root, 'mlruns')}")

def main():
    parser = argparse.ArgumentParser(description="Model Registration Job: Register trained model in MLflow.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model (joblib format).")
    parser.add_argument("--model_name", type=str, default="used_cars_price_prediction_model", help="Name to register the model under in MLflow.")
    parser.add_argument("--artifact_path", type=str, default="random_forest_price_regressor", help="Path in MLflow artifacts to store the model.")
    args = parser.parse_args()

    register_model(
        model_path=args.model_path,
        model_name=args.model_name,
        artifact_path=args.artifact_path
    )

if __name__ == "__main__":
    main()