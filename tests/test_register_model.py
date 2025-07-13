import os
import joblib
import mlflow
import pandas as pd
from src.training.train_model import train_model
from src.registration.register_model import register_model

def test_register_model(tmp_path, monkeypatch):
    # Create a dummy dataframe and train a model
    df = pd.DataFrame({
        "Segment": ["Luxury", "Non-Luxury"],
        "Kilometers_Driven": [10000, 20000],
        "Mileage": [15.0, 18.0],
        "Engine": [2000, 1500],
        "Power": [150, 100],
        "Seats": [5, 4],
        "Price": [20, 10]
    })
    train_csv = tmp_path / "train.csv"
    model_path = tmp_path / "model.joblib"
    df.to_csv(train_csv, index=False)
    train_model(str(train_csv), str(model_path), n_estimators=10, max_depth=2, random_state=42)

    # Patch mlflow.start_run to avoid actual logging during test
    class DummyRun:
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass
    monkeypatch.setattr(mlflow, "start_run", lambda: DummyRun())

    # Should not raise
    register_model(str(model_path), model_name="test_model", artifact_path="test_artifact")