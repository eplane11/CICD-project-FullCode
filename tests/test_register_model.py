import os
import joblib
import mlflow
import pandas as pd
from src.training.train_model import train_model
from src.registration.register_model import register_model

def test_register_model(tmp_path, monkeypatch):
    # Use the real used_cars.csv for testing
    import shutil
    used_cars_path = "data/used_cars.csv"
    assert os.path.exists(used_cars_path), "used_cars.csv not found in data/"
    df = pd.read_csv(used_cars_path)
    # Normalize column names to lowercase to match code expectations
    df.columns = [c.lower() for c in df.columns]
    # Use 80% for train, 20% for test
    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    model_path = tmp_path / "model.joblib"
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    train_model(str(train_csv), str(test_csv), str(model_path), n_estimators=10, max_depth=2, random_state=42)

    # Patch mlflow.start_run to avoid actual logging during test
    class DummyRun:
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass
    monkeypatch.setattr(mlflow, "start_run", lambda: DummyRun())

    # Should not raise
    register_model(str(model_path), model_name="test_model", artifact_path="test_artifact")