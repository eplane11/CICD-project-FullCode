import os
import pandas as pd
import joblib
from src.evaluation.evaluate_model import evaluate_model
from src.training.train_model import train_model

def test_evaluate_model(tmp_path):
    # Use the real used_cars.csv for testing
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
    mse = evaluate_model(str(model_path), str(test_csv))
    assert mse >= 0