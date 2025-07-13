import os
import pandas as pd
import joblib
from src.training.train_model import train_model

def test_train_model(tmp_path):
    # Create a dummy dataframe
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

    # For test, use train_csv as both train and test (minimal viable for signature)
    train_model(str(train_csv), str(train_csv), str(model_path), n_estimators=10, max_depth=2, random_state=42)
    assert os.path.exists(model_path)
    model = joblib.load(model_path)
    assert hasattr(model, "predict")