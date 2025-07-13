import pandas as pd
import joblib
from src.evaluation.evaluate_model import evaluate_model
from src.training.train_model import train_model

def test_evaluate_model(tmp_path):
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
    test_csv = tmp_path / "test.csv"
    model_path = tmp_path / "model.joblib"
    df.iloc[:1].to_csv(train_csv, index=False)
    df.iloc[1:].to_csv(test_csv, index=False)

    train_model(str(train_csv), str(test_csv), str(model_path), n_estimators=10, max_depth=2, random_state=42)
    mse = evaluate_model(str(model_path), str(test_csv))
    assert mse >= 0