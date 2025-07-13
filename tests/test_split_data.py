import os
import pandas as pd
from src.data_prep.split_data import split_data

def test_split_data(tmp_path):
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
    input_csv = tmp_path / "input.csv"
    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"
    df.to_csv(input_csv, index=False)

    split_data(str(input_csv), str(train_csv), str(test_csv), test_size=0.5, random_state=42)

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    assert len(train_df) == 1
    assert len(test_df) == 1
    assert set(train_df.columns) == set(df.columns)
    assert set(test_df.columns) == set(df.columns)