import pandas as pd
from sklearn.model_selection import train_test_split
import os
import mlflow

# Ensure mlruns is always inside CICD-project-FullCode
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
mlflow.set_tracking_uri(f"file:{os.path.join(project_root, 'mlruns')}")

def split_data(input_csv_path, train_output_path, test_output_path, test_size=0.2, random_state=42):
    """
    Splits the input CSV data into train and test sets and saves them as CSV files.

    Args:
        input_csv_path (str): Path to the input CSV file.
        train_output_path (str): Path to save the training data CSV.
        test_output_path (str): Path to save the testing data CSV.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
    """
    print(f"Loading input data from: {input_csv_path}")
    df = pd.read_csv(input_csv_path)

    # Handle integer columns with missing values by casting to float64
    int_cols_with_na = [col for col in df.select_dtypes(include=['int', 'int32', 'int64']).columns if df[col].isnull().any()]
    if int_cols_with_na:
        print(f"Casting integer columns with missing values to float64: {int_cols_with_na}")
        df[int_cols_with_na] = df[int_cols_with_na].astype('float64')

    print(f"Splitting data into train and test sets (test_size={test_size}, random_state={random_state})...")
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    print(f"Saving training data to: {train_output_path} (records: {len(train_df)})")
    train_df.to_csv(train_output_path, index=False)
    print(f"Saving test data to: {test_output_path} (records: {len(test_df)})")
    test_df.to_csv(test_output_path, index=False)
    print("Logging record counts to MLflow...")
    mlflow.set_experiment("data-prep")
    with mlflow.start_run(nested=True):
        mlflow.log_metric("train_record_count", len(train_df))
        mlflow.log_metric("test_record_count", len(test_df))
    print("Data preparation complete.")