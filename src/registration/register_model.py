import mlflow
import mlflow.sklearn
import joblib
import os
import sklearn
print(f"scikit-learn version (registration): {sklearn.__version__}")

def register_model(model_path, model_name="used_cars_price_prediction_model", artifact_path="random_forest_price_regressor"):
    """
    Registers a trained model in the MLflow model registry.

    Args:
        model_path (str): Path to the trained model (joblib format).
        model_name (str): Name to register the model under in MLflow.
        artifact_path (str): Path in MLflow artifacts to store the model.
    """
    import pandas as pd

    model = joblib.load(model_path)

    # Attempt to infer input_example from training data
    # Assume training data is in the same directory as model_path, named 'train.csv'
    model_dir = os.path.dirname(model_path)
    train_csv_path = os.path.join(model_dir, "train.csv")
    input_example = None
    if os.path.exists(train_csv_path):
        import numpy as np
        df = pd.read_csv(train_csv_path)
        X = df.drop(columns=["price"])
        # One-hot encode categorical columns as in training
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        int_cols = X.select_dtypes(include=["int", "int32", "int64"]).columns.tolist()
        X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        input_example = X_encoded.iloc[[0]].copy()
        # Set NaN in columns that originated from integer columns to force float schema
        for col in int_cols:
            if col in input_example.columns:
                input_example.at[input_example.index[0], col] = np.nan
    else:
        # Fallback: try to create a dummy input if train.csv is not found
        input_example = None

    mlflow.set_experiment("model-registration")
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            sk_model=model,
            registered_model_name=model_name,
            input_example=input_example,
            artifact_path=artifact_path
        )