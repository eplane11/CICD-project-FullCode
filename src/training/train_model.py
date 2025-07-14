import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
import joblib

import mlflow
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

import traceback
import sklearn
print(f"scikit-learn version (training): {sklearn.__version__}")

def train_model(
    train_csv_path,
    test_csv_path,
    model_output_path,
    n_estimators=100,
    max_depth=None,
    random_state=42,
    model_type="random_forest"
):
    """
    Trains a regression model (Random Forest, Linear Regression, or Decision Tree) on the training data,
    evaluates on test data, logs MSE to MLflow, and saves the model.

    Args:
        train_csv_path (str): Path to the training data CSV.
        test_csv_path (str): Path to the testing data CSV.
        model_output_path (str): Path to save the trained model (joblib format).
        n_estimators (int): Number of trees in the forest (for Random Forest).
        max_depth (int or None): Maximum depth of the trees (for tree-based models).
        random_state (int): Random seed for reproducibility.
        model_type (str): Type of model to train. Supported: "random_forest", "linear_regression", "decision_tree", "gradient_boosting", "extra_trees", "svr".
    """
    try:
        print(f"Loading training data from: {train_csv_path}")
        df = pd.read_csv(train_csv_path)
        print(f"Loading test data from: {test_csv_path}")
        test_df = pd.read_csv(test_csv_path)

        print("Splitting features and target variable...")
        X = df.drop(columns=["price"])
        y = df["price"]
        X_test = test_df.drop(columns=["price"])
        y_test = test_df["price"]

        # --- MLflow integer column warning fix: Cast integer columns to float64 ---
        int_cols = X.select_dtypes(include=["int", "int32", "int64"]).columns.tolist()
        if int_cols:
            print(f"Casting integer columns to float64 to handle potential missing values: {int_cols}")
            X[int_cols] = X[int_cols].astype("float64")
            X_test[int_cols] = X_test[int_cols].astype("float64")
        # --------------------------------------------------------------------------

        print("Identifying categorical columns...")
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        print(f"Categorical columns: {cat_cols}")

        print("Applying one-hot encoding to categorical columns...")
        X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
        X_test_encoded = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
        print("Aligning test set columns to match training set...")
        X_test_encoded = X_test_encoded.reindex(columns=X_encoded.columns, fill_value=0)

        # Model selection (modular and extensible)
        print(f"Training model: {model_type}")

        # Feature scaling for linear_regression and svr
        scaler = None
        if model_type in ["linear_regression", "svr"]:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_encoded = scaler.fit_transform(X_encoded)
            X_test_encoded = scaler.transform(X_test_encoded)
            # Explicitly convert to numpy arrays to avoid sklearn feature name warning
            import numpy as np
            X_encoded = np.asarray(X_encoded)
            X_test_encoded = np.asarray(X_test_encoded)

        # Define model constructors in a modular dictionary
        model_constructors = {
            "random_forest": lambda: RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state),
            "linear_regression": lambda: LinearRegression(),
            "decision_tree": lambda: DecisionTreeRegressor(max_depth=max_depth, random_state=random_state),
            "gradient_boosting": lambda: GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state),
            "extra_trees": lambda: ExtraTreesRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state),
            "svr": lambda: SVR(),  # SVR does not use n_estimators or max_depth
            # To add/remove models, comment/uncomment lines above as needed.
        }

        if model_type not in model_constructors:
            raise ValueError(f"Unsupported model_type: {model_type}. Available models: {list(model_constructors.keys())}")

        model = model_constructors[model_type]()

        # For SVR and linear_regression, ensure input is numpy array (no feature names)
        if model_type in ["linear_regression", "svr"]:
            model.fit(X_encoded, y)
        else:
            model.fit(X_encoded, y)
        print(f"Saving trained model to: {model_output_path}")
        joblib.dump(model, model_output_path)

        print("Evaluating model on test set...")
        # For SVR and linear_regression, ensure input is numpy array (no feature names)
        if model_type in ["linear_regression", "svr"]:
            y_pred = model.predict(X_test_encoded)
        else:
            y_pred = model.predict(X_test_encoded)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Test Mean Squared Error (MSE): {mse}")
        print("Logging MSE to MLflow...")
        mlflow.set_experiment("model-training")
        with mlflow.start_run(nested=True):
            mlflow.log_param("model_type", model_type)
            mlflow.log_metric("test_mse", mse)
        print("Model training and evaluation complete.")
    except Exception as e:
        print("ERROR: Exception occurred during model training or evaluation.")
        print(f"Exception: {e}")
        traceback.print_exc()
        print("Test Mean Squared Error (MSE): N/A")
        # Optionally, log error to MLflow
        try:
            mlflow.set_experiment("model-training")
            with mlflow.start_run(nested=True):
                mlflow.log_param("model_type", model_type)
                mlflow.log_param("error", str(e))
        except Exception as mlflow_exc:
            print(f"MLflow logging also failed: {mlflow_exc}")