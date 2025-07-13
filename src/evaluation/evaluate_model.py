import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error

def evaluate_model(model_path, test_csv_path):
    """
    Loads a trained model and evaluates it on the test data.

    Args:
        model_path (str): Path to the trained model (joblib format).
        test_csv_path (str): Path to the test data CSV.

    Returns:
        float: Mean Squared Error (MSE) of the model on the test set.
    """
    model = joblib.load(model_path)
    df = pd.read_csv(test_csv_path)
    X_test = df.drop(columns=["price"])
    y_test = df["price"]
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse