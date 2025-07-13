import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.training.train_model import train_model

import os
import mlflow
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
mlflow.set_tracking_uri(f"file:{os.path.join(project_root, 'mlruns')}")

def main():
    parser = argparse.ArgumentParser(description="Model Training Job: Train a regression model.")
    parser.add_argument("--train_csv_path", type=str, required=True, help="Path to the training data CSV.")
    parser.add_argument("--test_csv_path", type=str, required=True, help="Path to the testing data CSV.")
    parser.add_argument("--model_output_path", type=str, required=True, help="Path to save the trained model (joblib format).")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in the forest (for Random Forest).")
    parser.add_argument("--max_depth", type=str, default=None, help="Maximum depth of the trees (for tree-based models).")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--model_type",
        type=str,
        default="random_forest",
        choices=["random_forest", "linear_regression", "decision_tree", "gradient_boosting", "extra_trees", "svr"],
        help="Type of model to train."
    )
    args = parser.parse_args()
    # Convert max_depth from string to int or None
    if args.max_depth is not None:
        if args.max_depth.lower() == "none":
            args.max_depth = None
        else:
            args.max_depth = int(args.max_depth)

    train_model(
        train_csv_path=args.train_csv_path,
        test_csv_path=args.test_csv_path,
        model_output_path=args.model_output_path,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        model_type=args.model_type
    )

if __name__ == "__main__":
    main()