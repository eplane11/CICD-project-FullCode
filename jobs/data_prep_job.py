import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_prep.split_data import split_data

def main():
    parser = argparse.ArgumentParser(description="Data Preparation Job: Split data into train and test sets.")
    parser.add_argument("--input_csv_path", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--train_output_path", type=str, required=True, help="Path to save the training data CSV.")
    parser.add_argument("--test_output_path", type=str, required=True, help="Path to save the testing data CSV.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of the dataset for test split.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    split_data(
        input_csv_path=args.input_csv_path,
        train_output_path=args.train_output_path,
        test_output_path=args.test_output_path,
        test_size=args.test_size,
        random_state=args.random_state
    )

if __name__ == "__main__":
    main()