import os
import subprocess

def run_step(cmd, step_name, log_file=None):
    print(f"========== Starting: {step_name} ==========")
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "w") as f:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            f.write("=== STDOUT ===\n")
            f.write(result.stdout)
            f.write("\n=== STDERR ===\n")
            f.write(result.stderr)
        print(f"Output logged to {log_file}")
    else:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("ERROR:", result.stderr)
    print(f"========== Finished: {step_name} ==========\n")
    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {step_name}")

def main():
    # Load environment variables from keys.env if present
    env_path = os.path.join(os.path.dirname(__file__), "keys.env")
    if os.path.exists(env_path):
        print("Loading environment variables from keys.env")
        with open(env_path) as f:
            for line in f:
                if line.strip() and not line.strip().startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value.strip().strip('"')

    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    jobs_dir = os.path.join(base_dir, "jobs")
    input_csv = os.path.join(data_dir, "used_cars.csv")
    train_csv = os.path.join(data_dir, "train.csv")
    test_csv = os.path.join(data_dir, "test.csv")
    model_path = os.path.join(data_dir, "model.joblib")

    # Step 1: Data Preparation
    run_step(
        f"python3 {os.path.join(jobs_dir, 'data_prep_job.py')} --input_csv_path {input_csv} --train_output_path {train_csv} --test_output_path {test_csv}",
        "Data Preparation"
    )

    # Step 2: Model Training for multiple models, with MSE summary
    import re
    import csv
    from itertools import product
    from sklearn.metrics import r2_score, mean_absolute_error
    import concurrent.futures
    import multiprocessing
    import ast

    # Add or remove models here for modularity
    model_types = [
        "random_forest",
        "linear_regression",
        "decision_tree",
        "gradient_boosting",
        "extra_trees",
        "svr"
    ]
    model_grids = {
        "random_forest": {
            "n_estimators": [50, 100, 200, 300, 500],
            "max_depth": [None, 5, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["auto", "sqrt", "log2"],
            "bootstrap": [True, False],
            "random_state": [42]
        },
        "linear_regression": {
            "fit_intercept": [True, False],
            "normalize": [False],
            "copy_X": [True, False],
            "n_jobs": [None]
        },
        "decision_tree": {
            "max_depth": [None, 5, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["auto", "sqrt", "log2", None],
            "random_state": [42]
        },
        "gradient_boosting": {
            "n_estimators": [50, 100, 200, 300, 500],
            "max_depth": [3, 5, 10, 20],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.8, 1.0],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["auto", "sqrt", "log2"],
            "random_state": [42]
        },
        "extra_trees": {
            "n_estimators": [50, 100, 200, 300, 500],
            "max_depth": [None, 5, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["auto", "sqrt", "log2"],
            "random_state": [42]
        },
        "svr": {
            "C": [0.1, 1, 10, 100],
            "kernel": ["rbf", "linear", "poly"],
            "gamma": ["scale", "auto"],
            "epsilon": [0.1, 0.2, 0.5, 1.0]
        }
    }

    # --- Finetuning logic: check for previous best model and parameters ---
    summary_csv = os.path.join(data_dir, "model_comparison_summary.csv")
    best_model_type = None
    best_model_params = None
    best_model_path = None
    best_model_row = None
    if os.path.exists(summary_csv):
        with open(summary_csv, "r") as f:
            reader = csv.DictReader(f)
            best_mse = float("inf")
            for row in reader:
                try:
                    mse = float(row["best_test_mse"])
                except Exception:
                    continue
                if mse < best_mse and row["best_params"]:
                    best_mse = mse
                    best_model_type = row["model_type"]
                    best_model_params = ast.literal_eval(row["best_params"])
                    best_model_row = row
        if best_model_type:
            best_model_path = os.path.join(
                data_dir,
                f"model_{best_model_type}_{'_'.join(f'{k}-{v}' for k,v in best_model_params.items()) if best_model_params else 'default'}.joblib"
            )
            if not os.path.exists(best_model_path):
                best_model_path = None

    # If best model exists, prepare for finetuning
    finetune_mode = best_model_type is not None and best_model_path is not None
    if finetune_mode:
        print(f"Found previous best model: {best_model_type} with params {best_model_params}")
        print(f"Model file: {best_model_path}")
    else:
        print("No previous best model found, running full grid search.")

    # For other models, use best params as starting point (if compatible)
    def inject_best_params_to_grid(model_type, grid, best_params):
        # Only inject params that exist in the grid for this model
        if not best_params:
            return grid
        new_grid = {}
        for k, v in grid.items():
            if k in best_params and best_params[k] in v:
                # Place best param value first in the list
                new_grid[k] = [best_params[k]] + [x for x in v if x != best_params[k]]
            else:
                new_grid[k] = v
        return new_grid

    model_paths = {}
    model_metrics = {}
    mse_pattern = re.compile(r"Test Mean Squared Error \(MSE\): ([\d\.eE+-]+)")

    # Configurable parameters
    max_parallel_jobs = int(os.environ.get("MAX_PARALLEL_JOBS", multiprocessing.cpu_count()))
    early_stop_patience = int(os.environ.get("EARLY_STOP_PATIENCE", 5))  # Number of bad runs before stopping
    early_stop_mse = float(os.environ.get("EARLY_STOP_MSE", 1e10))  # MSE threshold for "bad" result

    def train_and_evaluate(cmd, log_file, model_out, test_csv):
        run_step(cmd, f"Train job", log_file=log_file)
        with open(log_file, "r") as lf:
            log_content = lf.read()
        match = mse_pattern.search(log_content)
        mse = float(match.group(1)) if match else None
        try:
            import joblib
            import pandas as pd
            model = joblib.load(model_out)
            test_df = pd.read_csv(test_csv)
            X_test = test_df.drop(columns=["price"])
            y_test = test_df["price"]
            cat_cols = X_test.select_dtypes(include=["object", "category"]).columns.tolist()
            X_test_encoded = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
            if hasattr(model, "feature_names_in_"):
                X_test_encoded = X_test_encoded.reindex(columns=model.feature_names_in_, fill_value=0)
            y_pred = model.predict(X_test_encoded)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
        except Exception as e:
            r2 = None
            mae = None
        return mse, r2, mae

    for model_type in model_types:
        # In finetune mode, run each model ONCE with the best params mapped as appropriate (no grid search)
        if finetune_mode:
            print(f"========== Fine-tuning all models with best params from {best_model_type} ==========")
            # For the best model, do a small grid search around the best params; for others, just use mapped best params
            if model_type == best_model_type:
                print(f"Fine-tuning best model {model_type} with a small grid around best params: {best_model_params}")
                # Build a small grid around each numeric/categorical param
                grid = {}
                for k, v in best_model_params.items():
                    if k in model_grids[model_type]:
                        values = model_grids[model_type][k]
                        if isinstance(v, int) and len(values) > 1:
                            idx = values.index(v) if v in values else None
                            # Take v and its neighbors in the grid
                            if idx is not None:
                                grid[k] = list(set([v] + [values[i] for i in [idx-1, idx+1] if 0 <= i < len(values)]))
                            else:
                                grid[k] = [v]
                        elif isinstance(v, float) and len(values) > 1:
                            idx = values.index(v) if v in values else None
                            if idx is not None:
                                grid[k] = list(set([v] + [values[i] for i in [idx-1, idx+1] if 0 <= i < len(values)]))
                            else:
                                grid[k] = [v]
                        elif isinstance(v, str) and v in values and len(values) > 1:
                            idx = values.index(v)
                            grid[k] = list(set([v] + [values[i] for i in [idx-1, idx+1] if 0 <= i < len(values)]))
                        else:
                            grid[k] = [v]
                    else:
                        grid[k] = [v]
                param_names = list(grid.keys())
                param_values = [grid[k] for k in param_names]
                all_results = []
                best_mse = float("inf")
                best_metrics = None
                best_params = None
                param_combos = list(product(*param_values))
                log_dir = os.path.join(base_dir, "logs")
                os.makedirs(log_dir, exist_ok=True)
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_jobs) as executor:
                    futures = []
                    for param_combo in param_combos:
                        params = dict(zip(param_names, param_combo))
                        model_out = os.path.join(
                            data_dir,
                            f"model_{model_type}_{'_'.join(f'{k}-{v}' for k,v in params.items()) if params else 'default'}_finetuned.joblib"
                        )
                        cmd = (
                            f"python3 {os.path.join(jobs_dir, 'training_job.py')} "
                            f"--train_csv_path {train_csv} "
                            f"--test_csv_path {test_csv} "
                            f"--model_output_path {model_out} "
                            f"--model_type {model_type}"
                        )
                        if model_type in ["random_forest", "gradient_boosting", "extra_trees"]:
                            cmd += f" --n_estimators {params.get('n_estimators', 100)} --max_depth {params.get('max_depth', None)}"
                        elif model_type == "decision_tree":
                            cmd += f" --max_depth {params.get('max_depth', None)} --random_state {params.get('random_state', 42)}"
                        log_file = os.path.join(
                            log_dir,
                            f"model_{model_type}_{'_'.join(f'{k}-{v}' for k,v in params.items()) if params else 'default'}_finetuned.log"
                        )
                        future = executor.submit(train_and_evaluate, cmd, log_file, model_out, test_csv)
                        futures.append((future, params, model_out))
                    for i, (future, params, model_out) in enumerate(futures):
                        mse, r2, mae = future.result()
                        metrics = {"mse": mse, "r2": r2, "mae": mae, "params": params, "model_path": model_out}
                        all_results.append(metrics)
                        if mse is not None and mse < best_mse:
                            best_mse = mse
                            best_metrics = metrics
                            best_params = params
                model_metrics[model_type] = {"best": best_metrics, "all": all_results}
            else:
                # For other models, just use mapped best params (single run)
                params = {}
                if best_model_params:
                    for k, v in best_model_params.items():
                        if k in model_grids[model_type]:
                            params[k] = v
                model_out = os.path.join(
                    data_dir,
                    f"model_{model_type}_{'_'.join(f'{k}-{v}' for k,v in params.items()) if params else 'default'}_finetuned.joblib"
                )
                cmd = (
                    f"python3 {os.path.join(jobs_dir, 'training_job.py')} "
                    f"--train_csv_path {train_csv} "
                    f"--test_csv_path {test_csv} "
                    f"--model_output_path {model_out} "
                    f"--model_type {model_type}"
                )
                if model_type in ["random_forest", "gradient_boosting", "extra_trees"]:
                    cmd += f" --n_estimators {params.get('n_estimators', 100)} --max_depth {params.get('max_depth', None)}"
                elif model_type == "decision_tree":
                    cmd += f" --max_depth {params.get('max_depth', None)} --random_state {params.get('random_state', 42)}"
                log_dir = os.path.join(base_dir, "logs")
                os.makedirs(log_dir, exist_ok=True)
                log_file = os.path.join(
                    log_dir,
                    f"model_{model_type}_{'_'.join(f'{k}-{v}' for k,v in params.items()) if params else 'default'}_finetuned.log"
                )
                mse, r2, mae = train_and_evaluate(cmd, log_file, model_out, test_csv)
                metrics = {"mse": mse, "r2": r2, "mae": mae, "params": params, "model_path": model_out}
                model_metrics[model_type] = {"best": metrics, "all": [metrics]}
            continue

        print(f"========== Hyperparameter tuning for: {model_type} ==========")
        # For other models, inject best params as first candidate if compatible
        grid = inject_best_params_to_grid(model_type, model_grids[model_type], best_model_params)
        param_names = [k for k in grid if k != "dummy"]
        param_values = [grid[k] for k in param_names] if param_names else [[None]]
        best_mse = float("inf")
        best_metrics = None
        best_params = None
        all_results = []
        futures = []
        param_combos = list(product(*param_values))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel_jobs) as executor:
            for param_combo in param_combos:
                params = dict(zip(param_names, param_combo)) if param_names else {}
                model_out = os.path.join(data_dir, f"model_{model_type}_{'_'.join(f'{k}-{v}' for k,v in params.items()) if params else 'default'}.joblib")
                cmd = (
                    f"python3 {os.path.join(jobs_dir, 'training_job.py')} "
                    f"--train_csv_path {train_csv} "
                    f"--test_csv_path {test_csv} "
                    f"--model_output_path {model_out} "
                    f"--model_type {model_type}"
                )
                if model_type in ["random_forest", "gradient_boosting", "extra_trees"]:
                    cmd += f" --n_estimators {params.get('n_estimators', 100)} --max_depth {params.get('max_depth', None)}"
                elif model_type == "decision_tree":
                    cmd += f" --max_depth {params.get('max_depth', None)} --random_state {params.get('random_state', 42)}"
                log_dir = os.path.join(base_dir, "logs")
                os.makedirs(log_dir, exist_ok=True)
                log_file = os.path.join(log_dir, f"model_{model_type}_{'_'.join(f'{k}-{v}' for k,v in params.items()) if params else 'default'}.log")
                future = executor.submit(train_and_evaluate, cmd, log_file, model_out, test_csv)
                futures.append((future, params, model_out))
            bad_count = 0
            for i, (future, params, model_out) in enumerate(futures):
                mse, r2, mae = future.result()
                metrics = {"mse": mse, "r2": r2, "mae": mae, "params": params, "model_path": model_out}
                all_results.append(metrics)
                if mse is not None and mse < best_mse:
                    best_mse = mse
                    best_metrics = metrics
                    best_params = params
                    bad_count = 0  # reset on improvement
                else:
                    bad_count += 1
                # Early stopping: if too many bad runs, break
                if bad_count >= early_stop_patience and (mse is None or mse > early_stop_mse):
                    print(f"Early stopping for {model_type}: {bad_count} consecutive bad results (MSE>{early_stop_mse})")
                    break
        model_metrics[model_type] = {"best": best_metrics, "all": all_results}

    # Print summary as a table
    print("========== Model Comparison Summary ==========")
    header = f"{'Model':<20} {'Best Test MSE':<20} {'Best Test R2':<20} {'Best Test MAE':<20} {'Best Params':<30}"
    print(header)
    print("-" * len(header))
    best_model = None
    best_mse = float("inf")
    for model_type, results in model_metrics.items():
        best = results["best"]
        if best is None:
            line = f"{model_type:<20} {'N/A':<20} {'N/A':<20} {'N/A':<20} {'N/A':<30}"
        else:
            mse = best["mse"]
            r2 = best["r2"]
            mae = best["mae"]
            params = best["params"]
            line = f"{model_type:<20} {mse if mse is not None else 'N/A':<20} {r2 if r2 is not None else 'N/A':<20} {mae if mae is not None else 'N/A':<20} {str(params):<30}"
            if mse is not None and mse < best_mse:
                best_mse = mse
                best_model = model_type
        print(line)
    if best_model:
        print(f"\nBest model: *** {best_model} *** (lowest Test MSE = {best_mse})")
    else:
        print("\nNo valid MSEs found for any model.")

    # Save summary to CSV
    summary_csv = os.path.join(data_dir, "model_comparison_summary.csv")
    with open(summary_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["model_type", "best_test_mse", "best_test_r2", "best_test_mae", "best_params"])
        for model_type, results in model_metrics.items():
            best = results["best"]
            if best is not None:
                writer.writerow([
                    model_type,
                    best["mse"] if best["mse"] is not None else "",
                    best["r2"] if best["r2"] is not None else "",
                    best["mae"] if best["mae"] is not None else "",
                    str(best["params"])
                ])
    print(f"\nModel comparison summary saved to: {summary_csv}")

    # Step 3: Model Registration
    run_step(
        f"python3 {os.path.join(jobs_dir, 'registration_job.py')} --model_path {model_path}",
        "Model Registration"
    )

if __name__ == "__main__":
    main()