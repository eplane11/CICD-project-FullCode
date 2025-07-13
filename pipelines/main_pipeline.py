from azure.ai.ml import command, Input, Output, dsl
from azure.ai.ml.entities import PipelineJob
from azure.ai.ml.constants import AssetTypes

from itertools import product

def get_pipeline(
    compute_name,
    environment_name,
    data_asset_name,
    train_test_ratio=0.2,
    n_estimators=100,
    max_depth=None,
    model_name="used_cars_price_prediction_model",
    artifact_path="random_forest_price_regressor"
):
    # Data Preparation Step
    data_prep_job = command(
        name="data_prep_job",
        display_name="Data Preparation Job",
        description="Splits the input data into train and test sets.",
        code="../jobs",
        command="python data_prep_job.py --input_csv_path ${{inputs.input_data}} --train_output_path ${{outputs.train_data}} --test_output_path ${{outputs.test_data}} --test_size {0}".format(train_test_ratio),
        inputs={
            "input_data": Input(type=AssetTypes.URI_FILE, path=data_asset_name)
        },
        outputs={
            "train_data": Output(type=AssetTypes.URI_FILE, mode="rw_mount"),
            "test_data": Output(type=AssetTypes.URI_FILE, mode="rw_mount")
        },
        environment=environment_name,
        compute=compute_name
    )

    # Model Training Step
    # Hyperparameter grids (should match local pipeline)
    model_grids = {
        "random_forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10]
        },
        "linear_regression": {
            "dummy": [None]
        },
        "decision_tree": {
            "max_depth": [None, 5, 10],
            "random_state": [42]
        }
    }

    # Training jobs for each model type and hyperparameter combination
    training_jobs = {}
    model_types = ["random_forest", "linear_regression", "decision_tree"]
    for model_type in model_types:
        grid = model_grids[model_type]
        param_names = [k for k in grid if k != "dummy"]
        param_values = [grid[k] for k in param_names] if param_names else [[]]
        for param_combo in product(*param_values):
            params = dict(zip(param_names, param_combo))
            param_str = "_".join(f"{k}-{v}" for k, v in params.items()) if params else "default"
            job_name = f"training_job_{model_type}_{param_str}"
            model_output_name = f"model_{model_type}_{param_str}"
            cmd = (
                "python training_job.py "
                "--train_csv_path ${{inputs.train_data}} "
                "--test_csv_path ${{inputs.test_data}} "
                "--model_output_path ${{outputs.model}} "
                f"--model_type {model_type}"
            )
            if model_type == "random_forest":
                cmd += f" --n_estimators {params.get('n_estimators', 100)} --max_depth {params.get('max_depth', None)}"
            elif model_type == "decision_tree":
                cmd += f" --max_depth {params.get('max_depth', None)} --random_state {params.get('random_state', 42)}"
            # linear_regression has no extra params

            training_jobs[job_name] = command(
                name=job_name,
                display_name=f"Model Training Job ({model_type}, {param_str})",
                description=f"Trains a {model_type.replace('_', ' ').title()} model with params {params}.",
                code="../jobs",
                command=cmd,
                inputs={
                    "train_data": data_prep_job.outputs.train_data,
                    "test_data": data_prep_job.outputs.test_data
                },
                outputs={
                    "model": Output(type=AssetTypes.URI_FILE, mode="rw_mount")
                },
                environment=environment_name,
                compute=compute_name
            )

    # Model Registration Step
    registration_job = command(
        name="registration_job",
        display_name="Model Registration Job",
        description="Registers the trained model in MLflow.",
        code="../jobs",
        command="python registration_job.py --model_path ${{inputs.model}} --model_name {0} --artifact_path {1}".format(model_name, artifact_path),
        inputs={
            "model": training_job.outputs.model
        },
        environment=environment_name,
        compute=compute_name
    )

    @dsl.pipeline(
        compute=compute_name,
        description="End-to-end pipeline for used car price prediction with data prep, training, and registration."
    )
    def car_price_prediction_pipeline(input_data: str = data_asset_name):
        data_prep = data_prep_job(input_data=input_data)
        train_results = {}
        for job_name, job in training_jobs.items():
            train_results[job_name] = job(train_data=data_prep.outputs.train_data)
        # Optionally, register only the best model or all models
        # Here, just return all models for comparison
        return {
            "train_data": data_prep.outputs.train_data,
            "test_data": data_prep.outputs.test_data,
            "models": {k: v.outputs.model for k, v in train_results.items()}
        }

    return car_price_prediction_pipeline