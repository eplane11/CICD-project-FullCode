# CICD-project-FullCode

A modular, production-ready AzureML MLOps pipeline for automated used car price prediction, with full CI/CD support and one-function-per-file modularity.

---

## 📁 Project Structure

- `src/` — Modular Python code for data prep, training, evaluation, and registration (one function per file)
- `jobs/` — Job scripts for each pipeline stage (CLI entrypoints)
- `pipelines/` — AzureML pipeline definition
- `environments/` — Conda environment YAML
- `data/` — Place your `used_cars.csv` here (see `data/README.md`)
- `mlruns/` — MLflow experiment and model tracking data (auto-created; do not delete if you want to keep experiment history)
- `tests/` — Unit tests
- `.github/workflows/` — GitHub Actions CI/CD (optional)
- `azure-pipelines.yml` — Azure Pipelines CI/CD
- `requirements.txt` — Local development dependencies

---

## 📦 MLflow Tracking and Model Registry

- The `mlruns/` directory is automatically created by MLflow and stores all experiment runs, metrics, parameters, and model artifacts locally.
- **Best models are automatically registered in MLflow** under the name `used_cars_price_prediction_model`, with versioning. This allows you to track, compare, and deploy the best models directly from the MLflow Model Registry.
- **Do not delete this folder** if you want to preserve experiment history and model versions.
- In CI/CD, this folder is re-created on each run unless you persist it as an artifact or use a remote MLflow tracking server.
- For production or team environments, consider configuring MLflow to use a remote tracking URI.

---

## 🚀 Quickstart

1. **Clone the repository and install dependencies:**
    ```bash
    git clone <your-repo-url>
    cd CICD-project-FullCode
    pip install -r requirements.txt
    ```
    
    > **Note:** After running the pipeline, the best models are automatically registered in the MLflow Model Registry under the name `used_cars_price_prediction_model` with versioning. You can view and manage these models using the MLflow UI or API.
    

2. **Prepare your data:**
    - Place your `used_cars.csv` in the `data/` directory.
    - See `data/README.md` for format and registration instructions.

3. **Register the dataset as an AzureML data asset:**
    See the code snippet in `data/README.md`.

4. **Configure your AzureML workspace and compute:**
    - Update your subscription, resource group, and workspace in your pipeline orchestration script.

5. **Run the pipeline:**
    - Use the pipeline definition in [`pipelines/main_pipeline.py`](CICD-project-FullCode/pipelines/main_pipeline.py:1) to submit jobs to AzureML.

---

## ⚙️ Pipeline Logic

- The pipeline automatically trains and compares **six different machine learning models**:
  - Random Forest
  - Linear Regression
  - Decision Tree
  - Gradient Boosting
  - Extra Trees
  - Support Vector Regressor (SVR)
- **First Run:**
  The first time you run the pipeline, it performs a full grid search over all models and hyperparameters. This process is computationally intensive and will take longer to complete.
- **Subsequent Runs:**
  On later runs, the pipeline detects the best model and parameters from previous results and performs fine-tuning only on the best model, making these runs significantly faster.

You can control parallelism with the `MAX_PARALLEL_JOBS` environment variable if needed.

---

## 🛠️ CI/CD

- **Azure Pipelines:** See [`azure-pipelines.yml`](CICD-project-FullCode/azure-pipelines.yml:1) for build, test, and full pipeline automation. The pipeline installs dependencies, runs unit tests, and executes [`local_pipeline.py`](CICD-project-FullCode/local_pipeline.py:1) as an end-to-end test.
- **GitHub Actions:** See [`.github/workflows/ci.yml`](CICD-project-FullCode/.github/workflows/ci.yml:1) for alternative CI/CD. This workflow mirrors Azure Pipelines: it sets up Python, installs dependencies, runs tests, and runs the full pipeline on every push and pull request to `main`.

### CI/CD Details

- **Environment Variables & Secrets:**
  - For local runs, environment variables can be set in `keys.env`.
  - In CI/CD, use platform secrets (GitHub/DevOps secrets) and set them as environment variables in the workflow/pipeline config. Do not commit sensitive data.
- **Directory Handling:**
  - The pipeline creates and uses `data/`, `logs/`, and `mlruns/` directories. `mlruns/` is where MLflow stores all experiment and model tracking data by default.
- **MLflow Tracking (`mlruns/`):**
  - MLflow is configured to use a local file backend, storing all runs and models in the `mlruns/` folder.
  - In CI/CD, this folder is ephemeral unless persisted as an artifact or replaced with a remote tracking server.
  - For production or team use, set the MLflow tracking URI to a remote server or Azure ML workspace.
- **Parallelism:**
  - The pipeline uses all available CPUs for parallel jobs. On CI runners, this may be limited; control with the `MAX_PARALLEL_JOBS` environment variable if needed.
- **Azure ML Integration:**
  - The Azure pipeline runs the local pipeline by default. For full Azure ML pipeline integration, modularize steps as Azure ML components and use datasets/outputs for data passing.

**No code changes are strictly required for CI/CD compatibility beyond these config and workflow updates, but following the above recommendations will ensure robust, portable, and secure operation across local, GitHub Actions, and Azure ML environments.**

---

## 🧩 Modularity

- Each function is in a separate file for maximum modularity and testability.
- Job scripts in `jobs/` import from `src/` modules.
- The pipeline is assembled in [`main_pipeline.py`](CICD-project-FullCode/pipelines/main_pipeline.py:1).

---

## 🧪 Testing

- Unit tests are in the `tests/` directory.
- Run tests with:
    ```bash
    pytest tests/
    ```

---

## 📄 License

MIT License (or specify your own).

---

## 📬 Contact

For questions or contributions, open an issue or pull request on GitHub.