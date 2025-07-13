#!/bin/bash
# Remove all pipeline outputs, models, logs, and MLflow runs to start fresh

echo "Deleting all model artifacts in data/ ..."
rm -f data/model_*.joblib

echo "Deleting model comparison summary ..."
rm -f data/model_comparison_summary.csv

echo "Deleting logs ..."
rm -rf logs/

echo "Deleting MLflow runs and model registry ..."
rm -rf mlruns/

echo "Cleanup complete. The pipeline is now reset to a clean state."