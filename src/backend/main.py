import pandas as pd
import h2o
import mlflow
import mlflow.h2o
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

# Set up an H2O cluster
h2o.init()
    
# Spin up Mlflow client
client = MlflowClient()

# Get the best model in the experiment used for training by first
# getting all experiments and searching for run with highest auc metric
experiments = mlflow.search_experiments(view_type=ViewType.ALL)
experiment_ids = [exp.experiment_id for exp in experiments]
runs_df = mlflow.search_runs(
    experiment_ids=experiment_ids,  # List of experiment IDs to search
    run_view_type=ViewType.ALL, # View all runs
    order_by=["metrics.best_model_validation_accuracy DESC"],  # Metrics to sort by and sort order
    max_results=1  # Maximum number of runs to return
)

# Extract the run_id and experiment_id of the top run
run_id = runs_df.iloc[0]["run_id"]
experiment_id = runs_df.iloc[0]["experiment_id"]

# Load best model based on experiment and run ids
best_model = mlflow.h2o.load_model(f"mlruns/{experiment_id}/{run_id}/artifacts/h2o_automl_model/")