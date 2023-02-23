import argparse
import h2o
import mlflow
import sys
import os
from mlflow.tracking import MlflowClient
from h2o.automl import H2OAutoML, get_leaderboard

# Add the src directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *

def main():

    # Set up an H2O cluster
    h2o.init()
    
    # Spin up Mlflow client
    client = MlflowClient()

    # Load the training data
    train_data = load_data(args.train_data)
    train_data = remove_id_column(train_data)
    train_data = h2o.H2OFrame(train_data)

    print(train_data)

    # Set the target column name
    target_col = args.target_column

    # Set the names of the feature columns
    feature_cols = [col for col in train_data.columns if col != target_col]

    # Ensure that H2O automl recognizes this as a classification task
    train_data[target_col] = train_data[target_col].asfactor()

    # Start an MLflow experiment for tracking
    mlflow.set_experiment(args.experiment_name)

    # Set the maximum number of models to train in AutoML
    max_models = args.max_models

    # Start an MLflow run for tracking the AutoML experiment
    with mlflow.start_run():
        
        # Get the ID of the current MLflow run
        run_id = mlflow.active_run().info.run_id

        # Split the training data into training and validation sets
        train, valid = train_data.split_frame(ratios=[0.8], seed=0)

        print(train)
        print(valid)
        
        # Train an H2O AutoML model
        automl = H2OAutoML(max_models=max_models, balance_classes=True, seed=0, sort_metric='AUC', verbosity='info')
        automl.train(x=feature_cols, y=target_col, training_frame=train, validation_frame=valid)
        
        # Log the AutoML leaderboard to MLflow and save it as a CSV file
        leaderboard = automl.leaderboard.as_data_frame()
        leaderboard.to_csv('references/leaderboard.csv', index=False)
        mlflow.log_param('max_models', max_models)
        mlflow.log_metric('auc', leaderboard['auc'][0])
        mlflow.log_artifact('references/leaderboard.csv')

        # Print the leaderboard
        print(leaderboard)

        # Get the best model from the leaderboard
        best_model_id = leaderboard.loc[0, "model_id"]
        best_model = h2o.get_model(best_model_id)

        # Save the H2O model to disk
        model_path = h2o.save_model(best_model, path='./h2o_automl_models', force=True)

        # Log the H2O model artifact to MLflow
        client.log_artifact(run_id, model_path)
        mlflow.h2o.log_model(best_model, "h2o_automl_model")

        # Get the cross-validation metrics summary table for the best model and save as csv
        cv_summary = best_model.cross_validation_metrics_summary().as_data_frame()
        cv_summary.to_csv("references/cv_summary.csv")
        
        # Log the CV Summary table 
        mlflow.log_artifact('references/cv_summary.csv')

        leaderboard_uri = mlflow.get_artifact_uri("references/leaderboard.csv")
        cv_summary_uri = mlflow.get_artifact_uri("references/cv_summary.csv")
        best_model_uri = mlflow.get_artifact_uri("h2o_automl_model")

        print(f"The leaderboard is located at: {leaderboard_uri}")
        print(f"The CV summary is located at: {cv_summary_uri}")
        print(f"The best model is located at: {best_model_uri}")

        # Evaluate the performance of the best model on the validation set
        best_model_performance = best_model.model_performance(valid)

        print(f"The validation auc is: {best_model_performance.auc()}")
        print(f"The validation accuracy is: {best_model_performance.accuracy()[0][1]}")
        print(f"The validation precision is: {best_model_performance.precision()[0][1]}")
        print(f"The validation recall is: {best_model_performance.recall()[0][1]}")

        # Log the performance of the best model on the validation set
        mlflow.log_metric("best_model_validation_auc", best_model_performance.auc())
        mlflow.log_metric("best_model_validation_accuracy", best_model_performance.accuracy()[0][1])
        mlflow.log_metric("best_model_validation_precision", best_model_performance.precision()[0][1])
        mlflow.log_metric("best_model_validation_recall", best_model_performance.recall()[0][1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, help="Name of the mlflow experiment")
    parser.add_argument("--max_models", type=int, help="Number of max models to train")
    parser.add_argument("--target_column", type=str, help="Name of target column")
    parser.add_argument("--train_data", help="path to train data file")
    args = parser.parse_args()

    main()

