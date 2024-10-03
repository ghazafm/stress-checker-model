import os
import argparse
import pandas as pd
from xgboost import XGBRegressor, XGBRFRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
)
import joblib
import datetime
import logging
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from mlflow.models.signature import infer_signature
from keras import Sequential, layers
from mlflow.models.signature import infer_signature

log_dir = os.path.join(os.path.join(os.getcwd()), "Log")
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "train.log")

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def load_data(data_dir):
    logging.info(f"Loading data from {data_dir}...")
    X_path = os.path.join(data_dir, "X_train.csv")
    y_path = os.path.join(data_dir, "y_train.csv")
    X_test_path = os.path.join(data_dir, "X_test.csv")
    y_test_path = os.path.join(data_dir, "y_test.csv")

    if not os.path.exists(X_path):
        raise FileNotFoundError(f"The file {X_path} does not exist.")

    if not os.path.exists(y_path):
        raise FileNotFoundError(f"The file {y_path} does not exist.")

    X_train = pd.read_csv(X_path)
    y_train = pd.read_csv(y_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)

    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.iloc[:, 0]  # Extract first column if y_train is a DataFrame

    logging.info("Data loaded successfully.")
    return X_train, y_train, X_test, y_test


def train_model(X_train, y_train, X_test, y_test, model_name="xgboost", params=None):
    logging.info(f"Training the model: {model_name}...")

    # Set MLflow tracking URI
    mlruns_dir = "Model/mlruns"
    mlflow.set_tracking_uri(mlruns_dir)

    model = ""
    # Start an MLflow run
    with mlflow.start_run():
        # Choose model based on input
        if model_name == "xgboost":
            model = XGBRegressor(**(params or {}))
            log_model_func = mlflow.xgboost.log_model
        elif model_name == "xgrfboost":
            model = XGBRFRegressor(**(params or {}))
            log_model_func = mlflow.xgboost.log_model
        elif model_name == "lgbm":
            model = LGBMRegressor(**(params or {}))
            log_model_func = mlflow.lightgbm.log_model
        elif model_name == "random_forest":
            model = RandomForestRegressor(
                n_estimators=100, random_state=42, **(params or {})
            )
            log_model_func = mlflow.sklearn.log_model  # Use Scikit-learn logging
        elif model_name == "svr":
            model = SVR(**(params or {}))
            log_model_func = mlflow.sklearn.log_model  # Use Scikit-learn logging
        elif model_name == "linear_regression":
            model = LinearRegression(**(params or {}))
            log_model_func = mlflow.sklearn.log_model  # Use Scikit-learn logging
        elif model_name == "ann":  # New ANN option
            # Define a simple ANN using Keras Sequential API
            model = Sequential()
            input_dim = X_train.shape[1]  # Number of input features
            model.add(layers.Dense(64, input_dim=input_dim, activation="relu"))
            model.add(layers.Dense(32, activation="relu"))
            model.add(layers.Dense(1))

            model.compile(optimizer="adam", loss="mean_squared_error")
            log_model_func = mlflow.keras.log_model  # Use MLflow's Keras logging

            # Train the ANN model
            mlflow.keras.autolog()
            model.fit(
                X_train,
                y_train,
                epochs=1000,
                batch_size=32,
                validation_data=(X_test, y_test),
                verbose=2,
            )
            mlflow.log_param("model_name", model_name)
            if params:
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)
            evaluate_model(model, X_test, y_test)
            return model
        else:
            raise ValueError(
                f"Model {model_name} is not supported. Choose from 'xgboost','xgrfboost, 'lgbm', 'random_forest', 'svr', or 'linear_regression'."
            )

        # Log parameters to MLflow
        mlflow.log_param("model_name", model_name)
        if params:
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)

        try:
            if model_name != "ann":  # ANN training already done above
                model.fit(X_train, y_train)
                logging.info(f"Model {model_name} trained successfully.")

            input_example = X_train[:5]
            predictions = np.clip(model.predict(X_train), 0, 100)  
            signature = infer_signature(X_train, predictions)

            # Log the model using the appropriate function
            log_model_func(
                model, "model", signature=signature, input_example=input_example
            )

            # Evaluate the model
            evaluate_model(model, X_test, y_test)

        except Exception as e:
            logging.error(f"Error during training: {e}")
            # Log error message to a text file
            error_file_path = "Model/error_log.txt"
            with open(error_file_path, "w") as error_file:
                error_file.write(str(e))
            mlflow.log_artifact(error_file_path)  # Log the error file to MLflow
            raise

    return model


def evaluate_model(model, X_test, y_test):
    logging.info("Evaluating the model...")

    # Predict on the test set
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, 100)  

    # Compute evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Optionally compute accuracy for classification models (if it's applicable)
    try:
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"Test Accuracy: {accuracy:.4f}")
        mlflow.log_metric("accuracy", accuracy)
    except ValueError:
        logging.info("Accuracy is not applicable for this model type.")

    # Log metrics to MLflow
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    logging.info(f"Evaluation metrics - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")


def save_model(model, model_dir, model_name, timestamp):
    # Create the model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)

    # Combine model_name with timestamp
    model_name_with_time = f"{model_name}_{timestamp}"

    # Use model_name_with_time in the file path
    model_path = os.path.join(model_dir, f"{model_name_with_time}.pkl")

    logging.info(f"Saving model to {model_path}...")

    try:
        # Save the model as a .pkl file
        joblib.dump(model, model_path)
        logging.info(f"Model saved successfully as {model_path}.")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise

    return model_path


def main(data_dir, model_dir, timestamp, model_name="xgboost", params=None):
    # Load training data
    X_train, y_train, X_test, y_test = load_data(data_dir)

    # Train the model
    model = train_model(X_train, y_train, X_test, y_test, model_name, params)

    # Save the trained model
    save_model(model, model_dir, model_name, timestamp)

    logging.info("Model training and saving completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a machine learning model.")
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        required=True,
        help="Directory where the processed data is stored.",
    )
    parser.add_argument(
        "-m",
        "--model_dir",
        type=str,
        required=True,
        help="Directory where the trained model will be saved.",
    )
    parser.add_argument(
        "-n",
        "--model_name",
        type=str,
        default="xgboost",
        help="Model to train. Options: 'xgboost', 'xgfrboost, 'lgbm', 'random_forest', 'svr', 'linear_regression'.",
    )
    parser.add_argument(
        "-p",
        "--params",
        type=str,
        default=None,
        help="Optional model hyperparameters in JSON format.",
    )
    parser.add_argument(
        "-t",
        "--timestamp",
        type=str,
        required=True,
        help="Timestamp when Makefile executed.",
    )

    args = parser.parse_args()

    # Convert params from JSON string to Python dictionary, if provided
    import json

    if args.params:
        params = json.loads(args.params)
    else:
        params = {}

    main(args.data_dir, args.model_dir, args.timestamp, args.model_name, params)
