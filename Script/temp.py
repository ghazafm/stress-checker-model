import os
import argparse
import joblib
import json
import logging
from datetime import datetime
import bentoml
import shap
import numpy as np

log_dir = os.path.join(os.path.join(os.getcwd()), "Log")
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "deploy.log")

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def load_model(model_path):
    logging.info(f"Loading model from {model_path}...")

    if not os.path.exists(model_path):
        logging.error(f"Model file {model_path} not found.")
        raise FileNotFoundError(f"The model file {model_path} does not exist.")

    model = joblib.load(model_path)
    logging.info("Model loaded successfully.")

    return model


def extract_model_metadata(model):
    model_metadata = {}

    # Identify model type based on its class name
    model_type = type(model).__name__
    model_metadata["model_type"] = model_type

    # Extract model parameters (if applicable)
    if hasattr(model, "get_params"):
        model_metadata["parameters"] = model.get_params()

    # For tree-based models, extract additional details
    if model_type in ["RandomForestRegressor", "XGBRegressor", "LGBMRegressor"]:
        if hasattr(model, "n_estimators"):
            model_metadata["n_estimators"] = model.n_estimators

    return model_metadata, model_type


def save_model_with_bentoml(model, model_name, model_metadata, metadata_dir):
    # Save model with BentoML
    logging.info(f"Saving model with BentoML...")
    bento_model = bentoml.sklearn.save_model(model_name, model, metadata=model_metadata)

    # Store metadata and save to a file
    metadata_file = os.path.join(metadata_dir, f"{model_name}_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(model_metadata, f, indent=4)

    logging.info(f"Model and metadata saved with BentoML as {bento_model.tag}")
    return bento_model.tag


def explain_model_with_shap(model, X_train, X_test):
    logging.info("Running SHAP feature importance analysis...")

    explainer = None
    model_type = type(model).__name__

    if model_type in ["RandomForestRegressor", "XGBRegressor", "LGBMRegressor"]:
        explainer = shap.TreeExplainer(model)
    elif model_type in ["LinearRegression", "SVR"]:
        explainer = shap.KernelExplainer(model.predict, X_train)
    else:
        raise ValueError(f"Model {model_type} is not supported for SHAP analysis.")

    # Compute SHAP values on test data
    shap_values = explainer.shap_values(X_test)

    # Calculate mean importance for each feature
    feature_importance = np.mean(np.abs(shap_values), axis=0)
    sorted_importance_indices = np.argsort(feature_importance)[
        ::-1
    ]  # Sort in descending order

    return shap_values, sorted_importance_indices


def recommend_based_on_shap(sorted_importance_indices, feature_names):
    recommendations = []
    logging.info("Generating recommendations based on SHAP values...")

    # Assume we recommend based on top 3 important features
    top_features = feature_names[sorted_importance_indices[:3]]

    for feature in top_features:
        recommendations.append(
            f"Focus on improving {feature} as it contributes the most to the prediction."
        )

    return recommendations


def deploy_model(
    model_path,
    model_dir,
    metadata_dir,
    additional_metadata,
    timestamp,
    X_train,
    X_test,
    feature_names,
):
    # Load the trained model
    model = load_model(model_path)

    # Extract dynamic metadata and model name from the model
    model_metadata, model_name = extract_model_metadata(model)

    # Merge additional metadata passed from the command line (if any)
    if additional_metadata:
        model_metadata.update(additional_metadata)

    # Save the model using BentoML and get the model tag
    model_tag = save_model_with_bentoml(model, model_name, model_metadata, metadata_dir)

    # Run SHAP feature importance and get top contributing features
    shap_values, sorted_importance_indices = explain_model_with_shap(
        model, X_train, X_test
    )

    # Generate recommendations based on feature importance
    recommendations = recommend_based_on_shap(sorted_importance_indices, feature_names)

    # Output recommendations and model tag
    print(f"Model deployed with BentoML tag: {model_tag}")
    print("Recommendations based on feature importance:")
    for rec in recommendations:
        print(rec)

    logging.info("Model deployment and SHAP analysis completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deploy a trained machine learning model with SHAP analysis."
    )

    # Arguments
    parser.add_argument(
        "-p",
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model file.",
    )
    parser.add_argument(
        "-m",
        "--model_dir",
        type=str,
        required=True,
        help="Directory to save the deployed model.",
    )
    parser.add_argument(
        "-md",
        "--metadata_dir",
        type=str,
        required=True,
        help="Directory to save model metadata.",
    )
    parser.add_argument(
        "-ma",
        "--metadata",
        type=str,
        help="Optional additional model metadata in JSON format.",
    )
    parser.add_argument(
        "-t",
        "--timestamp",
        type=str,
        required=True,
        help="Timestamp when Makefile executed.",
    )
    parser.add_argument(
        "--X_train",
        type=str,
        required=True,
        help="Path to training data for SHAP analysis (CSV format).",
    )
    parser.add_argument(
        "--X_test",
        type=str,
        required=True,
        help="Path to test data for SHAP analysis (CSV format).",
    )
    parser.add_argument(
        "--feature_names",
        type=str,
        required=True,
        help="Comma-separated list of feature names.",
    )

    args = parser.parse_args()

    # Load training and test data for SHAP analysis
    X_train = pd.read_csv(args.X_train)
    X_test = pd.read_csv(args.X_test)

    # Feature names (comma-separated string)
    feature_names = args.feature_names.split(",")

    # Load additional metadata if provided
    additional_metadata = {}
    if args.metadata:
        try:
            additional_metadata = json.loads(args.metadata)
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding metadata: {e}")
            raise ValueError("Invalid metadata JSON format.")

    # Deploy the model with SHAP analysis
    deploy_model(
        args.model_path,
        args.model_dir,
        args.metadata_dir,
        additional_metadata,
        timestamp=args.timestamp,
        X_train=X_train,
        X_test=X_test,
        feature_names=feature_names,
    )
