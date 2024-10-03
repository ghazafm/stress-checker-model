import bentoml
import argparse
import bentoml
import mlflow
import os

os.environ["BENTOML_HOME"] = "./Model/bentoml"

mlflow.set_tracking_uri("http://127.0.0.1:8888")


def import_model(model_name, model_version='latest'):
    print('importing....')
    client = mlflow.tracking.MlflowClient()
    print(f"client = {client}")
    if model_version == 'latest':
        versions = client.search_model_versions(f"name='{model_name}'")
        print(f'versions = {versions}')
        model_version = versions[0].version
    
    print(f"Using model version: {model_version}")
    
    bento_model = bentoml.mlflow.import_model(
        model_name, 
        f'models:/{model_name}/{model_version}'
    )
    print(f"Model imported into BentoML: {bento_model}")
    return bento_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a machine learning model.")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="model name",
    )
    parser.add_argument(
        "-v",
        "--version",
        default='latest',
        type=str,
        help="Model Version"
    )
    args = parser.parse_args()
    
    import_model(
        args.model,
        args.version
    )