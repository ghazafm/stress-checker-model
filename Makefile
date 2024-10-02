# ------------------------
# Directory and Path Variables
# ------------------------
DATA_DIR = Data/raw
NEW_DATA_DIR = Data/clean
MODEL_DIR = Model/model
PREPROCESSOR_DIR = Model/preprocessor
SCORE_RESULTS_DIR = Result/scores
TEST_RESULT_DIR = Result/test
METADATA_RESULT_DIR = Model/metadata
PREDICT_DATA_DIR = Result/predict
LOG_DIR = Log
TIMESTAMP_FILE = timestamp.txt

# ------------------------
# Model and Environment Variables
# ------------------------
DEPLOYED_MODEL_FILE = $(MODEL_DIR)/deploy_model_path.txt
LATEST_MODEL = $(shell ls -t $(MODEL_DIR)/*.pkl | head -n 1)
DEPLOYED_MODEL=$(shell cat $(DEPLOYED_MODEL_FILE))

COLUMN_TO_REMOVE = nothing
TARGET_COL = stress_level
ID_COL = PassengerId
RANDOM_STATE = 42

VENV_DIR = myenv
VENV_ACTIVATE = source $(VENV_DIR)/bin/activate
PYTHON = $(VENV_DIR)/bin/python

CONDA_ENV = myenv
REQUIREMENTS_FILE = requirements.txt
ENVIRONMENT_FILE = environment.yml

# ------------------------
# MLFlow Variables
# ------------------------
MLFLOW_PORT = 8888
MLFLOW_HOST = 0.0.0.0
ARTIFACT_ROOT = Model/mlruns

# ------------------------
# BentoML Variables
# ------------------------
BENTO_SERVICE = script.service
BENTO_MODEL = stress_checker
export BENTOML_HOME = ./Model/bentoml

# ------------------------
# Default Target
# ------------------------
.PHONY: all
all: help

# ------------------------
# Help Target
# ------------------------
# ------------------------
# Help Target
# ------------------------
.PHONY: help
help:
	@echo "Usage:"
	@echo "  Environment Setup:"
	@echo "    make conda_env               : Create and activate the Conda environment."
	@echo "    make venv_env                : Create and activate the Python virtual environment using venv."
	@echo ""
	@echo "  Data Processing:"
	@echo "    make data                    : Run data collection and preparation steps."
	@echo "    make clean_data              : Clean processed data files."
	@echo ""
	@echo "  Model Training, Evaluation, and Deployment:"
	@echo "    make train                   : Train the machine learning model."
	@echo "    make evaluate                : Evaluate the trained model."
	@echo "    make deploy                  : Save and deploy the trained model."
	@echo "    make clean_models            : Clean trained model files."
	@echo "    make clean_results           : Clean model evaluation results."
	@echo ""
	@echo "  MLflow Commands:"
	@echo "    make mlflow                  : Start the MLflow server."
	@echo "    make clean_mlflow            : Clean up MLflow files."
	@echo "    make clean_mlflow_deep       : Perform a deep clean of all MLflow files."
	@echo ""
	@echo "  BentoML Commands:"
	@echo "    make serve                   : Start BentoML HTTP service."
	@echo "    make serve_grpc              : Start BentoML gRPC service."
	@echo "    make build              		: Make BentoML Bentos."
	@echo "    make containerize            : Build Docker image for BentoML service."
	@echo "    make deploy_bento            : Run BentoML HTTP container."
	@echo "    make deploy_bento_grpc       : Run BentoML gRPC container."
	@echo "    make clean_bento             : Clean up BentoML files."
	@echo ""
	@echo "  Utility Commands:"
	@echo "    make timestamp               : Generate a timestamp for model tracking."
	@echo "    make clean_log               : Clean log files."
	@echo ""
	@echo "  Clean Up:"
	@echo "    make clean                   : Clean all data, models, results, logs, BentoML, and MLflow files."
	@echo "    make deep_clean              : Perform a deep clean (data, models, results, BentoML, and MLflow)."


# ------------------------
# Environment Setup
# ------------------------

.PHONY: check_env
check_env:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Virtual environment not found. Create one with 'make conda_env' or 'make venv_env'."; \
		exit 1; \
	fi

.PHONY: conda_env
conda_env:
	@echo "Creating Conda environment..."
	conda env create -f $(ENVIRONMENT_FILE) -p $(VENV_DIR) -y --solver=libmamba
	@echo "Environment created using $(ENVIRONMENT_FILE)."

.PHONY: venv_env
venv_env:
	@echo "Creating Python virtual environment using venv..."
	python3 -m venv $(VENV_DIR)
	$(VENV_ACTIVATE) && pip install --upgrade pip
	$(VENV_ACTIVATE) && pip install -r $(REQUIREMENTS_FILE)
	@echo "Virtual environment created and packages installed."

# ------------------------
# Utility Targets
# ------------------------

.PHONY: timestamp
timestamp:
	@echo "Generating timestamp..."
	@date +"%Y%m%d_%H%M%S" > $(TIMESTAMP_FILE)
	@echo "Timestamp saved to $(TIMESTAMP_FILE)."

# ------------------------
# Data Collection and Preparation
# ------------------------

.PHONY: data
data: check_env timestamp
	@echo "Collecting and preparing data..."
	$(PYTHON) script/data_preparation.py --data_dir $(DATA_DIR) --data_new $(NEW_DATA_DIR) --output_dir $(PREPROCESSOR_DIR) --target_col $(TARGET_COL) --random_state $(RANDOM_STATE) --columns_to_remove $(COLUMN_TO_REMOVE) --timestamp $(shell cat $(TIMESTAMP_FILE))
	@echo "Data preparation completed."

# ------------------------
# Model Training, Evaluation, and Deployment
# ------------------------

.PHONY: train
train: check_env timestamp
	@echo "Training the model with: $(MODEL_NAME)"
	$(PYTHON) script/train_model.py --data_dir $(NEW_DATA_DIR) --model_dir $(MODEL_DIR) --timestamp $(shell cat $(TIMESTAMP_FILE)) --model_name $(MODEL_NAME) --params '$(PARAMS)'
	@echo "Model training completed."

.PHONY: evaluate
evaluate: check_env timestamp
	@echo "Evaluating the trained model..."
	$(PYTHON) script/evaluate_model.py --model $(LATEST_MODEL) --results_dir $(SCORE_RESULTS_DIR) --data_dir $(NEW_DATA_DIR) --timestamp $(shell cat $(TIMESTAMP_FILE)) --problem_type regression
	@echo "Model evaluation completed."

.PHONY: deploy
deploy: check_env timestamp
	@echo "Deploying the trained model..."
	$(PYTHON) script/deploy_model.py --model_path $(LATEST_MODEL) --model_dir $(MODEL_DIR) --metadata_dir $(METADATA_RESULT_DIR) > $(DEPLOYED_MODEL_FILE) --timestamp $(shell cat $(TIMESTAMP_FILE))
	@echo "Model deployed. Path saved in $(DEPLOYED_MODEL_FILE)."

# ------------------------
# MLFlow Commands
# ------------------------

.PHONY: mlflow
mlflow: check_env
	@echo "Starting MLflow server..."
	$(VENV_ACTIVATE) && mlflow server --host $(MLFLOW_HOST) --port $(MLFLOW_PORT) --backend-store-uri $(ARTIFACT_ROOT)
	@echo "MLflow server running at http://$(MLFLOW_HOST):$(MLFLOW_PORT)"

# ------------------------
# BentoML Commands
# ------------------------

.PHONY: serve
serve: check_mlflow import_bentoml start_bentoml

.PHONY: serve_grpc
serve_grpc: check_mlflow import_bentoml start_bentoml_grpc

# Check if MLflow is running
.PHONY: check_mlflow
check_mlflow:
	@echo "Checking if MLflow server is running..."
	@if nc -zv $(MLFLOW_HOST) $(MLFLOW_PORT) >/dev/null 2>&1; then \
		echo "MLflow server is running on port $(MLFLOW_PORT). Proceeding to start BentoML..."; \
	else \
		echo "MLflow server is not running. Please start MLflow in a different terminal using the following command:"; \
		echo ""; \
		echo "    make mlflow"; \
		echo ""; \
		exit 1; \
	fi

# Import model to BentoML
.PHONY: import_bentoml
import_bentoml:
	@echo "Importing model into BentoML..."
	$(PYTHON) script/import.py -m $(BENTO_MODEL)
	@echo "Model imported into BentoML."

# Start BentoML HTTP service
.PHONY: start_bentoml
start_bentoml:
	@echo "Starting BentoML HTTP service..."
	$(VENV_ACTIVATE) && bentoml serve $(BENTO_SERVICE) --reload

# Start BentoML gRPC service
.PHONY: start_bentoml_grpc
start_bentoml_grpc:
	@echo "Starting BentoML gRPC service..."
	$(VENV_ACTIVATE) && bentoml serve-grpc $(BENTO_SERVICE) --reload

# Build Bentos for BentoML
.PHONY: build
build:
	@echo "Building Bentos..."
	$(VENV_ACTIVATE) && bentoml build -f ./Model/bentofile.yaml --no-strip-extras

.PHONY: containerize
containerize:
	@echo "Building Docker image for BentoML service..."
	$(VENV_ACTIVATE) && bentoml containerize stress_checker:latest

# Deploy BentoML container for HTTP service
.PHONY: deploy_bento
deploy_bento:
	@echo "Running BentoML HTTP container..."
	@if docker run -it --rm -p 3000:3000 stress_checker:latest serve; then \
		echo "BentoML HTTP container is running on port 3000."; \
	else \
		echo "Error: Failed to run the BentoML container." >&2; \
		exit 1; \
	fi

# Deploy BentoML container for gRPC service
.PHONY: deploy_bento_grpc
deploy_bento_grpc:
	@echo "Running BentoML gRPC container..."
	@if docker run -it --rm -p 3000:3000 stress_checker:latest serve-grpc; then \
		echo "BentoML gRPC container is running on port 3000."; \
	else \
		echo "Error: Failed to run the BentoML gRPC container." >&2; \
		exit 1; \
	fi


# ------------------------
# Cleanup Commands
# ------------------------

.PHONY: clean
clean: clean_data clean_models clean_results clean_preprocessor clean_bento clean clean_mlflow clean_mlflow_deep clean_bento
	@echo "Complete cleanup completed."

.PHONY: deep_clean
deep_clean: clean_data clean_models clean_results clean_preprocessor clean clean_mlflow
	@echo "Complete cleanup completed."

.PHONY: clean_data
clean_data:
	@echo "Cleaning data..."
	rm -rf $(NEW_DATA_DIR)/*.csv $(PREDICT_DATA_DIR)/*.csv
	@echo "Data cleaned."

.PHONY: clean_models
clean_models:
	@echo "Cleaning models..."
	rm -rf $(MODEL_DIR)/*.pkl $(METADATA_RESULT_DIR)/*.json
	@echo "Models cleaned."

.PHONY: clean_results
clean_results:
	@echo "Cleaning results..."
	rm -rf $(SCORE_RESULTS_DIR)/* $(TEST_RESULT_DIR)/*
	@echo "Results cleaned."

.PHONY: clean_preprocessor
clean_preprocessor:
	@echo "Cleaning preprocessor objects..."
	rm -rf $(PREPROCESSOR_DIR)/*
	@echo "Preprocessor objects cleaned."

.PHONY: clean_log
clean_log:
	@echo "Cleaning logs..."
	rm -rf $(LOG_DIR)/*
	@echo "Logs cleaned."

.PHONY: clean_bento
clean_bento:
	@echo "Cleaning up BentoML files..."
	rm -rf $(BENTOML_HOME)/*
	@echo "BentoML files cleaned."

.PHONY: clean_mlflow
clean_mlflow:
	@echo "Cleaning up MlFlow files..."
	rm -rf mlruns/
	@echo "MlFlow files cleaned."

.PHONY: clean_mlflow_deep
clean_mlflow_deep:
	@echo "Cleaning up MlFlow files..."
	rm -rf Model/mlruns/*
	@echo "MlFlow files cleaned."
