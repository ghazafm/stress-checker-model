# Paths
DATA_DIR = Data/raw
MODEL_DIR = Model/model
SCORE_RESULTS_DIR = Result/scores
TEST_RESULT_DIR = Result/test
METADATA_RESULT_DIR = Model/metadata
PREPROCESSOR_DIR = Model/preprocessor
NEW_DATA_DIR = Data/clean
PREDICT_DATA_DIR = Result/predict
DEPLOYED_MODEL_FILE = $(MODEL_DIR)/deploy_model_path.txt
COLUMN_TO_REMOVE = nothin
TARGET_COL = stress_level
ID_COL = PassengerId
RANDOM_STATE = 42
DEPLOYED_MODEL=$(shell cat $(DEPLOYED_MODEL_FILE))

TIMESTAMP_FILE = timestamp.txt
LATEST_MODEL = $(shell ls -t $(MODEL_DIR)/*.pkl | head -n 1)

VENV_DIR = myenv
CONDA_ENV = myenv/
VENV_ACTIVATE = source $(VENV_DIR)/bin/activate
PYTHON = $(VENV_DIR)/bin/python
REQUIREMENTS_FILE = requirements.txt
ENVIRONMENT_FILE = environment.yml

MLFLOW_PORT = 8888
MLFLOW_HOST = 0.0.0.0
ARTIFACT_ROOT = Model/mlruns

MODEL_NAME ?= xgboost
PARAMS ?= 

BENTO_SERVICE = script.service
BENTO_MODEL = stress_checker

export BENTOML_HOME = ./Model/bentoml

# Default target
.PHONY: all
all: help

# Display help for the Makefile
.PHONY: help
help:
	@echo "Usage:"
	@echo "  make conda_env   			: Create and activate the Conda environment."
	@echo "  make venv_env    			: Create and activate the Python virtual environment using venv."
	@echo "  make data      			: Run data collection and preparation steps."
	@echo "  make train     			: Train the machine learning model."
	@echo "  make evaluate  			: Evaluate the trained model."
	@echo "  make deploy    			: Save and deploy the trained model."
	@echo "  make predict   			: Run predictions on new data."
	@echo "  make clean     			: Clean data, model, and result directories."
	@echo "  make clean_models     		: Clean only the trained models."
	@echo "  make clean_results    		: Clean only the prediction and score results."
	@echo "  make clean_preprocessor	: Clean only the preprocessor objects."

# Check if environment exists, or prompt to create one
.PHONY: check_env
check_env:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Environment not found. Please create one with 'make conda_env' or 'make venv_env'."; \
		exit 1; \
	fi

# Create a Conda environment with Python 3.10 and install requirements
.PHONY: conda_env
conda_env:
	@echo "Checking for Conda..."
	@which conda >/dev/null 2>&1 || { echo >&2 "Conda is not installed. Please install Conda."; exit 1; }
	@echo "Creating Conda environment..."
	conda env create -f $(ENVIRONMENT_FILE) -p $(VENV_DIR) -y
	@echo "Environment created using $(ENVIRONMENT_FILE)."
	
.PHONY: venv_env
venv_env:
	@echo "Creating Python virtual environment using venv..."
	python3 -m venv $(VENV_DIR)
	@echo "Environment created at $(VENV_DIR)."
	@echo "Activating environment and upgrading pip..."
	$(VENV_ACTIVATE) && pip install --upgrade pip
	@echo "Installing packages from $(REQUIREMENTS_FILE)..."
	$(VENV_ACTIVATE) && pip install -r $(REQUIREMENTS_FILE)
	@echo "Packages installed in venv environment."

# Generate a timestamp and save it to a file
.PHONY: timestamp
timestamp:
	@echo "Generating timestamp..."
	@date +"%Y%m%d_%H%M%S" > $(TIMESTAMP_FILE)
	@echo "Timestamp saved to $(TIMESTAMP_FILE)."

# Data collection and preparation (checks if env exists)
.PHONY: data
data: check_env timestamp
	@echo
	@echo "Collecting and preparing data..."
	$(PYTHON) script/data_preparation.py --data_dir $(DATA_DIR) --data_new $(NEW_DATA_DIR) --output_dir $(PREPROCESSOR_DIR) --target_col $(TARGET_COL) --random_state $(RANDOM_STATE) --columns_to_remove $(COLUMN_TO_REMOVE) --timestamp $(shell cat $(TIMESTAMP_FILE))
	@echo "Data preparation completed."

# Model training
.PHONY: train
train:
	@echo
	@echo "Training the machine learning model with model: $(MODEL_NAME) and params: $(PARAMS)"
	$(PYTHON) script/train_model.py --data_dir $(NEW_DATA_DIR) --model_dir $(MODEL_DIR) --timestamp $(shell cat $(TIMESTAMP_FILE)) --model_name $(MODEL_NAME) --params '$(PARAMS)'
	@echo "Model training completed."

# Model evaluation
.PHONY: evaluate
evaluate:
	@echo
	@echo "Evaluating the trained model..."
	$(PYTHON) script/evaluate_model.py --model $(LATEST_MODEL) --results_dir $(SCORE_RESULTS_DIR) --data_dir $(NEW_DATA_DIR) --timestamp $(shell cat $(TIMESTAMP_FILE))
	@echo "Model evaluation completed."

# # Model deployment (saving the model)
# .PHONY: deploy
# deploy:
# 	@echo
# 	@echo "Deploying the trained model..."
# 	$(PYTHON) script/deploy_model.py --model_path $(LATEST_MODEL) --model_dir $(MODEL_DIR) --metadata_dir $(METADATA_RESULT_DIR) > $(DEPLOYED_MODEL_FILE) --timestamp $(shell cat $(TIMESTAMP_FILE))
# 	@echo "Model has been saved and deployed. Model path stored in $(DEPLOYED_MODEL_FILE)."

.PHONY: mlflow
mlflow: check-env
	@echo "Running MLflow server on http://$(MLFLOW_HOST):$(MLFLOW_PORT)"
	@if [ "$(USE_VENV)" = "true" ]; then \
		$(VENV_ACTIVATE) && \
		mlflow server --host $(MLFLOW_HOST) --port $(MLFLOW_PORT) \
		--backend-store-uri $(ARTIFACT_ROOT); \
	else \
		source $(shell conda info --base)/etc/profile.d/conda.sh && \
		conda activate $(CONDA_ENV) && \
		mlflow server --host $(MLFLOW_HOST) --port $(MLFLOW_PORT) \
		--backend-store-uri $(ARTIFACT_ROOT); \
	fi

# Check whether to use venv or conda
.PHONY: check_env
check-env:
	@if [ -d "$(VENV_DIR)" ]; then \
		echo "Using venv environment"; \
		USE_VENV=true; \
	elif conda info --envs | grep -q "$(CONDA_ENV)"; then \
		echo "Using conda environment"; \
		USE_VENV=false; \
	else \
		echo "No virtual environment found. Please set up venv or conda."; \
		exit 1; \
	fi

# Prediction on new data using the deployed model
# .PHONY: predict
# predict: deploy
# 	@echo
# 	@echo "Running predictions on new data using the deployed model..."
# 	@echo "Using deployed model: $(DEPLOYED_MODEL)"
# 	$(PYTHON) script/predict_data.py --model $(DEPLOYED_MODEL) --preprocessor $(PREPROCESSOR_DIR) --data_dir $(DATA_DIR) --predict_dir $(PREDICT_DATA_DIR) --timestamp $(shell cat $(TIMESTAMP_FILE)) --id_col $(ID_COL)
# 	@echo "Predictions saved."


# Command to run import.py and deploy with BentoML
.PHONY: serve
serve: import_bentoml start_bentoml

# Step 1: Import the model into BentoML using import.py
.PHONY: import_bentoml
import_bentoml:
	@echo "Importing model into BentoML..."
	$(PYTHON) script/import.py -m $(BENTO_MODEL)
	@echo "Model import completed."

# Step 2: Start BentoML service after importing the model
.PHONY: start_bentoml
start_bentoml: check_env
	@echo "Starting BentoML service with auto-reload..."
	@if [ "$(USE_VENV)" = "true" ]; then \
		echo "Using venv environment"; \
		$(VENV_ACTIVATE) && bentoml serve $(BENTO_SERVICE) --reload; \
	else \
		echo "Using conda environment"; \
		source $(shell conda info --base)/etc/profile.d/conda.sh && \
		conda activate $(CONDA_ENV) && \
		bentoml serve $(BENTO_SERVICE) --reload; \
	fi
	@echo "BentoML service started."

.PHONY: build
build: check_env
	@echo "Making BentoML bentos & Docker image"
	@if [ "$(USE_VENV)" = "true" ]; then \
		echo "Using venv environment"; \
		echo "building...."; \
		$(VENV_ACTIVATE) && bentoml build -f ./Model/bentofile.yaml;\
	else \
		echo "Using conda environment"; \
		echo "building...."; \
		source $(shell conda info --base)/etc/profile.d/conda.sh && \
		conda activate $(CONDA_ENV) && \
		bentoml build -f ./Model/bentofile.yaml; \
	fi
	@echo "BentoML bentos & Docker image builded."

.PHONY: containerize
containerize:
	@echo "Making BentoML image"
	@if [ "$(USE_VENV)" = "true" ]; then \
		echo "Using venv environment"; \
		echo "building...."; \
		$(VENV_ACTIVATE) && bentoml containerize stress_checker:latest; \
	else \
		echo "Using conda environment"; \
		echo "building...."; \
		source $(shell conda info --base)/etc/profile.d/conda.sh && \
		conda activate $(CONDA_ENV) && \
		bentoml containerize stress_checker:latest; \
	fi
	@echo "Docker image builded."

.PHONY: deploy
deploy:
	@echo "Running BentoML container..."
	@if docker run -it --rm -p 3000:3000 stress_checker:latest serve; then \
		echo "BentoML container is running on port 3000."; \
	else \
		echo "Error: Failed to run the BentoML container." >&2; \
		exit 1; \
	fi

# Clean all data, models, and results
.PHONY: clean
clean: clean_data clean_models clean_results clean_preprocessor
	@echo
	@echo "Complete cleanup completed."

# Clean all data, models, and results, (Log)
.PHONY: clean_all
clean_all: clean clean_log
	@echo
	@echo "Complete cleanup completed."

# Clean only data
.PHONY: clean_data
clean_data:
	@echo
	@echo "Cleaning up data..."
	rm -rf $(NEW_DATA_DIR)/*.csv
	rm -rf $(PREDICT_DATA_DIR)/*.csv
	@echo "Data cleaned."

# Clean only models
.PHONY: clean_models
clean_models:
	@echo
	@echo "Cleaning up models..."
	rm -rf $(MODEL_DIR)/*.pkl $(MODEL_DIR)/$(DEPLOYED_MODEL_FILE)
	rm -rf $(METADATA_RESULT_DIR)/*.json
	@echo "Models cleaned."

# Clean only results (predictions and scores)
.PHONY: clean_results
clean_results:
	@echo
	@echo "Cleaning up results (scores and predictions)..."
	rm -rf $(SCORE_RESULTS_DIR)/* $(TEST_RESULT_DIR)/*
	@echo "Results cleaned."

# Clean only preprocessor objects
.PHONY: clean_preprocessor
clean_preprocessor:
	@echo
	@echo "Cleaning up preprocessor objects..."
	rm -rf $(PREPROCESSOR_DIR)/*
	@echo "Preprocessor cleaned."

.PHONY: clean_log
clean_log:
	@echo
	@echo "Cleaning up preprocessor objects..."
	rm -rf Log/*
	@echo "Preprocessor cleaned."

