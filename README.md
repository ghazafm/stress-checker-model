# Stress Checker Machine Learning Project

## 📚 Table of Contents
- [📝 Project Overview](#overview)
- [📁 Directory Structure](#directory-structure)
- [⚙️ Setup and Installation](#setup-and-installation)
  - [🐍 Conda Environment Setup](#conda-environment-setup)
  - [🔧 Virtual Environment (venv) Setup](#virtual-environment-setup)
- [📊 Data Workflow](#data-workflow)
  - [📂 Data Collection and Preprocessing](#data-collection-and-preprocessing)
- [🧠 Modeling and Deployment](#modeling-and-deployment)
  - [🚀 Model Training](#model-training)
  - [📈 Model Evaluation](#model-evaluation)
  - [📦 Model Deployment](#model-deployment)
- [🔬 Experiment Tracking with MLFlow](#experiment-tracking-with-mlflow)
- [🌐 Model Serving with BentoML](#model-serving-with-bentoml)
  - [🖥️ Serving via HTTP](#serving-via-http)
  - [🔗 Serving via gRPC](#serving-via-grpc)
- [🛠️ Building and Containerizing](#building-and-containerizing)
  - [🏗️ Build BentoML Bentos](#build-bentoml-bentos)
  - [🐳 Containerize BentoML Service](#containerize-bentoml-service)
- [🧹 Cleaning Up](#cleaning-up)
  - [🧽 Full Cleanup](#full-cleanup)
  - [🧼 Deep Cleanup](#deep-cleanup)
- [📜 Commands Reference](#commands-reference)
- [👥 Contributing](#contributing)
- [📧 Contact and Support](#contact-and-support)
- [📄 License](#license)

---

## Overview
This project builds a machine learning model to predict stress levels based on various health, academic, and social factors. The workflow includes data collection, preprocessing, model training, evaluation, and deployment. It also supports serving models through BentoML, with full tracking using MLFlow.

## Project Structure
```
.
├── Data/
│   ├── raw/              # Raw data directory
│   ├── clean/            # Cleaned data after preprocessing
├── Model/
│   ├── model/            # Saved trained models
│   ├── preprocessor/      # Preprocessor objects (scalers, encoders, etc.)
│   ├── bentoml/          # BentoML directory for saved Bentos
├── Result/
│   ├── scores/           # Model evaluation results
│   ├── test/             # Test results
│   ├── predict/          # Predictions from the deployed model
├── Log/                  # Log files
├── script/               # Python scripts for training, evaluation, and serving
├── environment.yml       # Conda environment configuration
├── requirements.txt      # Python packages for virtual environments
├── Makefile              # Automation of tasks (training, serving, etc.)
└── README.md             # Project documentation
```

---

## Environment Setup
You can set up the environment using either Conda or Python's virtual environment (`venv`). Follow the instructions below to create and activate the environment.

### Conda Setup
```bash
make conda_env
```
This will create a Conda environment based on the `environment.yml` file.

### Virtual Environment Setup (venv)
```bash
make venv_env
```
This will create a virtual environment using `venv` and install the required dependencies from `requirements.txt`.

---

## Data Processing
Before training the model, you need to process the raw data. Use the following command to clean and prepare the data for training:
```bash
make data
```
This will run the data preprocessing script and store the cleaned data in `Data/clean`.

---

## Model Training, Evaluation, and Deployment

### Training the Model
To train the machine learning model, run:
```bash
make train MODEL_NAME=<model_name> PARAMS='<model_parameters>'
```
- `MODEL_NAME`: Specify the model type (e.g., `xgboost`, `random_forest`, `svr`, `ann`, etc.).
- `PARAMS`: Provide any additional model parameters in JSON format.

### Evaluating the Model
After training, evaluate the model's performance using:
```bash
make evaluate
```
This calculates metrics like Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-Squared (R²).

### Deploying the Model
Once the model is evaluated, deploy it for use in prediction tasks:
```bash
make deploy
```
The deployed model path is saved in the `deploy_model_path.txt` file within the `Model/model` directory.

---

## Running MLFlow
MLFlow is used for tracking experiments, storing models, and serving them. Start the MLFlow server using:
```bash
make mlflow
```
The server will run at `http://0.0.0.0:8888` and log everything to the `Model/mlruns` directory.

---

## Serving with BentoML
You can serve the model via BentoML either through HTTP or gRPC.

### Serving HTTP with BentoML
Make sure MLFlow is running, then serve the model using:
```bash
make serve
```

### Serving gRPC with BentoML
Serve the model via gRPC by running:
```bash
make serve_grpc
```

---

## Building and Containerizing with BentoML

### Build BentoML Bentos
To create BentoML bentos (the packaging format for models), run:
```bash
make build
```

### Containerizing BentoML
Create a Docker container for the BentoML service:
```bash
make containerize
```

### Deploying BentoML in a Container (HTTP/gRPC)
To run the BentoML service inside a Docker container for HTTP:
```bash
make deploy_bento
```
For gRPC:
```bash
make deploy_bento_grpc
```

---

## Cleaning Up
To clean various parts of the project, such as data, models, results, logs, and BentoML or MLFlow files:

### Clean All Files
```bash
make clean
```

### Perform Deep Clean
This performs a more thorough clean, removing BentoML and MLFlow files as well:
```bash
make deep_clean
```

---

## Complete Command List
Here is the complete list of available commands:

### Environment Setup
- `make conda_env` – Create and activate the Conda environment.
- `make venv_env` – Create and activate the Python virtual environment using `venv`.

### Data Processing
- `make data` – Run data collection and preparation steps.
- `make clean_data` – Clean processed data files.

### Model Training, Evaluation, and Deployment
- `make train` – Train the machine learning model.
- `make evaluate` – Evaluate the trained model.
- `make deploy` – Save and deploy the trained model.
- `make clean_models` – Clean trained model files.
- `make clean_results` – Clean model evaluation results.

### MLFlow Commands
- `make mlflow` – Start the MLFlow server.
- `make clean_mlflow` – Clean up MLFlow files.
- `make clean_mlflow_deep` – Perform a deep clean of all MLFlow files.

### BentoML Commands
- `make serve` – Start BentoML HTTP service.
- `make serve_grpc` – Start BentoML gRPC service.
- `make build` – Make BentoML Bentos.
- `make containerize` – Build Docker image for BentoML service.
- `make deploy_bento` – Run BentoML HTTP container.
- `make deploy_bento_grpc` – Run BentoML gRPC container.
- `make clean_bento` – Clean up BentoML files.

### Utility Commands
- `make timestamp` – Generate a timestamp for model tracking.
- `make clean_log` – Clean log files.

### Cleaning Up
- `make clean` – Clean all data, models, results, logs, BentoML, and MLflow files.
- `make deep_clean` – Perform a deep clean (data, models, results, BentoML, and MLflow).

---

## License
This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute this software, provided that the original authors are credited.

---

## Contact
For any questions or inquiries, please feel free to reach out:

- **Email**: contact@fauzanghaza.com
- **LinkedIn**: [LinkedIn Profile](https://www.linkedin.com/in/fauzanghaza)
- **GitHub**: [GitHub Profile](https://github.com/ghazafm)

---

## Contributions
Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push the branch (`git push origin feature-name`).
5. Open a pull request.

Please ensure your pull request adheres to the following guidelines:
- Code is properly documented.
- Code is tested and passes all tests.
- Follow the repository's coding style.

---

## Bug Reports and Feature Requests
If you encounter any issues or have suggestions for new features, please report them through the project's GitHub Issues page:

- [Submit a Bug Report](https://github.com/ghazafm/stress-checker-model/issues/new?template=bug_report.md)
- [Submit a Feature Request](https://github.com/ghazafm/stress-checker-model/issues/new?template=feature_request.md)

---

## Acknowledgments
This project makes use of several open-source tools and libraries, including but not limited to:

- [MLFlow](https://mlflow.org/)
- [BentoML](https://github.com/bentoml/BentoML)
- [scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.ai/)
- [LightGBM](https://github.com/microsoft/LightGBM)
- [TensorFlow/Keras](https://www.tensorflow.org/)

---

## Future Enhancements
In future iterations of this project, the following enhancements are planned:
- Adding support for additional model types (e.g., `CatBoost`, `ElasticNet`).
- Building a REST API for real-time predictions.
- Improving data preprocessing pipelines with more advanced feature engineering.
- Adding more advanced hyperparameter tuning options.
