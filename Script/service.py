import os
import logging
import pandas as pd
import pickle
import bentoml
import mlflow
import shap
import xgboost as xgb
import numpy as np
from bentoml.io import JSON
from bentoml.exceptions import BentoMLException

# Set BentoML directory for model storage
os.environ["BENTOML_HOME"] = "./Model/bentoml"
mlflow.set_tracking_uri("http://127.0.0.1:8888")

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict(X)

# Function to read the latest preprocessor path dynamically
def get_latest_preprocessor_path():
    try:
        # Path to the 'latest' file
        latest_file_path = './Model/bentoml/models/stress_checker_scaler/latest'
        
        # Read the content of the 'latest' file (which contains the folder name)
        with open(latest_file_path, 'r') as file:
            latest_version_folder = file.read().strip()
        
        # Full path to the latest version folder
        latest_dir = os.path.join('./Model/bentoml/models/stress_checker_scaler', latest_version_folder)
        
        logger.info(f"Latest preprocessor found in folder: {latest_dir}")
        
        # The scaler file inside the latest directory (e.g., 'saved_model.pkl')
        return os.path.join(latest_dir, 'saved_model.pkl')
    
    except Exception as e:
        logger.error(f"Error finding the latest preprocessor: {e}")
        raise

# Load the BentoML model
try:
    model_ref = bentoml.mlflow.get("stress_checker:latest")
    model_runner = model_ref.to_runner()
except BentoMLException as e:
    logger.error(f"Error loading model from BentoML: {e}")
    raise

# Load the scaler manually using the dynamically resolved path
try:
    scaler_path = get_latest_preprocessor_path()
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    logger.info("Scaler loaded successfully from pickle.")
except FileNotFoundError as e:
    logger.error(f"Scaler pickle file not found: {e}")
    raise
except Exception as e:
    logger.error(f"Error loading scaler from pickle: {e}")
    raise

# Define the BentoML service with the model runner
svc = bentoml.Service('stress_checker', runners=[model_runner])

# Define the API to accept JSON input and return JSON output
@svc.api(input=JSON(), output=JSON())
async def classify(input_data):
    try:
        # Log the input data for debugging purposes
        logging.info(f"Received input data: {input_data}")

        # Convert the input JSON data into a DataFrame
        input_df = pd.DataFrame([input_data])

        # Cast all input data to float64 to match the model schema
        input_df = input_df.astype('float32')

        # Validate the input data
        validate_input_data(input_df)

        # Scale the input data using the manually loaded scaler
        input_scaled = scaler.transform(input_df)
        
        input_scaled = pd.DataFrame(input_scaled, columns=input_df.columns)

        # Use the model to make predictions
        result = await model_runner.async_run(input_scaled)
        
        # Log the prediction result for debugging
        logging.info(f"Model prediction result: {result}")
        
        logged_model = 'models:/stress_checker/latest'
        model, flavor = load_model_based_on_flavor(logged_model)
        
        if "keras" in flavor:
            explainer = shap.Explainer(model, input_scaled)
        else:
            explainer = shap.Explainer(model)
        shap_values = explainer(input_df)
        
        recomendation = generate_shap_recommendations(input_scaled, shap_values)

        # Return the result as a JSON response
        return {
            "predictions": result.tolist(),
            'text': recomendation,
            }

    except Exception as e:
        logging.error(f"Error in classify function: {str(e)}")
        return {
            "error": str(e),
            "input": input_data,
            "flavor": flavor
            
        }

def validate_input_data(input_df):
    """
    Validates the input data to ensure it has the expected structure and columns.
    """
    # List of expected feature columns based on your dataset
    expected_columns = [
        "anxiety_level", "self_esteem", "mental_health_history", "depression", "headache",
        "blood_pressure", "sleep_quality", "breathing_problem", "noise_level", "living_conditions",
        "basic_needs", "academic_performance", "study_load", "teacher_student_relationship",
        "future_career_concerns", "social_support", "peer_pressure", "extracurricular_activities",
        "bullying"
    ]

    # Check if all expected columns are present in the input data
    missing_columns = [col for col in expected_columns if col not in input_df.columns]

    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    # Validate numeric columns
    for col in expected_columns:
        if not pd.api.types.is_numeric_dtype(input_df[col]):
            raise ValueError(f"Column {col} should be numeric.")

    # Example: Validate that values are within a plausible range (e.g., 0-100 for anxiety_level)
    if (input_df["anxiety_level"] < 0).any() or (input_df["anxiety_level"] > 100).any():
        raise ValueError("anxiety_level values should be between 0 and 100.")


# Function to load a model based on its flavor
def load_model_based_on_flavor(model_uri):
    # Get the model's flavor information
    model_info = mlflow.models.get_model_info(model_uri).flavors

    # Check if it's an XGBoost model
    if "xgboost" in model_info:
        print(f"Loading XGBoost model from {model_uri}")
        return mlflow.xgboost.load_model(model_uri), model_info
    # Check if it's a Scikit-learn model
    elif "sklearn" in model_info:
        print(f"Loading Scikit-learn model from {model_uri}")
        return mlflow.sklearn.load_model(model_uri), model_info
    # Check if it's a LightGBM model
    elif "lightgbm" in model_info:
        print(f"Loading LightGBM model from {model_uri}")
        return mlflow.lightgbm.load_model(model_uri), model_info
    # Check if it's a Keras model (for ANN)
    elif "keras" in model_info:
        print(f"Loading Keras model from {model_uri}")
        return mlflow.keras.load_model(model_uri), model_info
    # Check if it's a PyFunc model (generic ML models)
    elif "python_function" in model_info:
        print(f"Loading PyFunc model from {model_uri}")
        return mlflow.pyfunc.load_model(model_uri), model_info
    # Add support for more model flavors, like CatBoost, ONNX, etc.
    elif "catboost" in model_info:
        print(f"Loading CatBoost model from {model_uri}")
        return mlflow.catboost.load_model(model_uri), model_info
    elif "onnx" in model_info:
        print(f"Loading ONNX model from {model_uri}")
        return mlflow.onnx.load_model(model_uri), model_info
    else:
        raise ValueError(f"Model type {model_info} not supported or unknown flavor.")
    
    
def generate_shap_recommendations(input_df, shap_values):
    """
    Generate SHAP explanations and recommendations for the input data.
    """
    try:
        # # Assume that we already know the feature names
        feature_names = input_df.columns

        # # Calculate feature importance (mean absolute SHAP values)
        shap_values_array = shap_values.values
        feature_importance = np.abs(shap_values_array).mean(axis=0)
        sorted_importance_indices = np.argsort(feature_importance)[::-1]
        # print(model)

        # Generate recommendations based on top contributing features
        recommendations = recommend_based_on_shap(sorted_importance_indices, feature_names, input_df)
        # recommendations = feature_importance
        return recommendations
    except Exception as e:
        logger.error(f"Error during SHAP explanation generation: {e}")
        raise


def recommend_based_on_shap(sorted_importance_indices, feature_names, input_df):
    """
    Generate personalized recommendations based on the top 1-2 most important features
    as explained by SHAP values, combined into a single recommendation if both are highly relevant.
    """
    recommendations = []
    
    # Extract the top 2 important features based on SHAP values
    top_features = feature_names[sorted_importance_indices[:2]]  # Limit to the top 2 features
    
    # Get feature values for the top 2 features
    feature_1 = top_features[0]
    feature_2 = top_features[1] if len(top_features) > 1 else None  # Check if we have a second feature

    feature_1_value = input_df[feature_1].values[0]
    feature_2_value = input_df[feature_2].values[0] if feature_2 else None

    # Combined recommendations logic for various feature combinations
    if feature_1 == "anxiety_level" and feature_2 == "self_esteem":
        if feature_1_value > 85 and feature_2_value < 30:
            recommendations.append(
                "Your anxiety level is critically high and your self-esteem is very low. "
                "Immediate steps are necessary: reach out to a mental health professional, "
                "engage in deep relaxation techniques, and focus on building self-esteem through "
                "positive affirmations and small, achievable goals. It's important to avoid overwhelming situations "
                "and seek supportive relationships to boost your confidence."
            )
        elif 60 <= feature_1_value <= 85 and 30 <= feature_2_value <= 50:
            recommendations.append(
                "Your anxiety level is elevated and your self-esteem is below average. "
                "Consider practicing mindfulness and relaxation techniques to manage anxiety. "
                "At the same time, engage in activities that boost self-esteem, like volunteering, learning new skills, "
                "and focusing on your strengths."
            )
        else:
            recommendations.extend(single_feature_recommendations(feature_1, feature_1_value))
            recommendations.extend(single_feature_recommendations(feature_2, feature_2_value))

    elif feature_1 == "sleep_quality" and feature_2 == "anxiety_level":
        if feature_1_value < 40 and feature_2_value > 70:
            recommendations.append(
                "Your sleep quality is poor and your anxiety level is high. This combination can greatly affect your well-being. "
                "Improving sleep hygiene (e.g., creating a sleep routine, avoiding screens before bed) can help lower anxiety. "
                "Also, consider relaxation techniques like meditation to manage both anxiety and sleep issues."
            )
        else:
            recommendations.extend(single_feature_recommendations(feature_1, feature_1_value))
            recommendations.extend(single_feature_recommendations(feature_2, feature_2_value))

    elif feature_1 == "blood_pressure" and feature_2 == "anxiety_level":
        if feature_1_value > 140 and feature_2_value > 70:
            recommendations.append(
                "Your blood pressure and anxiety levels are both critically high. It's important to consult a healthcare provider. "
                "Try stress reduction techniques like deep breathing and meditation, and follow a heart-healthy lifestyle with reduced salt intake and regular exercise."
            )
        else:
            recommendations.extend(single_feature_recommendations(feature_1, feature_1_value))
            recommendations.extend(single_feature_recommendations(feature_2, feature_2_value))

    elif feature_1 == "depression" and feature_2 == "social_support":
        if feature_1_value > 70 and feature_2_value < 40:
            recommendations.append(
                "Your depression level is high, and your social support is low. Consider reaching out to a trusted friend or professional "
                "for support. Building a social network can help alleviate some of the burdens of depression."
            )
        else:
            recommendations.extend(single_feature_recommendations(feature_1, feature_1_value))
            recommendations.extend(single_feature_recommendations(feature_2, feature_2_value))

    elif feature_1 == "noise_level" and feature_2 == "sleep_quality":
        if feature_1_value > 70 and feature_2_value < 40:
            recommendations.append(
                "The noise level in your environment is high, and your sleep quality is poor. Consider reducing noise distractions using "
                "earplugs or noise-canceling headphones, and create a restful sleep environment with white noise or calming sounds."
            )
        else:
            recommendations.extend(single_feature_recommendations(feature_1, feature_1_value))
            recommendations.extend(single_feature_recommendations(feature_2, feature_2_value))

    # If no specific combination logic applies, just return the top single feature recommendation
    if not recommendations:
        recommendations.extend(single_feature_recommendations(feature_1, feature_1_value))
        if feature_2:
            recommendations.extend(single_feature_recommendations(feature_2, feature_2_value))

    return recommendations


def single_feature_recommendations(feature, feature_value):
    """
    Generate a recommendation for a single feature based on its value.
    This function can be extended for each feature to provide specific advice.
    """
    recommendations = []
    
    if feature == "anxiety_level":
        if feature_value > 85:
            recommendations.append(
                "Your anxiety level is critically high. Immediate steps are needed: reach out to a mental health professional, "
                "consider medication, and engage in deep relaxation techniques like meditation or progressive muscle relaxation."
            )
        elif 60 <= feature_value <= 85:
            recommendations.append(
                "Your anxiety level is elevated. Consider implementing stress-management techniques such as journaling, "
                "physical exercise, and reducing caffeine intake."
            )
        elif 40 <= feature_value < 60:
            recommendations.append(
                "Your anxiety level is moderate. Maintain a routine that includes balanced nutrition, regular physical activity, "
                "and mindfulness exercises."
            )
        else:
            recommendations.append(
                "Your anxiety level is low. Keep following your current routine and mental health practices."
            )

    elif feature == "self_esteem":
        if feature_value < 30:
            recommendations.append(
                "Your self-esteem is very low. Seek support from a counselor or therapist to work on improving self-perception. "
                "Engage in daily affirmations and surround yourself with positive influences."
            )
        elif 30 <= feature_value <= 50:
            recommendations.append(
                "Your self-esteem is below average. Focus on self-compassion and avoid comparing yourself to others."
            )
        elif 50 < feature_value <= 70:
            recommendations.append(
                "Your self-esteem is moderate. Continue with practices that reinforce positive self-image, such as setting realistic goals."
            )
        else:
            recommendations.append(
                "Your self-esteem is high. Keep nurturing this positive mindset by consistently focusing on your strengths."
            )

    # Add more cases for other features if needed...
    
    return recommendations