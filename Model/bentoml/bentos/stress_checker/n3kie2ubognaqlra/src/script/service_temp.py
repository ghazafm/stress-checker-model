import bentoml
import shap
import pandas as pd
import numpy as np
import os
import logging
from bentoml.io import JSON
from bentoml.exceptions import BentoMLException

# Set BentoML directory for model storage
os.environ["BENTOML_HOME"] = "./Model/bentoml"

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the saved model from BentoML's model store
try:
    model_ref = bentoml.mlflow.get("stress_checker:latest")
    model_runner = model_ref.to_runner()
except BentoMLException as e:
    logger.error(f"Error loading model from BentoML: {e}")
    raise

try:
    scaler_ref = bentoml.picklable_model.get("stress_checker_scaler:latest")
    scaler = scaler_ref.to_runner()
except BentoMLException as e:
    logger.error(f"Error loading preprocessor from BentoML: {e}")
    raise


# Define a BentoML service
svc = bentoml.Service("stress_checker", runners=[model_runner, scaler])

# Define the prediction API route
@svc.api(input=JSON(), output=JSON())
async def predict(input_data):
    try:
        # Validate and convert the input JSON data to a DataFrame
        # input_df = pd.read_json([input_data])
        input_df = pd.DataFrame([input_data])
        input_df = input_df.astype('float64')
        validate_input_data(input_df)
        
        # Preprocess the input data using the loaded StandardScaler
        input_scaled = await scaler.async_run(input_df)

        # Make a prediction using the model
        predictions = await model_runner.async_run(input_scaled)

        # Generate SHAP explanations for the input data
        shap_values, recommendations = generate_shap_recommendations(input, model_runner)

        # Return both predictions and recommendations
        return {
            "predictions": predictions.tolist(),
            "recommendations": recommendations,
            "important_cause": shap_values
        }

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return {"error": str(e)}


def generate_shap_recommendations(input_df, model_runner):
    """
    Generate SHAP explanations and recommendations for the input data.
    """
    try:
        # Assume that we already know the feature names
        feature_names = input_df.columns

        # Initialize SHAP explainer for tree-based models
        model = model_ref.to_runner()._model
        explainer = shap.TreeExplainer(model)

        # Calculate SHAP values for the input data
        shap_values = explainer.shap_values(input_df)

        # Calculate feature importance (mean absolute SHAP values)
        feature_importance = np.abs(shap_values).mean(axis=0)
        sorted_importance_indices = np.argsort(feature_importance)[::-1]

        # Generate recommendations based on top contributing features
        recommendations = recommend_based_on_shap(sorted_importance_indices, feature_names, input_df)

        return shap_values, recommendations
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

    # Additional validations can be added as needed (e.g., checking ranges, data types)
    # Example: Validate numeric columns
    for col in expected_columns:
        if not pd.api.types.is_numeric_dtype(input_df[col]):
            raise ValueError(f"Column {col} should be numeric.")

    # Example: Validate that values are within a plausible range (e.g., 0-10 for anxiety_level)
    if (input_df["anxiety_level"] < 0).any() or (input_df["anxiety_level"] > 100).any():
        raise ValueError("anxiety_level values should be between 0 and 100.")

    # Repeat similar checks for other columns based on domain knowledge

