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
        latest_file_path = "./Model/bentoml/models/stress_checker_scaler/latest"

        # Read the content of the 'latest' file (which contains the folder name)
        with open(latest_file_path, "r") as file:
            latest_version_folder = file.read().strip()

        # Full path to the latest version folder
        latest_dir = os.path.join(
            "./Model/bentoml/models/stress_checker_scaler", latest_version_folder
        )

        logger.info(f"Latest preprocessor found in folder: {latest_dir}")

        # The scaler file inside the latest directory (e.g., 'saved_model.pkl')
        return os.path.join(latest_dir, "saved_model.pkl")

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
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    logger.info("Scaler loaded successfully from pickle.")
except FileNotFoundError as e:
    logger.error(f"Scaler pickle file not found: {e}")
    raise
except Exception as e:
    logger.error(f"Error loading scaler from pickle: {e}")
    raise

# Define the BentoML service with the model runner
svc = bentoml.Service("stress_checker", runners=[model_runner])


# Define the API to accept JSON input and return JSON output
@svc.api(input=JSON(), output=JSON())
async def classify(input_data):
    try:
        # Log the input data for debugging purposes
        logging.info(f"Received input data: {input_data}")

        # Convert the input JSON data into a DataFrame
        input_df = pd.DataFrame([input_data])

        # Cast all input data to float64 to match the model schema
        input_df = input_df.astype("float32")

        # Validate the input data
        validate_input_data(input_df)

        # Scale the input data using the manually loaded scaler
        input_scaled = scaler.transform(input_df)

        input_scaled = pd.DataFrame(input_scaled, columns=input_df.columns)

        # Use the model to make predictions
        result = await model_runner.async_run(input_scaled)
        result = np.clip(result, 0, 100)

        # Log the prediction result for debugging
        logging.info(f"Model prediction result: {result}")

        logged_model = "models:/stress_checker/latest"
        model, flavor = load_model_based_on_flavor(logged_model)

        if "keras" in flavor:
            explainer = shap.Explainer(model, input_scaled)
        else:
            explainer = shap.Explainer(model)
        shap_values = explainer(input_scaled)

        recomendation, feature_1, feature_2, feature_1_value, feature_2_value = (
            generate_shap_recommendations(input_df, shap_values)
        )

        # Return the result as a JSON response
        return {
            "predictions": result.tolist(),
            "text": recomendation,
            "feature_1": feature_1,
            "feature_2": feature_2,
            "feature_1_value": feature_1_value,
            "feature_2_value": feature_2_value,
        }

    except Exception as e:
        logging.error(f"Error in classify function: {str(e)}")
        return {"error": str(e), "input": input_data, "flavor": flavor}


def validate_input_data(input_df):
    """
    Validates the input data to ensure it has the expected structure and columns.
    """
    # List of expected feature columns based on your dataset
    expected_columns = [
        "anxiety_level",
        "self_esteem",
        "mental_health_history",
        "depression",
        "headache",
        "blood_pressure",
        "sleep_quality",
        "breathing_problem",
        "noise_level",
        "living_conditions",
        "basic_needs",
        "academic_performance",
        "study_load",
        "teacher_student_relationship",
        "future_career_concerns",
        "social_support",
        "peer_pressure",
        "extracurricular_activities",
        "bullying",
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
        recommendations, feature_1, feature_2, feature_1_value, feature_2_value = (
            recommend_based_on_shap(sorted_importance_indices, feature_names, input_df)
        )
        # recommendations = feature_importance
        return recommendations, feature_1, feature_2, feature_1_value, feature_2_value
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
    top_features = feature_names[
        sorted_importance_indices[:2]
    ]  # Limit to the top 2 features

    # Get feature values for the top 2 features
    feature_1 = top_features[0]
    feature_2 = (
        top_features[1] if len(top_features) > 1 else None
    )  # Check if we have a second feature

    feature_1_value = input_df[feature_1].values[0]
    feature_2_value = input_df[feature_2].values[0] if feature_2 else None

    # Combined recommendations logic for various feature combinations
    if feature_1 == "anxiety_level" and feature_2 == "self_esteem":
        if feature_1_value > 18 and feature_2_value < 10:
            recommendations.append(
                "Your anxiety level is critically high and your self-esteem is very low. "
                "Immediate steps are necessary: reach out to a mental health professional, "
                "engage in deep relaxation techniques, and focus on building self-esteem through "
                "positive affirmations and small, achievable goals."
            )
        elif 14 <= feature_1_value <= 18 and 10 <= feature_2_value <= 15:
            recommendations.append(
                "Your anxiety level is elevated and your self-esteem is below average. "
                "Consider practicing mindfulness and relaxation techniques to manage anxiety, and "
                "engage in activities that boost self-esteem, like volunteering or learning new skills."
            )
        else:
            recommendations.extend(
                single_feature_recommendations(feature_1, feature_1_value)
            )
            recommendations.extend(
                single_feature_recommendations(feature_2, feature_2_value)
            )

    elif feature_1 == "sleep_quality" and feature_2 == "anxiety_level":
        if feature_1_value < 2 and feature_2_value > 14:
            recommendations.append(
                "Your sleep quality is poor and your anxiety level is high. This combination can greatly affect your well-being. "
                "Improving sleep hygiene (e.g., creating a sleep routine, avoiding screens before bed) can help lower anxiety. "
                "Also, consider relaxation techniques like meditation to manage both anxiety and sleep issues."
            )
        else:
            recommendations.extend(
                single_feature_recommendations(feature_1, feature_1_value)
            )
            recommendations.extend(
                single_feature_recommendations(feature_2, feature_2_value)
            )

    elif feature_1 == "blood_pressure" and feature_2 == "anxiety_level":
        if feature_1_value > 3 and feature_2_value > 14:
            recommendations.append(
                "Your blood pressure and anxiety levels are both critically high. It's important to consult a healthcare provider. "
                "Try stress reduction techniques like deep breathing and meditation, and follow a heart-healthy lifestyle."
            )
        else:
            recommendations.extend(
                single_feature_recommendations(feature_1, feature_1_value)
            )
            recommendations.extend(
                single_feature_recommendations(feature_2, feature_2_value)
            )

    elif feature_1 == "depression" and feature_2 == "social_support":
        if feature_1_value > 20 and feature_2_value < 2:
            recommendations.append(
                "Your depression level is high, and your social support is low. Consider reaching out to a trusted friend or professional "
                "for support. Building a social network can help alleviate some of the burdens of depression."
            )
        else:
            recommendations.extend(
                single_feature_recommendations(feature_1, feature_1_value)
            )
            recommendations.extend(
                single_feature_recommendations(feature_2, feature_2_value)
            )

    elif feature_1 == "noise_level" and feature_2 == "sleep_quality":
        if feature_1_value > 4 and feature_2_value < 2:
            recommendations.append(
                "The noise level in your environment is high, and your sleep quality is poor. Consider reducing noise distractions using "
                "earplugs or noise-canceling headphones, and create a restful sleep environment with white noise or calming sounds."
            )
        else:
            recommendations.extend(
                single_feature_recommendations(feature_1, feature_1_value)
            )
            recommendations.extend(
                single_feature_recommendations(feature_2, feature_2_value)
            )

    elif feature_1 == "depression" and feature_2 == "future_career_concerns":
        if feature_1_value > 20 and feature_2_value > 3:
            recommendations.append(
                "Your depression and future career concerns are both high. Consider talking to a career counselor or mentor to explore career options "
                "while also seeking support for managing depression. Break career goals into smaller steps to reduce overwhelm."
            )
        else:
            recommendations.extend(
                single_feature_recommendations(feature_1, feature_1_value)
            )
            recommendations.extend(
                single_feature_recommendations(feature_2, feature_2_value)
            )

    elif feature_1 == "breathing_problem" and feature_2 == "anxiety_level":
        if feature_1_value > 4 and feature_2_value > 14:
            recommendations.append(
                "Your breathing problems and anxiety level are both elevated. Consider practicing deep breathing exercises, and consult with a healthcare provider to rule out any physical causes."
            )
        else:
            recommendations.extend(
                single_feature_recommendations(feature_1, feature_1_value)
            )
            recommendations.extend(
                single_feature_recommendations(feature_2, feature_2_value)
            )

    elif feature_1 == "academic_performance" and feature_2 == "study_load":
        if feature_1_value < 2 and feature_2_value > 4:
            recommendations.append(
                "Your academic performance is low, and your study load is high. It's important to reassess your study routine. "
                "Consider breaking down tasks into smaller chunks, seeking academic support, and balancing study time with relaxation."
            )
        else:
            recommendations.extend(
                single_feature_recommendations(feature_1, feature_1_value)
            )
            recommendations.extend(
                single_feature_recommendations(feature_2, feature_2_value)
            )

    elif feature_1 == "teacher_student_relationship" and feature_2 == "peer_pressure":
        if feature_1_value < 2 and feature_2_value > 4:
            recommendations.append(
                "Your relationship with teachers is strained, and you're experiencing high peer pressure. Consider seeking help from a counselor "
                "to work through any social dynamics and strengthen your relationship with mentors."
            )
        else:
            recommendations.extend(
                single_feature_recommendations(feature_1, feature_1_value)
            )
            recommendations.extend(
                single_feature_recommendations(feature_2, feature_2_value)
            )

    # Default to single feature recommendations if no specific combination logic applies
    if not recommendations:
        recommendations.extend(
            single_feature_recommendations(feature_1, feature_1_value)
        )
        if feature_2:
            recommendations.extend(
                single_feature_recommendations(feature_2, feature_2_value)
            )

    # If no recommendations were generated, return a default recommendation
    if not recommendations:
        recommendations.append(
            "Based on your inputs, it's important to monitor your mental and physical health. "
            "Consider consulting with a healthcare professional for personalized guidance."
        )

    return recommendations, feature_1, feature_2, feature_1_value, feature_2_value


def single_feature_recommendations(feature, feature_value):
    """
    Generate a recommendation for a single feature based on its value.
    This function can be extended for each feature to provide specific advice.
    """
    recommendations = []

    if feature == "anxiety_level":
        if feature_value > 18:
            recommendations.append(
                "Your anxiety level is critically high. Immediate steps are needed: reach out to a mental health professional, "
                "consider medication, and engage in deep relaxation techniques like meditation or progressive muscle relaxation."
            )
        elif 14 <= feature_value <= 18:
            recommendations.append(
                "Your anxiety level is elevated. Consider implementing stress-management techniques such as journaling, "
                "physical exercise, and reducing caffeine intake."
            )
        elif 8 <= feature_value < 14:
            recommendations.append(
                "Your anxiety level is moderate. Maintain a routine that includes balanced nutrition, regular physical activity, "
                "and mindfulness exercises."
            )
        else:
            recommendations.append(
                "Your anxiety level is low. Keep following your current routine and mental health practices."
            )

    elif feature == "self_esteem":
        if feature_value < 10:
            recommendations.append(
                "Your self-esteem is very low. Seek support from a counselor or therapist to work on improving self-perception. "
                "Engage in daily affirmations and surround yourself with positive influences."
            )
        elif 10 <= feature_value <= 15:
            recommendations.append(
                "Your self-esteem is below average. Focus on self-compassion and avoid comparing yourself to others."
            )
        elif 15 < feature_value <= 25:
            recommendations.append(
                "Your self-esteem is moderate. Continue with practices that reinforce positive self-image, such as setting realistic goals."
            )
        else:
            recommendations.append(
                "Your self-esteem is high. Keep nurturing this positive mindset by consistently focusing on your strengths."
            )

    elif feature == "mental_health_history":
        if feature_value == 1:
            recommendations.append(
                "You have a history of mental health issues. Continue seeking support from mental health professionals and maintain regular check-ups."
            )
        else:
            recommendations.append(
                "No reported mental health history. Continue maintaining a healthy lifestyle and monitor your mental well-being."
            )

    elif feature == "depression":
        if feature_value > 20:
            recommendations.append(
                "Your depression level is high. It's important to seek support from a mental health professional and explore therapies like CBT, "
                "mindfulness, or medication as needed."
            )
        elif 10 <= feature_value <= 20:
            recommendations.append(
                "Your depression level is moderate. Consider engaging in activities you enjoy and reaching out to a supportive social network."
            )
        else:
            recommendations.append(
                "Your depression level is low. Keep up with positive routines that maintain your emotional well-being."
            )

    elif feature == "headache":
        if feature_value > 3:
            recommendations.append(
                "Frequent headaches can be a sign of stress or other underlying issues. Consider consulting a healthcare provider for further evaluation."
            )
        else:
            recommendations.append(
                "Headaches seem to be infrequent. Continue managing your stress and hydration to minimize headaches."
            )

    elif feature == "blood_pressure":
        if feature_value > 3:
            recommendations.append(
                "Your blood pressure is high. Consider consulting a healthcare provider and following heart-healthy lifestyle practices like reducing salt intake and regular exercise."
            )
        else:
            recommendations.append(
                "Your blood pressure is within normal range. Continue maintaining a balanced diet and regular physical activity."
            )

    elif feature == "sleep_quality":
        if feature_value < 2:
            recommendations.append(
                "Your sleep quality is poor. Consider improving your sleep hygiene by maintaining a regular sleep schedule, reducing screen time before bed, and creating a restful sleep environment."
            )
        elif feature_value >= 2 and feature_value < 4:
            recommendations.append(
                "Your sleep quality is below average. Consider practicing relaxation techniques before bed, like meditation or breathing exercises."
            )
        else:
            recommendations.append(
                "Your sleep quality is good. Keep maintaining your current routine for a restful night's sleep."
            )

    elif feature == "breathing_problem":
        if feature_value > 4:
            recommendations.append(
                "You reported frequent and severe breathing problems. This could be a sign of a serious health issue such as asthma, "
                "allergies, or another respiratory condition. It's important to consult with a healthcare provider immediately to evaluate the cause. "
                "Avoid exposure to potential triggers like allergens, smoke, or pollutants, and consider doing light breathing exercises if recommended by your doctor."
            )
        elif 2 <= feature_value <= 4:
            recommendations.append(
                "You reported occasional breathing problems. This could be related to mild respiratory conditions, stress, or environmental factors. "
                "Consider practicing deep breathing exercises or yoga to improve lung capacity. If symptoms persist or worsen, consult with a healthcare professional."
            )
        else:
            recommendations.append(
                "Your breathing appears to be normal. Continue maintaining a healthy lifestyle with regular cardiovascular exercise to support respiratory health. "
                "Be mindful of environmental factors that could cause breathing issues, such as air quality and allergens."
            )

    elif feature == "noise_level":
        if feature_value > 4:
            recommendations.append(
                "You are frequently exposed to high levels of noise, which can cause stress, anxiety, and even hearing damage over time. "
                "Consider using earplugs or noise-cancelling headphones in noisy environments, and try to create a quieter space at home to relax. "
                "Exposure to constant noise can also affect sleep and concentration, so finding ways to reduce noise exposure is crucial for overall well-being."
            )
        elif 2 <= feature_value <= 4:
            recommendations.append(
                "You are sometimes exposed to moderate levels of noise. While this is not overly concerning, be mindful of prolonged exposure to noisy environments, "
                "especially if you experience difficulty concentrating or sleeping. Try to find quiet periods during the day to help your mind and body relax."
            )
        else:
            recommendations.append(
                "Your exposure to noise seems minimal and well-managed. Continue maintaining a quiet and peaceful environment for optimal focus and relaxation."
            )

    elif feature == "living_conditions":
        if feature_value < 2:
            recommendations.append(
                "Your living conditions appear to be stressful or inadequate. This could be affecting your overall well-being, including your mental and physical health. "
                "Consider evaluating whether you can make improvements to your environment, such as organizing your space, improving lighting, or creating a calming area for relaxation. "
                "If the issue is more severe (such as lack of privacy, unsafe conditions, or overcrowding), you may want to explore housing support services for assistance in finding better living arrangements."
            )
        elif 2 <= feature_value <= 3:
            recommendations.append(
                "Your living conditions seem moderate but may still present some stressors. Assess whether small improvements, like decluttering, adding personal touches, or enhancing your comfort, can help reduce stress. "
                "Making your living space a more comfortable and personal environment can improve your mental health and sense of security."
            )
        else:
            recommendations.append(
                "Your living conditions seem stable and supportive. Keep maintaining a healthy and peaceful environment by staying organized and ensuring your space remains conducive to relaxation and focus."
            )

    elif feature == "basic_needs":
        if feature_value < 2:
            recommendations.append(
                "It appears your basic needs, such as access to food, shelter, and safety, are not being adequately met. This can have a significant impact on your physical and mental well-being. "
                "Reach out to local support services, such as food banks, housing assistance, or counseling, to get help in addressing these critical needs. "
                "Prioritizing these areas is essential to building a stable foundation for overall health and stress management."
            )
        elif 2 <= feature_value <= 3:
            recommendations.append(
                "Your basic needs are mostly being met, but you may still experience occasional challenges. Ensure that you have consistent access to essentials like nutritious food, secure housing, and healthcare. "
                "Address any gaps that may be causing stress, such as irregular meal patterns or financial difficulties, by seeking support from community resources if necessary."
            )
        else:
            recommendations.append(
                "Your basic needs appear to be well taken care of, providing a stable foundation for your overall well-being. Keep focusing on maintaining a healthy, balanced lifestyle that ensures all your physical, emotional, and security needs are met."
            )


    elif feature == "academic_performance":
        if feature_value < 2:
            recommendations.append(
                "Your academic performance is significantly below average, which might indicate difficulties in understanding the material or managing your study time effectively. "
                "Consider seeking academic support, such as tutoring, joining study groups, or consulting with your teachers or professors for personalized advice. "
                "It may also help to review and adjust your study techniques, focusing on areas where you're struggling the most."
            )
        elif 2 <= feature_value <= 3:
            recommendations.append(
                "Your academic performance is slightly below average. You may benefit from reviewing your study strategies and identifying the subjects where you need more focus. "
                "Consider attending extra classes, using online resources, or collaborating with peers to strengthen your understanding of the material."
            )
        else:
            recommendations.append(
                "Your academic performance is on track, and you're doing well in your studies. Keep maintaining good study habits, and consider setting higher academic goals to challenge yourself. "
                "It's important to continue seeking feedback from your teachers and staying organized to ensure consistent performance."
            )

    elif feature == "study_load":
        if feature_value > 4:
            recommendations.append(
                "Your study load is exceptionally high, which could lead to stress and burnout if not managed properly. It's essential to prioritize your tasks, break down complex assignments, "
                "and create a structured study schedule. Make sure to allocate time for relaxation and self-care to maintain your mental and physical health. "
                "If you're feeling overwhelmed, consider speaking with a counselor or academic advisor to adjust your workload."
            )
        elif 3 <= feature_value <= 4:
            recommendations.append(
                "Your study load is somewhat high. While it's manageable, it's important to stay organized and ensure you're not taking on too much at once. "
                "Use tools like a planner or digital calendar to map out your study sessions, assignments, and deadlines. Don't hesitate to reach out for help if you're struggling to balance your workload."
            )
        else:
            recommendations.append(
                "Your study load appears to be balanced, and you're managing your time effectively. Keep practicing good time management habits and ensure that you continue to allocate sufficient time for rest and relaxation. "
                "Maintaining this balance will help you sustain your academic performance without feeling overwhelmed."
            )

    elif feature == "teacher_student_relationship":
        if feature_value < 2:
            recommendations.append(
                "Your relationship with your teachers may be strained or lacking. Open communication with your teachers can be crucial for academic success. "
                "Consider scheduling a meeting with them to discuss any concerns or difficulties you're experiencing in their classes. Building a rapport with your teachers can provide you with the support and guidance needed to improve academically."
            )
        elif 2 <= feature_value <= 3:
            recommendations.append(
                "Your relationship with your teachers is somewhat neutral, but there may be room for improvement. Consider being more proactive in class by asking questions and seeking feedback on your progress. "
                "A strong teacher-student relationship can enhance your learning experience, so it's worth making the effort to engage more with your instructors."
            )
        else:
            recommendations.append(
                "Your relationship with your teachers appears to be strong and supportive. Keep nurturing these connections by actively participating in class and seeking out additional learning opportunities. "
                "Teachers can be valuable mentors, and maintaining good relationships with them can benefit you academically and professionally."
            )

    elif feature == "future_career_concerns":
        if feature_value > 4:
            recommendations.append(
                "Your concerns about your future career are high, which may be causing significant stress or uncertainty. It might help to seek career counseling or mentorship to gain clarity on your career path. "
                "Consider breaking your long-term career goals into smaller, achievable steps and focusing on building relevant skills. Engaging in internships, networking, or seeking guidance from professionals in your field can also provide reassurance and direction."
            )
        elif 3 <= feature_value <= 4:
            recommendations.append(
                "You're experiencing moderate concerns about your future career, which is normal. To manage this, it could help to set clear, realistic career goals and develop a step-by-step plan to achieve them. "
                "Regularly revisiting your career goals and seeking feedback from mentors can reduce anxiety and provide a sense of direction as you move forward."
            )
        else:
            recommendations.append(
                "Your future career concerns seem manageable, and you're likely on a path toward achieving your professional goals. Keep setting realistic career goals and periodically re-evaluating them to ensure you're on track. "
                "Consider seeking opportunities for professional development, such as internships, certifications, or networking events, to stay competitive in your field."
            )

    elif feature == "social_support":
        if feature_value < 2:
            recommendations.append(
                "Your social support network is limited, which may be affecting your emotional well-being. It's important to build connections with others, whether through family, friends, or community groups. "
                "Consider joining clubs, volunteering, or participating in group activities to meet new people and strengthen your social ties. Having a reliable support system is crucial for managing stress and maintaining mental health."
            )
        elif 2 <= feature_value <= 3:
            recommendations.append(
                "Your social support is moderate, but there's potential to expand it. Strengthen your current relationships by spending more time with friends or family members, and don't hesitate to reach out to them when you need emotional support. "
                "Building a more robust social network can improve your emotional health and provide a sense of belonging."
            )
        else:
            recommendations.append(
                "You have strong social support, which is great for your mental and emotional well-being. Continue nurturing those relationships and offering support to others, as this will maintain the positive dynamics of your social network."
            )

    elif feature == "peer_pressure":
        if feature_value > 4:
            recommendations.append(
                "You may be experiencing significant peer pressure, which can be difficult to manage. It's important to prioritize your own values and make decisions based on what is best for you, "
                "rather than feeling compelled to follow the crowd. Consider setting clear boundaries with peers who may be encouraging behaviors that make you uncomfortable. "
                "Engaging in open conversations with trusted friends or mentors can help you resist negative influences and remain confident in your choices."
            )
        elif 2 <= feature_value <= 4:
            recommendations.append(
                "You may feel some peer pressure in certain situations, which is common. It's crucial to practice assertiveness and express your feelings or concerns when you're uncomfortable. "
                "You might benefit from rehearsing how to say 'no' in a respectful yet firm manner when you're faced with peer pressure. Surround yourself with supportive peers who respect your decisions."
            )
        else:
            recommendations.append(
                "Peer pressure doesn't seem to be a major issue for you. Continue practicing assertiveness and maintaining strong boundaries when it comes to peer influence. "
                "Having the confidence to stand by your own values is a great strength, and it can also help others see you as a positive role model."
            )

    elif feature == "extracurricular_activities":
        if feature_value < 2:
            recommendations.append(
                "Your involvement in extracurricular activities is quite limited, which may be impacting your ability to build connections outside of academics. Engaging in activities aligned with your personal interests, "
                "such as sports, arts, or community service, can help you develop new skills, relieve stress, and create a more balanced lifestyle. Additionally, participating in extracurricular activities can enhance your resume, "
                "boost your confidence, and give you opportunities to form friendships with like-minded individuals."
            )
        elif 2 <= feature_value <= 3:
            recommendations.append(
                "You have some involvement in extracurricular activities, but there may be room to explore more options that align with your interests. Expanding your participation in activities such as clubs, sports, or volunteering can provide "
                "a break from academic pressures, help you develop new skills, and offer valuable networking opportunities. Consider joining activities that you are passionate about or that could help you with career or personal development."
            )
        else:
            recommendations.append(
                "Your involvement in extracurricular activities is balanced and likely contributing positively to your overall well-being. Keep participating in activities that bring you joy and foster personal growth. "
                "Extracurriculars are also a great way to build leadership skills, relieve stress, and make meaningful connections with others. Maintaining this balance can help you manage academic pressures more effectively."
            )

    elif feature == "bullying":
        if feature_value > 1:
            recommendations.append(
                "You may be experiencing bullying, which can significantly affect your mental and emotional health. It's crucial to address this situation by seeking support from trusted individuals such as teachers, counselors, or family members. "
                "Consider talking to a school counselor or mental health professional who can provide strategies to cope with bullying and offer guidance on how to handle the situation. It's important to remember that you don't have to face this alone, "
                "and taking steps to protect yourself is a sign of strength."
            )
        elif feature_value == 1:
            recommendations.append(
                "You may have encountered some form of bullying or negative social interactions. It's important to remain aware of your feelings and ensure that you're not being treated unfairly or disrespectfully. "
                "If you feel uncomfortable or threatened by someone's behavior, consider discussing the situation with a counselor or trusted adult. Standing up for yourself and seeking support can help prevent further issues."
            )
        else:
            recommendations.append(
                "Bullying doesn't seem to be an issue for you at the moment, which is a positive sign. Continue maintaining healthy boundaries in your social interactions and fostering respectful relationships with others. "
                "It's important to be aware of the signs of bullying, both for yourself and others, so you can provide support if you or someone you know ever experiences it."
            )


    # Default recommendation if the feature is unrecognized or value is missing
    else:
        recommendations.append(
            f"There are some important factors related to {feature} that could benefit from further attention. "
            "For personalized advice, consider discussing this with a healthcare or mental health professional who can provide specific guidance."
        )


    return recommendations
