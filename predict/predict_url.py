import os
import joblib
import numpy as np
from dotenv import load_dotenv
from sklearn.ensemble import IsolationForest
import google.generativeai as genai
from feature_extraction.extract_features import transform_url

# Load .env and Gemini API key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Load ML model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "phishing_model.pkl")
MODEL_PATH = os.path.abspath(MODEL_PATH)
ml_model = joblib.load(MODEL_PATH)

# Use the correct Gemini model ID (check Google AI Studio if unsure)
GEMINI_MODEL_ID = "models/gemini-1.5-pro-latest"  # Change if needed
gemini_model = genai.GenerativeModel(GEMINI_MODEL_ID)


def gemini_classifier(url: str) -> dict:
    try:
        prompt = (
            f"You are a cybersecurity analyst. Classify the following URL as either 'phishing' or 'safe'. "
            f"Give only one word as output: phishing or safe. URL: {url}"
        )
        response = gemini_model.generate_content(prompt)
        content = response.text.strip().lower()

        if "phishing" in content:
            label = "phishing"
        elif "safe" in content:
            label = "safe"
        else:
            label = "unknown"

        return {
            "label": label,
            "confidence": 0.9 if label in ["phishing", "safe"] else 0.5,
            "raw_response": content
        }

    except Exception as e:
        return {
            "label": "error",
            "confidence": 0.0,
            "error": str(e),
            "raw_response": ""
        }


def predict_url(url: str) -> dict:
    try:
        # Extract features
        features = transform_url(url)
        features_array = np.array(features).reshape(1, -1)

        # ML model prediction
        ml_pred = ml_model.predict(features_array)[0]
        ml_label = "phishing" if ml_pred == 1 else "safe"

        # Gemini model prediction
        gemini_result = gemini_classifier(url)

        return {
            "url": url,
            "ml_model_prediction": ml_label,
            "gemini_prediction": gemini_result.get("label", "error"),
            "gemini_confidence": gemini_result.get("confidence", 0.0),
            "gemini_error": gemini_result.get("error", None),
            "gemini_raw_response": gemini_result.get("raw_response", "")
        }

    except Exception as e:
        return {"error": f"❌ ❌ Internal prediction error: {str(e)}"}
