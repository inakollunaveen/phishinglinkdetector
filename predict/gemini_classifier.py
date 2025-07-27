# predict/gemini_classifier.py

import os
import google.generativeai as genai
from dotenv import load_dotenv

# ✅ Load .env for API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# ✅ Configure API
genai.configure(api_key=api_key)

# ✅ Use gemini-pro and v1 (correct model & method)
model = genai.GenerativeModel(model_name="models/gemini-pro")

def gemini_classifier(url: str) -> dict:
    try:
        # Minimal prompt
        prompt = (
            f"Classify this URL as 'phishing' or 'safe'. "
            f"Reply with only one word.\n\nURL: {url}"
        )

        # ✅ Call generate_content (supported for gemini-pro)
        response = model.generate_content(prompt)
        result = response.text.strip().lower()

        if "phishing" in result:
            label = "phishing"
            confidence = 0.9
        elif "safe" in result:
            label = "safe"
            confidence = 0.99
        else:
            label = "error"
            confidence = 0.0

        return {
            "label": label,
            "confidence": confidence,
            "raw_response": result,
            "error": None
        }

    except Exception as e:
        return {
            "label": "error",
            "confidence": 0.0,
            "raw_response": "",
            "error": str(e)
        }
