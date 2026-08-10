import os
import google.generativeai as genai
from app.config.settings import settings

api_key = settings.GEMINI_API_KEY.strip().strip("'\"")
genai.configure(api_key=api_key)

print(f"Testing API key: {api_key[:10]}...")
print("Available models for generateContent:")
try:
    for model in genai.list_models():
        if "generateContent" in model.supported_generation_methods:
            print(f" - {model.name}")
except Exception as e:
    print(f"Error fetching models: {e}")