# main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

# Define the path to your joblib model file
MODEL_PATH = os.path.join("model", "lightgbm_model.joblib")

# 1. Load Model
try:
    # This will crash the app if the model file is not in 'model/lightgbm_model.joblib'
    MODEL = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except FileNotFoundError:
    raise RuntimeError(f"Deployment Error: Model file not found at {MODEL_PATH}. Did you commit it?")
except Exception as e:
    raise RuntimeError(f"Deployment Error: Failed to load model: {e}")

# 2. Initialize FastAPI App
app = FastAPI(title="Phishing Detection API")

# 3. CORS Configuration (Fixes the frontend connection error)
# Use your GitHub Pages URL here, e.g., "https://nehanronakgalla.github.io/Phishing-Detection..."
# Using "*" allows all origins for testing.
origins = ["*"] 

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)

# 4. Input Schema (Must match the data your frontend sends)
class PhishingInput(BaseModel):
    text_content: str
    url: str
    metadata_age_days: int # Example metadata feature

# 5. Endpoints
@app.get("/")
def root():
    return {"status": "NanoDetect API is LIVE", "model": "LightGBM"}

@app.post("/predict")
def predict_phishing(input_data: PhishingInput):
    # This must replicate your exact feature extraction from training
    
    feature_vector = [
        len(input_data.url), 
        input_data.url.count('@'), 
        input_data.metadata_age_days
        # Add your other features here...
    ]
    
    # Create DataFrame with correct column names matching training data
    features_df = pd.DataFrame([feature_vector], 
                               columns=['url_length', 'at_count', 'age_days']) 

    prediction = MODEL.predict(features_df)[0]
    
    result = "PHISHING" if prediction > 0.5 else "LEGITIMATE"
        
    return {
        "prediction": result,
        "probability": float(prediction)
    }
