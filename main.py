import pandas as pd
import numpy as np
import random
import re
import joblib
import time
import lightgbm as lgb
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

# --- 1. CONFIGURATION AND MODEL LOADING ---

MODEL_PATH = 'lgbm_phishing_model.joblib'
lgbm_clf = None
print(f"Loading model from {MODEL_PATH}...")
try:
    lgbm_clf = joblib.load(MODEL_PATH)
    print("LightGBM model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Check your file structure.")
    # Exit if the model is missing, as the API cannot function
    sys.exit(1)

# --- 2. INPUT DATA MODEL (Pydantic for FastAPI) ---

class Message(BaseModel):
    """Defines the expected input structure for the API."""
    text: str
    sender: str
    channel: str # e.g., 'Email', 'SMS', 'WhatsApp', 'Social_Media'

# --- 3. FEATURE ENGINEERING LOGIC (The Core Intelligence) ---

# Helper function to count URLs (including obfuscated 'hxxp')
def extract_url_count(text):
    # Matches http:// or https:// or hxxp://
    url_pattern = r'https?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    obfuscated_pattern = r'hxxps?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return len(re.findall(url_pattern, text, re.IGNORECASE)) + len(re.findall(obfuscated_pattern, text, re.IGNORECASE))

# Helper function to check for suspicious sender keywords
def extract_suspicious_pattern(sender):
    suspicious_keywords = r'security|alert|verify|update|support|invoice|payment|\d{4,}'
    return 1 if re.search(suspicious_keywords, sender, re.IGNORECASE) else 0

# Main function to create the feature vector for the model
def create_feature_vector(text, sender, channel):
    # Initialize the feature vector (must match the training features)
    features = {
        'message_length': len(text),
        'caps_ratio': sum(1 for c in text if c.isupper()) / (len(text) or 1),
        'suspicious_sender': extract_suspicious_pattern(sender),
        'url_count': extract_url_count(text),
        'digit_count': sum(1 for c in text if c.isdigit()),
        'channel_Email': 0,
        'channel_SMS': 0,
        'channel_WhatsApp': 0,
        'channel_Social_Media': 0,
    }
    
    # One-Hot Encoding for Channel
    channel_key = f'channel_{channel.replace(" ", "_")}'
    if channel_key in features:
        features[channel_key] = 1
    
    # Convert to DataFrame row for LGBM prediction
    return pd.DataFrame([features])

# --- 4. FASTAPI APPLICATION SETUP ---

app = FastAPI(title="NanoDetect Phishing API", version="1.0")

@app.get("/")
def read_root():
    """Root endpoint for status check."""
    return {"status": "Service Running", "model_loaded": lgbm_clf is not None}

@app.post("/score")
async def score_message(message: Message) -> Dict[str, Any]:
    """Endpoint to score a message and return the phishing risk."""
    
    start_time = time.time()
    
    if lgbm_clf is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Service unavailable.")
        
    try:
        # 1. Create Feature Vector
        feature_df = create_feature_vector(message.text, message.sender, message.channel)
        
        # 2. Predict Probability
        # Get the probability of class 1 (Phishing)
        phishing_score = lgbm_clf.predict_proba(feature_df)[:, 1][0]
        
        # 3. Apply Decision Logic
        action = "Allow"
        if phishing_score >= 0.85:
            action = "Quarantine"
        elif phishing_score >= 0.55:
            action = "Alert User"
            
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        return {
            "phishing_score": float(phishing_score),
            "action": action,
            "latency_ms": float(latency_ms),
            "model": "LightGBM",
            "features_used": feature_df.iloc[0].to_dict()
        }

    except Exception as e:
        # Log the error for debugging on the server side
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail="Internal processing error during prediction.")
