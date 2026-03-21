from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pickle
import os

# =============================
# CREATE FASTAPI APP FIRST
# =============================
app = FastAPI(title="Kisan Sathi Crop Recommendation API")

# =============================
# ENABLE CORS
# =============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# LOAD MODEL FILES
# =============================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

model = pickle.load(open(os.path.join(BASE_DIR, "models", "crop_recommendation_model.pkl"), "rb"))
feature_encoders = pickle.load(open(os.path.join(BASE_DIR, "models", "feature_encoders.pkl"), "rb"))
target_encoder = pickle.load(open(os.path.join(BASE_DIR, "models", "target_encoder.pkl"), "rb"))
feature_order = pickle.load(open(os.path.join(BASE_DIR, "models", "feature_order.pkl"), "rb"))

# =============================
# INPUT SCHEMA
# =============================
class CropInput(BaseModel):
    state: str
    season: str
    N: float
    P: float
    K: float
    pH: float
    rainfall: float
    soil_moisture: float
    min_temp: float
    max_temp: float

# =============================
# PREDICTION API
# =============================
@app.post("/predict")
def predict_crop(data: CropInput):

    # Clean inputs
    state = data.state.strip().lower()
    season = data.season.strip().capitalize()

    # Prepare input dictionary
    input_dict = {
        "N": data.N,
        "P": data.P,
        "K": data.K,
        "pH": data.pH,
        "Season": season,
        "Avg_Seasonal_Rainfall_mm": data.rainfall,
        "Avg_Soil_Moisture": data.soil_moisture,
        "Avg_Min_Temp (?C)": data.min_temp,
        "Avg_Max_Temp (?C)": data.max_temp,
        "State": state
    }

    # Encode + arrange features
    encoded_sample = []

    for col in feature_order:
        if col in feature_encoders:
            encoded_sample.append(
                feature_encoders[col].transform([input_dict[col]])[0]
            )
        else:
            encoded_sample.append(input_dict[col])

    features = np.array([encoded_sample])

    # Prediction
    prediction = model.predict(features)
    crop = target_encoder.inverse_transform(prediction)

    return {"recommended_crop": crop[0]}

# =============================
# ROOT API
# =============================
@app.get("/")
def home():
    return {"message": "Kisan Sathi Crop Recommendation API is running"}
