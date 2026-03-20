from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import os

# Load model files
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

model = pickle.load(open(f"{BASE_DIR}/models/crop_recommendation_model.pkl","rb"))
feature_encoders = pickle.load(open(f"{BASE_DIR}/models/feature_encoders.pkl","rb"))
target_encoder = pickle.load(open(f"{BASE_DIR}/models/target_encoder.pkl","rb"))
feature_order = pickle.load(open(f"{BASE_DIR}/models/feature_order.pkl", "rb"))

app = FastAPI(title="Kisan Sathi Crop Recommendation API")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


@app.post("/predict")
def predict_crop(data: CropInput):

    state = data.state.strip().lower()
    season = data.season.strip().capitalize()

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

    encoded_sample = []

    for col in feature_order:
        if col in feature_encoders:
            encoded_sample.append(
                feature_encoders[col].transform([input_dict[col]])[0]
            )
        else:
            encoded_sample.append(input_dict[col])

    features = np.array([encoded_sample])

    prediction = model.predict(features)

    crop = target_encoder.inverse_transform(prediction)

    return {"recommended_crop": crop[0]}


@app.get("/")
def home():
    return {"message": "Kisan Sathi Crop Recommendation API is running"}