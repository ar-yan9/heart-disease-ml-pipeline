from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predicts heart disease using ML trained on UCI dataset",
    version="1.0.0"
)

# Fix CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
preprocessor = joblib.load(BASE_DIR / "models" / "preprocessor.pkl")
model        = joblib.load(BASE_DIR / "models" / "best_model.pkl")

class PatientData(BaseModel):
    age:      float
    sex:      float
    cp:       float
    trestbps: float
    chol:     float
    fbs:      float
    restecg:  float
    thalach:  float
    exang:    float
    oldpeak:  float
    slope:    float
    ca:       float
    thal:     float

    class Config:
        json_schema_extra = {
            "example": {
                "age": 52, "sex": 1, "cp": 0, "trestbps": 125,
                "chol": 212, "fbs": 0, "restecg": 1, "thalach": 168,
                "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3
            }
        }

@app.get("/")
def home():
    return {
        "message": "Heart Disease Prediction API is running!",
        "docs":    "Visit /docs to test the API"
    }

@app.post("/predict")
def predict(data: PatientData):
    try:
        features = np.array([[
            data.age, data.sex, data.cp, data.trestbps,
            data.chol, data.fbs, data.restecg, data.thalach,
            data.exang, data.oldpeak, data.slope, data.ca, data.thal
        ]])
        processed   = preprocessor.transform(features)
        prediction  = model.predict(processed)[0]
        probability = model.predict_proba(processed)[0][1]

        return {
            "prediction":  int(prediction),
            "label":       "Heart Disease Detected" if prediction == 1 else "No Heart Disease",
            "confidence":  f"{round(float(probability) * 100, 2)}%",
            "risk_level":  "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low"
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health():
    return {"status": "API is healthy"}