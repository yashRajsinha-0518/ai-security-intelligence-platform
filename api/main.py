from fastapi import FastAPI
import pickle
import os
import numpy as np
from pydantic import BaseModel

from api.llm_explainer import generate_explanation

app = FastAPI(title="AI Security Intelligence API")

# =========================
# LOAD MODEL ARTIFACTS
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

with open(os.path.join(MODEL_DIR, "ids_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(MODEL_DIR, "feature_columns.pkl"), "rb") as f:
    feature_columns = pickle.load(f)


# =========================
# REQUEST SCHEMA
# =========================
class TrafficInput(BaseModel):
    features: list[float]


# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def home():
    return {"message": "AI Security Intelligence API Running 🚀"}


# =========================
# PREDICT ONLY
# =========================
@app.post("/predict")
def predict(data: TrafficInput):
    arr = np.array(data.features).reshape(1, -1)
    scaled = scaler.transform(arr)
    pred = int(model.predict(scaled)[0])

    return {
        "prediction": pred,
        "label": "Attack" if pred == 1 else "Normal"
    }


# =========================
# EXPLAIN ONLY
# =========================
@app.get("/explain/{prediction}")
def explain(prediction: int):
    explanation = generate_explanation(prediction)
    return {"explanation": explanation}


# =========================
# FULL ANALYSIS (MAIN DEMO)
# =========================
@app.post("/analyze")
def analyze(data: TrafficInput):
    arr = np.array(data.features).reshape(1, -1)
    scaled = scaler.transform(arr)
    pred = int(model.predict(scaled)[0])

    explanation = generate_explanation(pred)

    return {
        "prediction": pred,
        "label": "Attack" if pred == 1 else "Normal",
        "explanation": explanation
    }