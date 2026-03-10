from fastapi import FastAPI, UploadFile, File
import io
import pickle
import os
import pandas as pd
from pydantic import BaseModel

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from training.feature_engineer import add_engineered_features
from inference.explainer import ShapExplainer
from api.llm_explainer import generate_explanation

app = FastAPI(title="AI Security Intelligence API")

# =========================
# LOAD MODEL PIPELINE
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

print("Loading ids_pipeline.pkl...")
with open(os.path.join(MODEL_DIR, "ids_pipeline.pkl"), "rb") as f:
    pipeline = pickle.load(f)

with open(os.path.join(MODEL_DIR, "feature_columns.pkl"), "rb") as f:
    feature_columns = pickle.load(f)
    
transformed_feature_names = pipeline.named_steps["preprocessing"].get_feature_names_out()
explainer = ShapExplainer(pipeline.named_steps["classifier"], transformed_feature_names)


# =========================
# REQUEST SCHEMA
# =========================
class TrafficInput(BaseModel):
    dur: float
    proto: str
    service: str
    state: str
    spkts: float
    dpkts: float
    sbytes: float
    dbytes: float
    rate: float
    sload: float
    dload: float
    sloss: float
    dloss: float
    sinpkt: float
    dinpkt: float
    sjit: float
    djit: float
    swin: float
    stcpb: float
    dtcpb: float
    dwin: float
    tcprtt: float
    synack: float
    ackdat: float
    smean: float
    dmean: float
    trans_depth: float
    response_body_len: float
    ct_src_dport_ltm: float
    ct_dst_sport_ltm: float
    is_ftp_login: float
    ct_ftp_cmd: float
    ct_flw_http_mthd: float
    is_sm_ips_ports: float
    attack_cat: str


# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def home():
    return {"message": "AI Security Intelligence API Running 🚀"}


# =========================
# POST /predict
# =========================
@app.post("/predict")
def predict(data: TrafficInput):
    input_df = pd.DataFrame([data.dict()])
    
    # Feature engineering
    input_df = add_engineered_features(input_df)
    
    # Ensure correct feature order (drop unused)
    cols = [c for c in feature_columns if c in input_df.columns]
    input_df = input_df[cols]

    # Predict
    pred = int(pipeline.predict(input_df)[0])

    return {
        "prediction": pred,
        "label": "Attack" if pred == 1 else "Normal"
    }


# =========================
# FULL ANALYSIS
# =========================
@app.post("/analyze")
def analyze(data: TrafficInput):
    input_df = pd.DataFrame([data.dict()])

    # Feature engineering
    input_df = add_engineered_features(input_df)
    cols = [c for c in feature_columns if c in input_df.columns]
    input_df = input_df[cols]

    pred = int(pipeline.predict(input_df)[0])

    # Explain SHAP
    top_features = {}
    if pred == 1:
        # Transform data using preprocessor to get SHAP correctly
        X_scaled = pipeline.named_steps["preprocessing"].transform(input_df)
        shap_impacts = explainer.explain_instance(X_scaled, top_k=5)
        top_features = {x["feature"]: x["importance"] for x in shap_impacts}

    # LLM Explanation
    explanation = generate_explanation(pred, top_features)

    return {
        "prediction": pred,
        "label": "Attack" if pred == 1 else "Normal",
        "explanation": explanation,
        "top_features": top_features
    }

# =========================
# BATCH CSV ANALYSIS
# =========================
@app.post("/analyze-csv")
async def analyze_csv(file: UploadFile = File(...)):
    contents = await file.read()
    input_df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
    
    # Feature engineering
    engineered_df = add_engineered_features(input_df)
    cols = [c for c in feature_columns if c in engineered_df.columns]
    
    # Add missing cols with 0
    for c in cols:
        if c not in engineered_df.columns:
            engineered_df[c] = 0
            
    engineered_df = engineered_df[cols]
    
    preds = pipeline.predict(engineered_df)
    
    results = []
    for i, p in enumerate(preds):
        results.append({
            "row": i,
            "prediction": int(p),
            "label": "Attack" if int(p) == 1 else "Normal"
        })
        
    return {"results": results}