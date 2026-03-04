import pandas as pd
import xgboost as xgb
import ollama
import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, Union
from context_analyzer import ContextAnalyzer

app = FastAPI()
MODELS = {}
analyzer = ContextAnalyzer()

# Human-Readable Names
FEATURE_LEGEND = {
    "LAB_CREATININE": "Serum Creatinine",
    "LAB_BUN": "Blood Urea Nitrogen",
    "LAB_SODIUM": "Sodium",
    "LAB_POTASSIUM": "Potassium",
    "LAB_URINE_ALBUMIN": "Urine Albumin",
    "LAB_URINE_SG": "Urine Specific Gravity",
    "LAB_BILIRUBIN_TOTAL": "Total Bilirubin",
    "LAB_SGPT": "ALT (SGPT)",
    "LAB_SGOT": "AST (SGOT)",
    "LAB_HEMOGLOBIN": "Hemoglobin",
    "LAB_HCT": "Hematocrit",
    "LAB_RBC": "RBC Count",
    "LAB_WBC": "WBC Count",
    "LAB_PLATELETS": "Platelet Count",
    "LAB_AG_RATIO": "Albumin/Globulin Ratio",
    "LAB_SGOT_SGPT_RATIO": "AST/ALT Ratio"
}

# MARKERS THAT MUST BE SHOWN FOR EACH ORGAN
ORGAN_CORE_MARKERS = {
    "kidney": ["LAB_CREATININE", "LAB_BUN", "LAB_URINE_ALBUMIN"],
    "liver": ["LAB_SGPT", "LAB_SGOT", "LAB_BILIRUBIN_TOTAL"]
}

@app.on_event("startup")
def load_models():
    for name in ["kidney", "liver"]:
        path = f"models/{name}_optimized.json"
        if os.path.exists(path):
            bst = xgb.Booster()
            bst.load_model(path)
            MODELS[name] = bst
            print(f"✅ Loaded {name.upper()}")

class Request(BaseModel):
    task: str
    features: Dict[str, Union[float, str]] 
    notes: Optional[str] = ""
    is_pregnant: Optional[bool] = False

class IntentRequest(BaseModel):
    query: str

@app.post("/agent/infer_intent")
async def infer(payload: IntentRequest):
    try:
        prompt = f"Classify: '{payload.query}' as UPLOAD_REPORT, HEALTH_QUERY, or CLARIFICATION. Reply ONLY category."
        res = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
        return {"intent": res['message']['content'].strip()}
    except: return {"intent": "UNKNOWN"}

def engineer_features(feats):
    def get_f(k): return float(feats.get(k, 0.0))
    if "LAB_SGOT" in feats and "LAB_SGPT" in feats:
        sgpt = get_f("LAB_SGPT")
        if sgpt > 0:
            feats["LAB_SGOT_SGPT_RATIO"] = get_f("LAB_SGOT") / sgpt
    return feats

@app.post("/predict")
async def predict(payload: Request):
    if payload.task not in MODELS: raise HTTPException(400, "Invalid Task")
    
    task_key = payload.task.lower()
    model = MODELS[task_key]
    feats = payload.features.copy()
    feats = engineer_features(feats)
    
    age = float(feats.get("AGE", 45))
    gender = 0 if "f" in str(feats.get("GENDER", "")).lower() else 1
    feats["AGE"], feats["GENDER"] = age, gender
    
    # 1. Prediction
    df = pd.DataFrame([feats])
    if hasattr(model, "feature_names"):
        for f in model.feature_names:
            if f not in df.columns: df[f] = float('nan')
        df = df[model.feature_names]
    
    df = df.apply(pd.to_numeric, errors='coerce')
    score = float(model.predict(xgb.DMatrix(df))[0])
    risk = "HIGH" if score > 0.5 else "LOW"
    
    # 2. ANALYSIS
    text_insights, evaluated_data = analyzer.analyze(
        feats, age=age, gender=gender, is_pregnant=payload.is_pregnant
    )
    
    # 3. Explainability (With CORE MARKER INJECTION)
    factors_json = {}
    
    try:
        # A. Start with Core Organ Markers (Creatinine, etc.)
        priority_drivers = []
        seen_keys = set()
        
        # Always show core markers first
        if task_key in ORGAN_CORE_MARKERS:
            for k in ORGAN_CORE_MARKERS[task_key]:
                if k in feats:
                    priority_drivers.append(k)
                    seen_keys.add(k)

        # B. Add Abnormal Drivers (High/Low)
        imp = model.get_score(importance_type='gain')
        sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)
        
        for k, v in sorted_imp:
            if k not in seen_keys and k in evaluated_data and evaluated_data[k]["status"] != "NORMAL":
                priority_drivers.append(k)
                seen_keys.add(k)
                # Keep score for chart
                factors_json[evaluated_data[k]["label"]] = float(v)

        # C. Fill remaining slots with Top Mathematical Drivers
        for k, v in sorted_imp:
            if len(priority_drivers) >= 5: break # Limit to 5 total lines
            if k not in seen_keys and k in feats:
                priority_drivers.append(k)
                seen_keys.add(k)
                factors_json[FEATURE_LEGEND.get(k, k)] = float(v)

        # Generate Text Block
        driver_lines = []
        for k in priority_drivers:
            label = FEATURE_LEGEND.get(k, k)
            val = feats.get(k)
            
            # Use Analyzer data if available
            if k in evaluated_data:
                d = evaluated_data[k]
                line = f"- {d['label']}: {d['value']} (Ref: {d['ref_range']}) -> {d['status']}"
            else:
                line = f"- {label}: {val}"
            
            driver_lines.append(line)

        driver_txt = "\n".join(driver_lines)
        
    except Exception as e:
        driver_txt = f"Data Error: {e}"

    # 4. Calibrated Prompt
    try:
        prompt = f"""
        Act as a Consultant Pathologist.
        
        PATIENT: {age} yrs, {'Male' if gender==1 else 'Female'}
        ORGAN ANALYSIS: {task_key.upper()}
        RISK ASSESSMENT: {risk} ({score:.1%})
        
        KEY LAB INDICATORS:
        {driver_txt}
        
        CLINICAL NOTES (May include incidental findings):
        {text_insights}
        
        TASK:
        Write a single, authoritative paragraph interpreting these results.
        
        GUIDELINES:
        1. Start by confirming the status of the TARGET ORGAN ({task_key.upper()}).
           - If Risk is LOW, cite the Normal markers (e.g. "Kidney function is preserved as Creatinine is normal").
        2. Address Abnormalities:
           - If Risk is LOW but markers are HIGH (e.g. RBC), describe them as "incidental findings" or "requiring monitoring" but NOT as the primary organ failure.
           - Reconcile the Low Score with the High Markers (e.g. "Despite elevated RBC, renal filtration is intact").
        3. Maintain a professional, reassuring tone if risk is Low.
        """
        res = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt}])
        reasoning = res['message']['content']
    except: reasoning = "Clinical assessment unavailable."

    disclaimer = "\n\n---\n**Disclaimer:** *AI-generated report. Consult a doctor for medical advice.*"
    
    return {
        "risk_score": score,
        "risk_level": risk,
        "explainability_drivers": factors_json,
        "context_insights": text_insights,
        "llm_explanation": reasoning + disclaimer
    }
