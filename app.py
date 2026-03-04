import streamlit as st
import requests
import pandas as pd
import os
from pdf_parser import parse_lab_report

st.set_page_config(page_title="MedLLM Pro", layout="wide")
API_URL = "http://127.0.0.1:8000"

# --- STATE ---
if "data" not in st.session_state: st.session_state.data = {}
if "verified" not in st.session_state: st.session_state.verified = False
if "final_features" not in st.session_state: st.session_state.final_features = {}

# --- SIDEBAR ---
with st.sidebar:
    st.title("MedLLM: Organ Risk")
    uploaded = st.file_uploader("Upload Lab Report", type=['pdf'])
    
    if st.button("Reset System"):
        st.session_state.clear()
        st.rerun()

# --- HELPER: SMART ROUTER (Fixed) ---
def intelligent_router(feats):
    """
    Decides the model based on which organ has ABNORMAL values.
    Includes safe float conversion to prevent TypeErrors.
    """
    scores = {"Kidney": 0, "Liver": 0}
    
    # Safe helper to handle strings like "1.5" or "Absent"
    def get_val(key):
        try:
            return float(feats.get(key, 0))
        except (ValueError, TypeError):
            return 0.0
    
    # 1. Kidney Triggers
    if get_val("LAB_CREATININE") > 1.4: scores["Kidney"] += 5
    if get_val("LAB_URINE_ALBUMIN") > 0: scores["Kidney"] += 3
    if get_val("LAB_BUN") > 20: scores["Kidney"] += 2
    if "LAB_URINE_SG" in feats: scores["Kidney"] += 1 
    
    # 2. Liver Triggers
    if get_val("LAB_SGPT") > 50: scores["Liver"] += 5
    if get_val("LAB_SGOT") > 50: scores["Liver"] += 5
    if get_val("LAB_BILIRUBIN_TOTAL") > 1.2: scores["Liver"] += 5
    if "LAB_ALP" in feats: scores["Liver"] += 1 

    # Default to Liver if tie or specific extraction
    if scores["Liver"] >= scores["Kidney"]: return "Liver"
    return "Kidney"

# --- MAIN FLOW ---
if uploaded:
    # 1. PARSE
    if not st.session_state.data:
        with open("temp.pdf", "wb") as f: f.write(uploaded.getbuffer())
        with st.spinner("Extracting..."):
            st.session_state.data = parse_lab_report("temp.pdf")
        os.remove("temp.pdf")

    # 2. VERIFY
    st.subheader("1. Extracted Data Verification")
    
    items = list(st.session_state.data.items())
    items.sort(key=lambda x: 0 if x[0] in ["AGE", "GENDER"] else 1)
    
    df = pd.DataFrame(items, columns=["Parameter", "Value"])
    
    # Explicitly configure the Value column as Number to encourage float storage
    edited = st.data_editor(
        df,
        column_config={
            "Parameter": st.column_config.TextColumn("Test Name", width="medium", disabled=True),
            "Value": st.column_config.Column("Result", width="small", required=True)
        },
        use_container_width=True,
        num_rows="dynamic"
    )
    
    if st.button("Confirm Data", type="primary"):
        st.session_state.verified = True
        st.session_state.final_features = dict(zip(edited["Parameter"], edited["Value"]))
        st.rerun()

# 3. ANALYZE
if st.session_state.verified:
    st.divider()
    feats = st.session_state.final_features
    
    # SMART AUTO-ROUTE
    auto_mode = intelligent_router(feats)
    
    st.subheader("2. Analysis Module")
    
    mode = st.radio("Target Organ", ["Liver", "Kidney"], index=["Liver", "Kidney"].index(auto_mode))
    
    if mode == auto_mode:
        st.caption(f"Auto-detected **{mode}** based on abnormal values.")

    # Safe Gender handling
    gender_raw = feats.get("GENDER", "Male")
    is_female = "f" in str(gender_raw).lower()
    
    is_pregnant = False
    if is_female:
        is_pregnant = st.checkbox("Patient is Pregnant?")

    if st.button(f"Analyze {mode} Risk"):
        with st.spinner(f"Running {mode} Assessment..."):
            try:
                res = requests.post(f"{API_URL}/predict", json={
                    "task": mode.lower(), "features": feats, "is_pregnant": is_pregnant
                }).json()
                
                c1, c2 = st.columns([1, 1.5])
                with c1:
                    st.metric(f"{mode} Risk Score", f"{res['risk_score']:.1%}", res['risk_level'])
                    
                    if res.get('context_insights'):
                        st.info("Clinical Context:")
                        for i in res['context_insights']:
                            st.write(f"- {i}")
                
                with c2:
                    st.success("AI Assessment")
                    st.write(res['llm_explanation'])
                    st.divider()
                    st.write("**Primary Risk Drivers:**")
                    st.json(res['explainability_drivers'])
                    
            except Exception as e:
                st.error(f"Analysis Error: {e}")
