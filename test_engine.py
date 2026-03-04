import requests
import json

# The Server URL
url = "http://127.0.0.1:8000/predict"

# 🚑 TEST CASE: High Risk ICU Patient
mimic_payload = {
    "task": "mimic",
    "features": {
        "AGE": 85.0,
        "GENDER": 1.0,           
        "LAB_HCT": 30.2,         
        "LAB_GLUCOSE": 140.0,    
        "LAB_BUN": 45.0,         
        "LAB_WBC": 18.5,         
        "LAB_CREATININE": 2.8,   # High Kidney Stress
        "LAB_PLATELETS": 100.0,  
        "LAB_SODIUM": 130.0      
    },
    "notes": "Patient presented with acute kidney injury and septic shock. Unresponsive to initial fluids."
}

print("sending request...")
response = requests.post(url, json=mimic_payload)

if response.status_code == 200:
    res = response.json()
    print("\n✅ SUCCESS! ENGINE OUTPUT:")
    print("="*40)
    print(f"📊 Risk Score: {res['risk_score']:.2%}")
    print(f"⚠️ Risk Level: {res['risk_level']}")
    
    # NEW KEYS HERE vvv
    print(f"🔍 Top Factors: {res['explainability_drivers']}") 
    print(f"🛡️ Guardrail Triggered: {res['guardrail_triggered']}")
    print(f"📝 Guardrail Log: {res['guardrail_log']}")
    
    print("-"*40)
    print(f"🤖 LLM Explanation:\n{res['llm_explanation']}")
    print("="*40)
else:
    print("❌ Error:", response.text)
