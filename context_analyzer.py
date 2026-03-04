import re

class ContextAnalyzer:
    def __init__(self):
        # Centralized Medical Database (Derived from your JSON)
        self.REF_DB = {
            # KIDNEY
            "LAB_CREATININE": {"label": "Creatinine", "min": 0.7, "max": 1.3, "unit": "mg/dL"},
            "LAB_BUN": {"label": "Blood Urea Nitrogen", "min": 15, "max": 45, "unit": "mg/dL"},
            "LAB_URINE_ALBUMIN": {"label": "Urine Albumin", "min": 0, "max": 0, "unit": "mg/dL"}, # Should be absent
            "LAB_URINE_SG": {"label": "Urine Specific Gravity", "min": 1.005, "max": 1.030, "unit": ""},
            "LAB_SODIUM": {"label": "Sodium", "min": 135, "max": 145, "unit": "mEq/L"},
            "LAB_POTASSIUM": {"label": "Potassium", "min": 3.5, "max": 5.0, "unit": "mEq/L"},

            # LIVER
            "LAB_BILIRUBIN_TOTAL": {"label": "Total Bilirubin", "min": 0.3, "max": 1.2, "unit": "mg/dL"},
            "LAB_BILIRUBIN_DIRECT": {"label": "Direct Bilirubin", "min": 0.0, "max": 0.3, "unit": "mg/dL"},
            "LAB_SGPT": {"label": "ALT (SGPT)", "min": 10, "max": 40, "unit": "U/L"},
            "LAB_SGOT": {"label": "AST (SGOT)", "min": 10, "max": 40, "unit": "U/L"},
            "LAB_ALP": {"label": "Alkaline Phosphatase", "min": 44, "max": 147, "unit": "U/L"},
            "LAB_PROTEIN": {"label": "Total Protein", "min": 6.0, "max": 8.3, "unit": "g/dL"},
            "LAB_ALBUMIN": {"label": "Albumin", "min": 3.5, "max": 5.5, "unit": "g/dL"},
            "LAB_AG_RATIO": {"label": "A/G Ratio", "min": 1.0, "max": 2.0, "unit": ""},

            # BLOOD / COUNTS
            "LAB_WBC": {"label": "WBC Count", "min": 4, "max": 11, "unit": "* 10^3 /cumm"},
            "LAB_PLATELETS": {"label": "Platelet Count", "min": 150000, "max": 450000, "unit": "/cumm"},
            # Gender Specific Logic handled in method below
            "LAB_HEMOGLOBIN": {"label": "Hemoglobin", "unit": "g/dL", "gender_split": {"m": (13.0, 17.0), "f": (12.0, 15.0)}},
            "LAB_HCT": {"label": "Hematocrit", "unit": "%", "gender_split": {"m": (40.0, 50.0), "f": (36.0, 46.0)}},
            "LAB_RBC": {"label": "RBC Count", "unit": "mill/cumm", "gender_split": {"m": (4.5, 5.5), "f": (3.8, 4.8)}},
        }

    def analyze(self, features, age=30, gender=1, is_pregnant=False):
        """
        Returns:
        1. insights: List of text strings for LLM Context.
        2. evaluated: Dict of {key: {'status': 'HIGH', 'ref': '10-40', 'label': 'ALT'}}
        """
        insights = []
        evaluated = {}
        
        # Gender string map for internal logic
        sex = "m" if gender == 1 else "f"

        for key, val in features.items():
            if key not in self.REF_DB:
                continue
                
            try:
                val = float(val)
            except: continue

            ref = self.REF_DB[key]
            
            # Determine Ranges
            if "gender_split" in ref:
                low, high = ref["gender_split"][sex]
            else:
                low, high = ref["min"], ref["max"]

            # Determine Status
            status = "NORMAL"
            if val < low: status = "LOW"
            elif val > high: status = "HIGH"

            # Store Evaluated Data (For Main.py to use in Prompt)
            evaluated[key] = {
                "label": ref["label"],
                "value": val,
                "status": status,
                "ref_range": f"{low}-{high} {ref['unit']}"
            }

            # Generate Insights (Only for abnormal values)
            if status != "NORMAL":
                insights.append(f"{ref['label']}: {val} is {status} (Normal: {low}-{high}).")

        # --- SPECIAL CONTEXT RULES (Pediatric/Geriatric/Pregnancy) ---
        # These override or add nuance to the basic High/Low checks
        
        # Pregnancy Rules
        if is_pregnant and sex == "f":
            if "LAB_HEMOGLOBIN" in evaluated and evaluated["LAB_HEMOGLOBIN"]["status"] == "LOW":
                insights.append("Note: Mild anemia is often physiological during pregnancy (dilutional).")
            if "LAB_ALP" in evaluated and evaluated["LAB_ALP"]["status"] == "HIGH":
                insights.append("Note: Elevated ALP is expected in pregnancy (placental origin).")

        # Pediatric Rules (<18)
        if age < 18:
            if "LAB_ALP" in evaluated and evaluated["LAB_ALP"]["status"] == "HIGH":
                insights.append("Note: High ALP is normal in children due to bone growth.")
            if "LAB_CREATININE" in evaluated and evaluated["LAB_CREATININE"]["status"] == "LOW":
                insights.append("Note: Lower creatinine is normal in children (less muscle mass).")

        return insights, evaluated
