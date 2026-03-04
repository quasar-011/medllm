import pdfplumber
import re

# --- STRICT MAPPING ---
# Keys must match the model features in train_models.py
LAB_MAPPING = {
    # === KIDNEY MODEL INPUTS ===
    "LAB_CREATININE": ["creatinine - serum", "serum creatinine", "creatinine"],
    "LAB_BUN": ["blood urea nitrogen", "bun"],
    "LAB_SODIUM": ["sodium"],
    "LAB_POTASSIUM": ["potassium"],
    "LAB_HEMOGLOBIN": ["hemoglobin"], 
    "LAB_HCT": ["hematocrit", "pcv"],
    "LAB_WBC": ["total leucocyte count", "total leukocyte count", "wbc"],
    # Note: "Total RBC" is specific to blood. "Red Blood Cells" often appears in Urine.
    "LAB_RBC": ["total rbc"], 
    "LAB_GLUCOSE_RANDOM": ["random blood sugar", "rbs"],
    "LAB_BP": ["blood pressure"],
    
    # URINE (Critical for CKD)
    "LAB_URINE_SG": ["specific gravity"],
    "LAB_URINE_ALBUMIN": ["urinary protein", "urine albumin", "urine protein"],
    "LAB_URINE_SUGAR": ["urinary glucose", "urine sugar"],
    # "Red Blood Cells" is strictly for Urine in many reports (Blood is "Total RBC")
    "LAB_URINE_RBC": ["red blood cells"], 
    "LAB_URINE_PUS": ["urinary leucocytes", "pus cells", "urine wbc"],
    "LAB_URINE_CLUMPS": ["pus cell clumps"],
    "LAB_URINE_BACTERIA": ["bacteria"],

    # === LIVER MODEL INPUTS ===
    "LAB_BILIRUBIN_TOTAL": ["bilirubin - total", "total bilirubin"],
    "LAB_BILIRUBIN_DIRECT": ["bilirubin -direct", "direct bilirubin"],
    "LAB_ALP": ["alkaline phosphatase"],
    # Specific phrasing to avoid mixing with "SGOT/SGPT Ratio"
    "LAB_SGPT": ["alanine transaminase", "sgpt"], 
    "LAB_SGOT": ["aspartate aminotransferase", "sgot"],
    "LAB_PROTEIN": ["protein - total", "total protein", "protein total"],
    "LAB_ALBUMIN": ["albumin - serum", "serum albumin"],
    "LAB_AG_RATIO": ["serum alb/globulin ratio", "a/g ratio"],
    
    # === DIABETES / EXTRA ===
    "LAB_HBA1C": ["hba1c"],
    "LAB_GLUCOSE_FASTING": ["fasting blood sugar", "fasting glucose"]
}

def clean_text(text):
    if not text: return ""
    # Normalize: Lowercase, remove multi-spaces
    return re.sub(r'\s+', ' ', str(text).lower().replace('\n', ' ')).strip()

def extract_value_from_row(row_text, keyword):
    """
    Intelligent extractor that handles:
    1. '15.8' (Simple number)
    2. 'Absent', 'Nil' -> 0.0
    3. 'Present', 'Trace' -> 1.0
    4. Ignores ranges like '13.0-17.0'
    """
    # 1. Remove the keyword from the text to look at the REST of the row
    # (Prevents finding numbers inside the keyword itself, though rare)
    # We split by keyword and take the right side (after the match)
    try:
        search_zone = row_text.split(keyword)[-1]
    except:
        search_zone = row_text

    # 2. Handle Textual Results (Common in Urine)
    search_zone_lower = search_zone.lower()
    if "absent" in search_zone_lower or "nil" in search_zone_lower:
        return 0.0
    if "present" in search_zone_lower or "detected" in search_zone_lower:
        return 1.0
    if "trace" in search_zone_lower:
        return 0.5 # Treat trace as small positive

    # 3. Clean "UnitValue" jams (e.g., "mg/dL12.5" -> "mg/dL 12.5")
    # Insert space between Letter and Number
    search_zone = re.sub(r'([a-zA-Z%])(\d)', r'\1 \2', search_zone)

    # 4. Extract Numbers
    # Regex explanation:
    # \d+(\.\d+)?  -> Finds integers or decimals (e.g., 15, 15.8)
    # (?!-|\d)     -> Negative Lookahead: Ensures the number is NOT followed by a hyphen (Range) or another digit
    # But checking for ranges is tricky. Better strategy:
    # Iterate all numbers found. If a number is followed immediately by "-", ignore it.
    
    tokens = search_zone.split()
    candidates = []
    
    for i, token in enumerate(tokens):
        # Remove common units to clean up
        token_clean = token.replace('mg/dl', '').replace('g/dl', '').replace('%', '')
        
        # Check if it looks like a number
        try:
            val = float(token_clean)
            
            # FILTERS:
            # A. Ignore Years (2020-2030)
            if 2020 <= val <= 2030 and "." not in token_clean: continue
            
            # B. Ignore Ranges (If next token is "-" or token contains "-")
            # e.g., "13.0-17.0" -> token might be "13.0-17.0" or "13.0" then "-"
            if "-" in token: continue
            if i+1 < len(tokens) and tokens[i+1] == "-": continue
            
            # C. Ignore Reference Range Starts (Heuristic)
            # If the row has "15.8" then "13.0-17.0", we want 15.8.
            # Usually the Result comes BEFORE the Reference Range.
            candidates.append(val)
        except:
            continue

    if candidates:
        return candidates[0] # Return the FIRST valid number found (Result)
    
    return None

def extract_patient_details(pdf_path):
    details = {"AGE": 30, "GENDER": "Male"} # Defaults
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = pdf.pages[0].extract_text()
            # Age: Look for "34Y", "Age: 34"
            age_match = re.search(r"(?:Age|Yrs?)[\s:.-]*(\d{2})", text, re.IGNORECASE)
            if not age_match: age_match = re.search(r"(\d{2})\s*Y", text, re.IGNORECASE)
            if age_match: details["AGE"] = int(age_match.group(1))
            
            # Gender
            if re.search(r"\b(Female|F)\b", text, re.IGNORECASE) or "/F" in text:
                details["GENDER"] = "Female"
            elif re.search(r"\b(Male|M)\b", text, re.IGNORECASE) or "/M" in text:
                details["GENDER"] = "Male"
    except: pass
    return details

def parse_lab_report(pdf_path):
    extracted = extract_patient_details(pdf_path)
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extract text line-by-line (Robust for messy tables)
            text = page.extract_text()
            if not text: continue
            
            lines = text.split('\n')
            
            for line in lines:
                clean_line = clean_text(line)
                
                for key, aliases in LAB_MAPPING.items():
                    # Don't overwrite if we already found a value (assuming first match is best)
                    if key in extracted: continue
                    
                    for alias in aliases:
                        # Exact substring match
                        if alias in clean_line:
                            # Specific fix for "Nucleated Red Blood Cells" triggering "Red Blood Cells"
                            if alias == "red blood cells" and "nucleated" in clean_line:
                                continue
                                
                            val = extract_value_from_row(clean_line, alias)
                            if val is not None:
                                extracted[key] = val
                                break # Stop checking aliases for this key
    
    # Post-Processing: Calculate Ratios if missing (Backend usually does this too, but good to have)
    if "LAB_SGOT" in extracted and "LAB_SGPT" in extracted:
        if "LAB_SGOT_SGPT_RATIO" not in extracted and extracted["LAB_SGPT"] > 0:
            extracted["LAB_SGOT_SGPT_RATIO"] = extracted["LAB_SGOT"] / extracted["LAB_SGPT"]

    return extracted
