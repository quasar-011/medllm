import pandas as pd
import numpy as np
import xgboost as xgb
import glob
import os
import joblib
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, classification_report

os.makedirs("models", exist_ok=True)

class ClinicalAutoTuner:
    """
    A wrapper to perform rigorous GridSearch training for clinical models.
    """
    def __init__(self, model_name):
        self.model_name = model_name
        self.best_model = None
        
    def tune_and_train(self, X, y, scoring='roc_auc'):
        print(f"\nTuning {self.model_name}...")
        
        param_grid = {
            'max_depth': [3, 4, 6],           # Lower depth = less overfitting
            'learning_rate': [0.01, 0.05, 0.1], # Slower learning = more robust
            'n_estimators': [100, 200],       # Number of trees
            'subsample': [0.8, 1.0],          # Train on subset of data (robustness)
            'colsample_bytree': [0.8, 1.0],   # Train on subset of features
            'scale_pos_weight': [1, 3]        # 3 helps if the class is imbalanced (e.g. Mortality)
        }
        
        xgb_clf = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            n_jobs=1  # Let GridSearch handle parallelization
        )
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        grid = GridSearchCV(
            estimator=xgb_clf,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            verbose=1,
            n_jobs=-1 
        )
        
        grid.fit(X, y)
        
        print(f"Best Params: {grid.best_params_}")
        print(f"Best Cross-Val {scoring.upper()}: {grid.best_score_:.4f}")
        
        self.best_model = grid.best_estimator_
        
        self.best_model.save_model(f"models/{self.model_name}_optimized.json")
        print(f"Saved models/{self.model_name}_optimized.json")
        
        return self.best_model

try:
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features
    y = heart_disease.data.targets
    
    # Cleanup
    X = X.fillna(X.mean())
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = X[col].astype('category').cat.codes
        
    tuner = ClinicalAutoTuner("heart")
    tuner.tune_and_train(X, y)

except Exception as e:
    print(f"Heart Model Failed: {e}")

def load_diabetes_data(folder_path):
    patient_data = []
    files = glob.glob(f"{folder_path}/data-*")
    if not files: return pd.DataFrame()
    
    for file in files:
        try:
            df = pd.read_csv(file, sep='\t', names=['Date', 'Time', 'Code', 'Value'], on_bad_lines='skip')
            df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
            df.dropna(subset=['Value'], inplace=True)
            
            stats = {
                'mean_glucose': df[df['Code'].isin([58, 60, 62, 64])]['Value'].mean(),
                'glucose_variability': df['Value'].std(),
                'n_hypo': (df['Value'] < 70).sum(),
                'n_hyper': (df['Value'] > 200).sum(),
            }
            stats['target'] = 1 if stats['mean_glucose'] > 140 else 0
            
            if not np.isnan(stats['mean_glucose']):
                patient_data.append(stats)
        except: continue
    return pd.DataFrame(patient_data)

df_diab = load_diabetes_data("data/diabetes_raw") 

if not df_diab.empty:
    X_diab = df_diab.drop(columns=['target'])
    y_diab = df_diab['target']
    
    tuner = ClinicalAutoTuner("diabetes")
    tuner.tune_and_train(X_diab, y_diab, scoring='precision') 
else:
    print("Diabetes data missing.")

try:
    path = "data/mimic_demo"
    adm = pd.read_csv(f"{path}/ADMISSIONS.csv")
    pat = pd.read_csv(f"{path}/PATIENTS.csv")
    lab = pd.read_csv(f"{path}/LABEVENTS.csv")
    
    adm['admittime'] = pd.to_datetime(adm['ADMITTIME'])
    pat['dob'] = pd.to_datetime(pat['DOB'])
    merged = pd.merge(adm, pat, on="SUBJECT_ID")
    merged['age'] = (merged['admittime'] - merged['dob']).dt.days // 365
    merged['age'] = np.where(merged['age'] > 100, 91, merged['age'])
    
    target_labs = [51221, 50931, 51006, 51301] 
    lab_filtered = lab[lab['ITEMID'].isin(target_labs)].copy()
    lab_filtered['VALUENUM'] = pd.to_numeric(lab_filtered['VALUENUM'], errors='coerce')
    lab_pivot = lab_filtered.pivot_table(index='HADM_ID', columns='ITEMID', values='VALUENUM', aggfunc='mean')
    lab_pivot.columns = [f'LAB_{c}' for c in lab_pivot.columns]
    
    final_df = pd.merge(merged, lab_pivot, on="HADM_ID", how='left')
    features = ['age', 'GENDER'] + list(lab_pivot.columns)
    X_mimic = final_df[features].copy()
    X_mimic['GENDER'] = X_mimic['GENDER'].apply(lambda x: 1 if x == 'M' else 0)
    X_mimic = X_mimic.fillna(X_mimic.mean())
    y_mimic = final_df['HOSPITAL_EXPIRE_FLAG']
    
    tuner = ClinicalAutoTuner("mimic")
    tuner.tune_and_train(X_mimic, y_mimic, scoring='roc_auc')

except Exception as e:
    print(f"MIMIC Model Failed: {e}")
