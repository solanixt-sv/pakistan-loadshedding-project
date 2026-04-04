"""
Pakistan Load Shedding - ML Training Script
Trains Linear Regression, Random Forest, and XGBoost models,
compares them, and saves the best one as models/model.pkl
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

def run_training():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    DATA_PATH = os.path.join(DATA_DIR, 'load_shedding_data.csv')
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # ── Load data ─────────────────────────────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: {DATA_PATH} not found.")
        return
        
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows from {DATA_PATH}\n")
    
    # ── Feature engineering ───────────────────────────────────────────────────────
    le_city   = LabelEncoder()
    le_area   = LabelEncoder()
    le_season = LabelEncoder()
    le_dow    = LabelEncoder()
    
    df['city_enc']   = le_city.fit_transform(df['city'])
    df['area_enc']   = le_area.fit_transform(df['area'])
    df['season_enc'] = le_season.fit_transform(df['season'])
    df['dow_enc']    = le_dow.fit_transform(df['day_of_week'])
    
    FEATURES = ['city_enc', 'area_enc', 'hour', 'season_enc', 'temperature', 'is_weekend', 'dow_enc']
    TARGET   = 'load_shedding_hours'
    
    X = df[FEATURES]
    y = df[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ── Models ────────────────────────────────────────────────────────────────────
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest':     RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        'XGBoost':           XGBRegressor(n_estimators=300, learning_rate=0.05,
                                          max_depth=6, random_state=42, verbosity=0),
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae   = mean_absolute_error(y_test, preds)
        rmse  = np.sqrt(mean_squared_error(y_test, preds))
        r2    = r2_score(y_test, preds)
        results[name] = {'model': model, 'MAE': mae, 'RMSE': rmse, 'R2': r2}
        print(f"{name:22s}  MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}")
    
    # ── Pick best model (highest R²) ──────────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]['R2'])
    best_model = results[best_name]['model']
    print(f"\n🏆 Best model: {best_name}")
    
    # ── Save artefacts ────────────────────────────────────────────────────────────
    joblib.dump(best_model,  os.path.join(MODEL_DIR, 'model.pkl'))
    joblib.dump(le_city,     os.path.join(MODEL_DIR, 'le_city.pkl'))
    joblib.dump(le_area,     os.path.join(MODEL_DIR, 'le_area.pkl'))
    joblib.dump(le_season,   os.path.join(MODEL_DIR, 'le_season.pkl'))
    joblib.dump(le_dow,      os.path.join(MODEL_DIR, 'le_dow.pkl'))
    joblib.dump(FEATURES,    os.path.join(MODEL_DIR, 'feature_names.pkl'))
    
    print(f"✅ Saved artefacts to: {MODEL_DIR}")
    print("✅ Saved: models/model.pkl  +  encoder files")

if __name__ == "__main__":
    run_training()
