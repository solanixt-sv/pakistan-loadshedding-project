"""
Pakistan Load Shedding - Synthetic Dataset Generator
Generates a realistic 5000-row CSV dataset for ML training.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

random.seed(42)
np.random.seed(42)

def run_generator():
    # ── Config ────────────────────────────────────────────────────────────────────
    CITIES = ['Karachi', 'Lahore', 'Islamabad', 'Peshawar', 'Quetta']
    
    CITY_BASE_SHEDDING = {
        'Karachi':   2.5,
        'Lahore':    3.5,
        'Islamabad': 1.5,
        'Peshawar':  4.0,
        'Quetta':    4.5,
    }
    
    CITY_AREAS = {
        'Karachi': ['Clifton', 'DHA Karachi', 'Gulshan-e-Iqbal', 'Nazimabad', 'Korangi', 'Orangi Town', 'Lyari', 'Malir', 'Saddar', 'SITE Area', 'Gulistan-e-Johar'],
        'Lahore': ['Gulberg', 'DHA Lahore', 'Johar Town', 'Model Town', 'Bahria Town Lahore', 'Wapda Town', 'Allama Iqbal Town', 'Samanabad', 'Lahore Cantt', 'Township'],
        'Islamabad': ['F-8', 'F-10', 'G-11', 'G-13', 'E-11', 'I-8', 'Blue Area', 'Bani Gala', 'DHA Islamabad', 'Bahria Town Islamabad'],
        'Peshawar': ['Hayatabad', 'University Town', 'Peshawar Saddar', 'Peshawar Cantt', 'Karkhano Market', 'Gulbahar', 'Warsak Road', 'Dalazak Road'],
        'Quetta': ['Satellite Town Quetta', 'Jinnah Town', 'Nawa Killi', 'Sariab Road', 'Quetta Cantt', 'Hazara Town', 'Prince Road', 'Zarghoon Road']
    }
    
    AREA_MULTIPLIER = {}
    for a in ['Clifton', 'DHA Karachi', 'Gulberg', 'DHA Lahore', 'Bahria Town Lahore', 'Model Town', 'F-8', 'F-10', 'E-11', 'Blue Area', 'DHA Islamabad', 'Bahria Town Islamabad', 'Hayatabad', 'University Town', 'Peshawar Cantt', 'Jinnah Town', 'Quetta Cantt', 'Zarghoon Road']:
        AREA_MULTIPLIER[a] = 0.6
    for a in ['Gulshan-e-Iqbal', 'Nazimabad', 'Gulistan-e-Johar', 'Saddar', 'Johar Town', 'Wapda Town', 'Allama Iqbal Town', 'Lahore Cantt', 'G-11', 'I-8', 'Bani Gala', 'Peshawar Saddar', 'Gulbahar', 'Satellite Town Quetta', 'Prince Road']:
        AREA_MULTIPLIER[a] = 1.0
    for a in ['Korangi', 'Orangi Town', 'Lyari', 'Malir', 'SITE Area', 'Samanabad', 'Township', 'G-13', 'Karkhano Market', 'Warsak Road', 'Dalazak Road', 'Nawa Killi', 'Sariab Road', 'Hazara Town']:
        AREA_MULTIPLIER[a] = 1.6
    
    SEASON_MULTIPLIER = {
        'Summer':  1.8,
        'Winter':  0.9,
        'Monsoon': 1.3,
        'Spring':  1.0,
    }
    
    def get_season(month: int) -> str:
        if month in [3, 4]:          return 'Spring'
        elif month in [5, 6, 7]:     return 'Summer'
        elif month in [8, 9, 10]:    return 'Monsoon'
        else:                        return 'Winter'
    
    def get_temperature(city: str, season: str) -> float:
        base = {'Summer': 38, 'Monsoon': 32, 'Spring': 25, 'Winter': 12}[season]
        city_adj = {'Karachi': +3, 'Lahore': +2, 'Islamabad': 0, 'Peshawar': +1, 'Quetta': -3}[city]
        return round(base + city_adj + np.random.normal(0, 2.5), 1)
    
    def generate_load_shedding(city: str, area: str, hour: int, season: str, is_weekend: int,
                               temperature: float) -> float:
        base = CITY_BASE_SHEDDING[city]
        season_mult = SEASON_MULTIPLIER[season]
        if 6 <= hour <= 9:   hour_mult = 1.4
        elif 18 <= hour <= 22: hour_mult = 1.7
        elif 0 <= hour <= 5: hour_mult = 0.4
        else:                hour_mult = 1.0
        weekend_mult = 0.85 if is_weekend else 1.0
        temp_effect = max(0, (temperature - 30) * 0.05)
        area_mult = AREA_MULTIPLIER.get(area, 1.0)
        shedding = base * season_mult * hour_mult * weekend_mult * area_mult + temp_effect
        shedding += np.random.normal(0, 0.4)
        return round(max(0.0, min(12.0, shedding)), 2)
    
    # ── Generate rows ─────────────────────────────────────────────────────────────
    rows = []
    start_date = datetime(2022, 1, 1)
    for _ in range(5000):
        days_offset = random.randint(0, 730)
        date = start_date + timedelta(days=days_offset)
        city = random.choice(CITIES)
        area = random.choice(CITY_AREAS[city])
        hour = random.randint(0, 23)
        season = get_season(date.month)
        temperature = get_temperature(city, season)
        day_of_week = date.strftime('%A')
        is_weekend = 1 if date.weekday() >= 5 else 0
        load_shedding_hours = generate_load_shedding(city, area, hour, season, is_weekend, temperature)
        rows.append({
            'date': date.strftime('%Y-%m-%d'), 'city': city, 'area': area, 'hour': hour,
            'season': season, 'temperature': temperature, 'day_of_week': day_of_week,
            'is_weekend': is_weekend, 'load_shedding_hours': load_shedding_hours,
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values(['date', 'city', 'hour']).reset_index(drop=True)
    
    # ── Save Dataset ──────────────────────────────────────────────────────────────
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    DATA_PATH = os.path.join(DATA_DIR, 'load_shedding_data.csv')
    
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"✅ Dataset generated: {len(df)} rows at {DATA_PATH}")

if __name__ == "__main__":
    run_generator()
print(df.describe())
print("\nSample rows:")
print(df.head(10).to_string())
