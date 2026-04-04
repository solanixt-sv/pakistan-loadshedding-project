# Pakistan Load Shedding Predictor & Analyzer

This repository contains a Streamlit app for predicting load shedding duration in major Pakistani cities.

## 📁 Project structure

- `app.py`: Main Streamlit dashboard and prediction logic
- `requirements.txt`: Python dependencies
- `data/load_shedding_data.csv`: dataset used for analytics
- `models/model.pkl`: trained ML model
- `models/le_city.pkl`: city label encoder
- `models/le_season.pkl`: season label encoder
- `models/le_dow.pkl`: day-of-week label encoder
- `models/feature_names.pkl`: feature names (optional)
- `data/generate_dataset.py`, `models/train_model.py`: data preparation and training scripts

## 🚀 Run locally

1. Activate your virtual environment:

   ```powershell
   cd "c:\Users\HP\OneDrive\Desktop\final year project"
   & ".venv\Scripts\Activate.ps1"
   ```

2. Go into app folder:

   ```powershell
   cd "c:\Users\HP\OneDrive\Desktop\final year project\load_shedding_project"
   ```

3. Start Streamlit:

   ```powershell
   streamlit run app.py
   ```

4. Open browser:

   - http://localhost:8501

## 🧾 Verification

- Confirm you can see the Streamlit dashboard.
- Select city, hour, season, temperature, weekend and day of week.
- Observe predicted load shedding hours and alert state.

## 🛠️ Troubleshooting

- If `localhost refused to connect`, confirm the terminal is still running and no errors are shown.
- If the app exits immediately, run with:

  ```powershell
  streamlit run app.py --server.headless true --server.enableCORS false
  ```

- Ensure all model files exist in `models/` and data in `data/`.

## 🏁 Submission note

Deadline: 2 April 2026
Trainer: Sir Nasir Hussain
TA: Muhammad Ahmed Raza

Project is ready for evaluation.
