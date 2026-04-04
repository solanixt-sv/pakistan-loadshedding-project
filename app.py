"""
Pakistan Load Shedding Predictor & Analyzer
Complete Streamlit Dashboard — app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import sys
import subprocess
from datetime import datetime, date

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pakistan Load Shedding Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE, 'data', 'load_shedding_data.csv')
MODEL_PATH = os.path.join(BASE, 'models', 'model.pkl')
LE_CITY    = os.path.join(BASE, 'models', 'le_city.pkl')
LE_AREA    = os.path.join(BASE, 'models', 'le_area.pkl')
LE_SEASON  = os.path.join(BASE, 'models', 'le_season.pkl')
LE_DOW     = os.path.join(BASE, 'models', 'le_dow.pkl')

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1527 50%, #0a1628 100%);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1a2e 0%, #0a1120 100%);
    border-right: 1px solid rgba(99,179,237,0.15);
}

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(13,24,45,0.9), rgba(15,28,52,0.8));
    border: 1px solid rgba(99,179,237,0.25);
    border-radius: 16px;
    padding: 20px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
}

[data-testid="metric-container"]:hover {
    border-color: rgba(99,179,237,0.5);
    transform: translateY(-2px);
    transition: all 0.3s ease;
}

/* Headers */
h1 { color: #e2e8f0 !important; }
h2, h3 { color: #cbd5e0 !important; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(13,24,45,0.6);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid rgba(99,179,237,0.15);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #94a3b8 !important;
    font-weight: 500;
    transition: all 0.2s;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
    color: white !important;
    box-shadow: 0 4px 15px rgba(59,130,246,0.4);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6, #1d4ed8);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 24px;
    font-weight: 600;
    font-size: 14px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(59,130,246,0.35);
}

.stButton > button:hover {
    background: linear-gradient(135deg, #60a5fa, #3b82f6);
    box-shadow: 0 6px 20px rgba(59,130,246,0.5);
    transform: translateY(-1px);
}

/* Select boxes & sliders */
.stSelectbox > div > div, .stSlider > div { border-radius: 8px; }

/* Chat messages */
.chat-user {
    background: linear-gradient(135deg, rgba(59,130,246,0.2), rgba(37,99,235,0.15));
    border: 1px solid rgba(59,130,246,0.3);
    border-radius: 16px 16px 4px 16px;
    padding: 12px 16px;
    margin: 8px 0 8px 20%;
    color: #e2e8f0;
}
.chat-bot {
    background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(5,150,105,0.1));
    border: 1px solid rgba(16,185,129,0.25);
    border-radius: 16px 16px 16px 4px;
    padding: 12px 16px;
    margin: 8px 20% 8px 0;
    color: #e2e8f0;
}
.chat-label-user { text-align: right; color: #60a5fa; font-size: 12px; font-weight: 600; margin-bottom: 2px; }
.chat-label-bot  { color: #34d399; font-size: 12px; font-weight: 600; margin-bottom: 2px; }

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, rgba(37,99,235,0.3), rgba(124,58,237,0.2));
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 20px;
    padding: 30px 36px;
    margin-bottom: 24px;
    backdrop-filter: blur(10px);
}

.alert-card {
    border-radius: 12px;
    padding: 14px 18px;
    margin: 8px 0;
    font-weight: 500;
}
.alert-high   { background: rgba(239,68,68,0.15);  border: 1px solid rgba(239,68,68,0.3);  color: #fca5a5; }
.alert-medium { background: rgba(245,158,11,0.15); border: 1px solid rgba(245,158,11,0.3); color: #fcd34d; }
.alert-low    { background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.3); color: #6ee7b7; }

/* Divider */
hr { border-color: rgba(99,179,237,0.1) !important; }

/* Plotly charts dark */
.js-plotly-plot .plotly { background: transparent !important; }

/* Input box */
.stTextInput > div > div > input {
    background: rgba(13,24,45,0.8);
    border: 1px solid rgba(99,179,237,0.3);
    border-radius: 12px;
    color: #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

# ── Load data & model ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['month_name'] = df['date'].dt.strftime('%b')
    return df

@st.cache_resource
def load_model():
    try:
        model   = joblib.load(MODEL_PATH)
        le_city = joblib.load(LE_CITY)
        le_area = joblib.load(LE_AREA)
        le_sea  = joblib.load(LE_SEASON)
        le_dow  = joblib.load(LE_DOW)
        return model, le_city, le_area, le_sea, le_dow
    except Exception:
        return None, None, None, None, None

# ── Initialization ────────────────────────────────────────────────────────────
try:
    from data.generate_dataset import run_generator
    from models.train_model import run_training
    
    if not os.path.exists(DATA_PATH):
        with st.spinner("Initializing complete dataset (First run only)..."):
            run_generator()
            st.success("✅ Dataset generated successfully.")
        st.cache_data.clear()
    
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Training Machine Learning models (First run only)..."):
            run_training()
            st.success("✅ ML Model trained successfully.")
        st.cache_resource.clear()
        
    df = load_data()
    model, le_city, le_area, le_sea, le_dow = load_model()

except Exception as e:
    st.error("🚨 Startup Error Detected")
    st.exception(e)
    st.stop()

CITIES  = ['Karachi', 'Lahore', 'Islamabad', 'Peshawar', 'Quetta']
CITY_AREAS = {
    'Karachi': ['Clifton', 'DHA Karachi', 'Gulshan-e-Iqbal', 'Nazimabad', 'Korangi', 'Orangi Town', 'Lyari', 'Malir', 'Saddar', 'SITE Area', 'Gulistan-e-Johar'],
    'Lahore': ['Gulberg', 'DHA Lahore', 'Johar Town', 'Model Town', 'Bahria Town Lahore', 'Wapda Town', 'Allama Iqbal Town', 'Samanabad', 'Lahore Cantt', 'Township'],
    'Islamabad': ['F-8', 'F-10', 'G-11', 'G-13', 'E-11', 'I-8', 'Blue Area', 'Bani Gala', 'DHA Islamabad', 'Bahria Town Islamabad'],
    'Peshawar': ['Hayatabad', 'University Town', 'Peshawar Saddar', 'Peshawar Cantt', 'Karkhano Market', 'Gulbahar', 'Warsak Road', 'Dalazak Road'],
    'Quetta': ['Satellite Town Quetta', 'Jinnah Town', 'Nawa Killi', 'Sariab Road', 'Quetta Cantt', 'Hazara Town', 'Prince Road', 'Zarghoon Road']
}

if df is not None and not df.empty:
    CITIES = sorted(df['city'].unique().tolist())
    CITY_AREAS = df.groupby('city')['area'].apply(lambda x: sorted(list(set(x)))).to_dict()

SEASONS = ['Summer', 'Winter', 'Monsoon', 'Spring']

CITY_COLORS = {
    'Karachi':   '#3b82f6',
    'Lahore':    '#10b981',
    'Islamabad': '#f59e0b',
    'Peshawar':  '#ef4444',
    'Quetta':    '#8b5cf6',
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(13,24,45,0.6)',
    font=dict(color='#94a3b8', family='Inter'),
    title_font=dict(color='#e2e8f0', size=16),
    xaxis=dict(gridcolor='rgba(99,179,237,0.1)', linecolor='rgba(99,179,237,0.2)'),
    yaxis=dict(gridcolor='rgba(99,179,237,0.1)', linecolor='rgba(99,179,237,0.2)'),
    margin=dict(l=10, r=10, t=40, b=10),
)

# ── Helper: predict ────────────────────────────────────────────────────────────
def predict_shedding(city: str, area: str, hour: int, season: str, temperature: float,
                     is_weekend: int, day_of_week: str) -> float:
    if model is None:
        return -1.0
    try:
        ce  = le_city.transform([city])[0]
        ae  = le_area.transform([area])[0]
        se  = le_sea.transform([season])[0]
        de  = le_dow.transform([day_of_week])[0]
        X   = pd.DataFrame([[ce, ae, hour, se, temperature, is_weekend, de]],
                           columns=['city_enc','area_enc','hour','season_enc',
                                    'temperature','is_weekend','dow_enc'])
        return round(float(model.predict(X)[0]), 2)
    except ValueError: # Occurs if city or area is not in label encoders (Newly added Data)
        return -2.0
    except Exception:
        return -1.0

def severity_badge(hours: float) -> str:
    if hours >= 6:   return "🔴 Critical"
    if hours >= 3.5: return "🟡 High"
    if hours >= 1.5: return "🟠 Medium"
    return "🟢 Low"

def alert_class(hours: float) -> str:
    if hours >= 6:   return "alert-high"
    if hours >= 3.5: return "alert-medium"
    return "alert-low"

def get_season_now() -> str:
    m = datetime.now().month
    if m in [3, 4]:    return 'Spring'
    if m in [5, 6, 7]: return 'Summer'
    if m in [8, 9, 10]:return 'Monsoon'
    return 'Winter'

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px;'>
        <div style='font-size:48px; margin-bottom:8px;'>⚡</div>
        <div style='font-size:18px; font-weight:700; color:#e2e8f0;'>Load Shedding</div>
        <div style='font-size:13px; color:#60a5fa;'>Pakistan Analyzer</div>
    </div>
    <hr/>
    """, unsafe_allow_html=True)

    st.markdown("### 🏙️ City Filter")
    city_filter = st.multiselect(
        "Select Cities", CITIES, default=CITIES,
        help="Filter data by city"
    )

    st.markdown("### 🗓️ Date Range")
    if df is not None:
        min_d = df['date'].min().date()
        max_d = df['date'].max().date()
    else:
        min_d = date(2022, 1, 1)
        max_d = date(2023, 12, 31)

    date_range = st.date_input(
        "Select Range", value=(min_d, max_d),
        min_value=min_d, max_value=max_d
    )

    st.markdown("### 🌡️ Season Filter")
    season_filter = st.multiselect(
        "Select Seasons", SEASONS, default=SEASONS,
    )

    st.markdown("---")
    if df is not None:
        st.markdown(f"""
        <div style='background:rgba(59,130,246,0.1); border:1px solid rgba(59,130,246,0.2);
                    border-radius:10px; padding:12px; text-align:center;'>
            <div style='color:#60a5fa; font-size:12px; font-weight:600;'>DATASET</div>
            <div style='color:#e2e8f0; font-size:22px; font-weight:700;'>{len(df):,}</div>
            <div style='color:#94a3b8; font-size:11px;'>Records</div>
        </div>
        """, unsafe_allow_html=True)

    if model is None:
        st.warning("⚠️ Model not loaded. Run `train_model.py` first.")
    else:
        st.success("✅ ML Model Loaded")

# ── Filter data ────────────────────────────────────────────────────────────────
if df is not None:
    fdf = df[df['city'].isin(city_filter) & df['season'].isin(season_filter)].copy()
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        fdf = fdf[(fdf['date'].dt.date >= date_range[0]) &
                  (fdf['date'].dt.date <= date_range[1])]
else:
    fdf = None

# ══════════════════════════════════════════════════════════════════════════════
# Hero Banner
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
    <div style='display:flex; align-items:center; gap:16px;'>
        <div style='font-size:52px;'>⚡</div>
        <div>
            <h1 style='margin:0; font-size:28px; font-weight:800; color:#e2e8f0;'>
                Pakistan Load Shedding Predictor & Analyzer
            </h1>
            <p style='margin:6px 0 0; color:#94a3b8; font-size:14px;'>
                AI-powered dashboard for electricity outage analysis, prediction & insights across major Pakistan cities
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# KPI Row
# ══════════════════════════════════════════════════════════════════════════════
if fdf is not None and not fdf.empty:
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📊 Avg Load Shedding",
                  f"{fdf['load_shedding_hours'].mean():.2f} hrs",
                  f"±{fdf['load_shedding_hours'].std():.2f}")
    with col2:
        worst_city = fdf.groupby('city')['load_shedding_hours'].mean().idxmax()
        worst_val  = fdf.groupby('city')['load_shedding_hours'].mean().max()
        st.metric("🔴 Worst City", worst_city, f"{worst_val:.2f} hrs avg")
    with col3:
        best_city = fdf.groupby('city')['load_shedding_hours'].mean().idxmin()
        best_val  = fdf.groupby('city')['load_shedding_hours'].mean().min()
        st.metric("🟢 Best City", best_city, f"{best_val:.2f} hrs avg")
    with col4:
        peak_hour = int(fdf.groupby('hour')['load_shedding_hours'].mean().idxmax())
        st.metric("⏰ Peak Hour", f"{peak_hour}:00", "Worst outage time")
    with col5:
        worst_season = fdf.groupby('season')['load_shedding_hours'].mean().idxmax()
        st.metric("🌡️ Worst Season", worst_season, "Highest outages")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "🗺️ City & Area Analysis",
    "🔮 Prediction",
    "📅 Trends",
    "🤖 AI Chatbot",
    "➕ Add Data",
])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    if fdf is None or fdf.empty:
        st.warning("No data found. Run `generate_dataset.py` first.")
    else:
        c1, c2 = st.columns(2)

        # Bar chart: city-wise average
        with c1:
            city_avg = fdf.groupby('city')['load_shedding_hours'].mean().reset_index()
            city_avg.columns = ['City', 'Avg Hours']
            city_avg = city_avg.sort_values('Avg Hours', ascending=True)

            fig = px.bar(city_avg, x='Avg Hours', y='City', orientation='h',
                         title="🏙️ City-wise Average Load Shedding",
                         color='Avg Hours',
                         color_continuous_scale=['#1d4ed8', '#f59e0b', '#ef4444'])
            fig.update_layout(**PLOTLY_LAYOUT)
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        # Pie: season distribution
        with c2:
            season_sum = fdf.groupby('season')['load_shedding_hours'].mean().reset_index()
            fig2 = px.pie(season_sum, names='season', values='load_shedding_hours',
                          title="🌡️ Season-wise Distribution",
                          color_discrete_sequence=['#3b82f6','#f59e0b','#10b981','#8b5cf6'])
            fig2.update_layout(**PLOTLY_LAYOUT)
            fig2.update_traces(textposition='inside', textinfo='percent+label',
                               hole=0.4)
            st.plotly_chart(fig2, use_container_width=True)

        # Heatmap: hour × day
        st.markdown("### 🗓️ Hour × Day of Week Heatmap")
        pivot = fdf.pivot_table(values='load_shedding_hours',
                                index='day_of_week', columns='hour',
                                aggfunc='mean')
        day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        pivot = pivot.reindex([d for d in day_order if d in pivot.index])

        fig3 = px.imshow(pivot, aspect='auto',
                         color_continuous_scale='RdYlBu_r',
                         title="⏰ Load Shedding Heatmap (Hour vs Day)",
                         labels={'x':'Hour of Day','y':'Day','color':'Avg Hours'})
        fig3.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig3, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — CITY ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    if fdf is None or fdf.empty:
        st.warning("No data found.")
    else:
        av_cities = sorted(fdf['city'].unique())
        selected_city = st.selectbox("Choose a City", av_cities, key="city_analysis")
        city_data = fdf[fdf['city'] == selected_city]

        kc1, kc2, kc3, kc4 = st.columns(4)
        with kc1: st.metric("📊 Average", f"{city_data['load_shedding_hours'].mean():.2f} hrs")
        with kc2: st.metric("🔺 Maximum", f"{city_data['load_shedding_hours'].max():.2f} hrs")
        with kc3: st.metric("🔻 Minimum", f"{city_data['load_shedding_hours'].min():.2f} hrs")
        with kc4:
            ph = int(city_data.groupby('hour')['load_shedding_hours'].mean().idxmax())
            st.metric("⏰ Peak Hour", f"{ph}:00")

        ca1, ca2 = st.columns(2)

        # Area bar chart
        with ca1:
            area_avg = city_data.groupby('area')['load_shedding_hours'].mean().reset_index()
            area_avg = area_avg.sort_values('load_shedding_hours')
            fig0 = px.bar(area_avg, x='load_shedding_hours', y='area', orientation='h',
                          title=f"🏘️ {selected_city} — Top Areas by Load Shedding",
                          color='load_shedding_hours',
                          color_continuous_scale=['#10b981', '#f59e0b', '#ef4444'])
            fig0.update_layout(**PLOTLY_LAYOUT)
            fig0.update_coloraxes(showscale=False)
            st.plotly_chart(fig0, use_container_width=True)

        ca1b, ca2b = st.columns(2)

        # Hourly pattern
        with ca1b:
            hr_avg = city_data.groupby('hour')['load_shedding_hours'].mean().reset_index()
            fig = px.line(hr_avg, x='hour', y='load_shedding_hours',
                          title=f"⏰ {selected_city} — Hourly Pattern",
                          markers=True,
                          color_discrete_sequence=[CITY_COLORS.get(selected_city,'#3b82f6')])
            fig.update_layout(**PLOTLY_LAYOUT)
            fig.update_traces(line_width=2.5, marker_size=6)
            st.plotly_chart(fig, use_container_width=True)

        # Season boxplot
        with ca2b:
            fig2 = px.box(city_data, x='season', y='load_shedding_hours',
                          title=f"📦 {selected_city} — Season Boxplot",
                          color='season',
                          color_discrete_sequence=['#3b82f6','#f59e0b','#10b981','#8b5cf6'])
            fig2.update_layout(**PLOTLY_LAYOUT, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        # Temperature scatter
        st.markdown(f"### 🌡️ Temperature vs Load Shedding — {selected_city}")
        fig3 = px.scatter(city_data, x='temperature', y='load_shedding_hours',
                          color='season',
                          trendline='ols',
                          color_discrete_sequence=['#3b82f6','#f59e0b','#10b981','#8b5cf6'],
                          title=f"Temperature Impact on Load Shedding in {selected_city}")
        fig3.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig3, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — PREDICTION TOOL
# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("## 🔮 Load Shedding Prediction Tool")
    st.markdown("*Enter parameters below to predict expected load shedding duration.*")

    p1, p2 = st.columns(2)

    with p1:
        pred_city    = st.selectbox("🏙️ City", CITIES, key="pred_city")
        pred_area    = st.selectbox("🏘️ Area", CITY_AREAS[pred_city], key="pred_area")
        pred_season  = st.selectbox("🌡️ Season", SEASONS,
                                    index=SEASONS.index(get_season_now()), key="pred_season")
        pred_day     = st.selectbox("📅 Day of Week",
                                    ['Monday','Tuesday','Wednesday','Thursday',
                                     'Friday','Saturday','Sunday'], key="pred_day")

    with p2:
        pred_hour    = st.slider("⏰ Hour of Day", 0, 23, datetime.now().hour, key="pred_hour",
                                 format="%d:00")
        pred_temp    = st.slider("🌡️ Temperature (°C)", -5, 50, 30, key="pred_temp")
        pred_weekend = st.checkbox("🏖️ Is Weekend?",
                                   value=datetime.now().weekday() >= 5, key="pred_weekend")

    if st.button("⚡ Predict Load Shedding", key="predict_btn"):
        if model is None:
            st.error("Model not loaded! Run `train_model.py` first.")
        else:
            result = predict_shedding(
                pred_city, pred_area, pred_hour, pred_season, pred_temp,
                1 if pred_weekend else 0, pred_day
            )
            
            if result == -2.0:
                st.error("⚠️ The selected City or Area is new and not recognized by the Machine Learning model. Please Retrain the model to include it.")
            elif result == -1.0:
                st.error("❌ Model prediction failed.")
            else:
                severity = severity_badge(result)
                ac = alert_class(result)
    
                st.markdown("---")
                rc1, rc2, rc3 = st.columns([1, 2, 1])
                with rc2:
                    st.markdown(f"""
                    <div style='text-align:center; background:linear-gradient(135deg,rgba(37,99,235,0.25),rgba(124,58,237,0.15));
                                border:1px solid rgba(99,179,237,0.3); border-radius:20px; padding:32px;'>
                        <div style='font-size:64px; font-weight:800; color:#60a5fa;'>{result}</div>
                        <div style='font-size:18px; color:#94a3b8; margin-top:4px;'>hours predicted</div>
                        <div style='font-size:24px; margin-top:16px;'>{severity}</div>
                        <div style='margin-top:20px; color:#cbd5e0; font-size:14px;'>
                            📍 {pred_city} ({pred_area}) &nbsp;|&nbsp; ⏰ {pred_hour}:00 &nbsp;|&nbsp;
                            🌡️ {pred_season} &nbsp;|&nbsp; 🌡️ {pred_temp}°C
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
                # Show all-city comparison
                st.markdown("---")
                st.markdown("### 🏙️ All Cities Comparison (same conditions, randomly selected area)")
                comp_rows = []
                import random
                for c in CITIES:
                    areas_list = CITY_AREAS.get(c, [])
                    if areas_list:
                        mock_area = areas_list[0]
                        h = predict_shedding(c, mock_area, pred_hour, pred_season, pred_temp,
                                             1 if pred_weekend else 0, pred_day)
                        if h >= 0:
                            comp_rows.append({'City': c, 'Predicted Hours': h,
                                              'Severity': severity_badge(h)})
                
                if comp_rows:
                    comp_df = pd.DataFrame(comp_rows).sort_values('Predicted Hours', ascending=True)
        
                    fig = px.bar(comp_df, x='Predicted Hours', y='City', orientation='h',
                                 color='Predicted Hours',
                                 color_continuous_scale=['#10b981', '#f59e0b', '#ef4444'],
                                 title="City-wise Predicted Load Shedding")
                    fig.update_layout(**PLOTLY_LAYOUT)
                    fig.update_coloraxes(showscale=False)
                    st.plotly_chart(fig, use_container_width=True)
        
                    # Download prediction
                    st.download_button(
                        "📥 Download Prediction Report",
                        data=comp_df.to_csv(index=False),
                        file_name=f"prediction_{pred_city}_comparison.csv",
                        mime="text/csv",
                    )

# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — TRENDS
# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    if fdf is None or fdf.empty:
        st.warning("No data found.")
    else:
        st.markdown("## 📅 Time-Series & Trend Analysis")

        # Monthly trend (all cities)
        monthly = fdf.groupby(['month', 'month_name', 'city'])['load_shedding_hours'].mean().reset_index()
        monthly = monthly.sort_values('month')

        fig = px.line(monthly, x='month_name', y='load_shedding_hours',
                      color='city',
                      markers=True,
                      title="📅 Monthly Load Shedding Trend by City",
                      color_discrete_map=CITY_COLORS)
        fig.update_layout(**PLOTLY_LAYOUT, xaxis_title="Month", yaxis_title="Avg Hours")
        fig.update_traces(line_width=2, marker_size=7)
        st.plotly_chart(fig, use_container_width=True)

        t1, t2 = st.columns(2)

        # Weekend vs weekday
        with t1:
            wd = fdf.groupby(['city', 'is_weekend'])['load_shedding_hours'].mean().reset_index()
            wd['Type'] = wd['is_weekend'].map({1: 'Weekend', 0: 'Weekday'})
            fig2 = px.bar(wd, x='city', y='load_shedding_hours', color='Type',
                          barmode='group',
                          title="📅 Weekday vs Weekend by City",
                          color_discrete_sequence=['#3b82f6', '#f59e0b'])
            fig2.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig2, use_container_width=True)

        # Season radar chart
        with t2:
            sea_city = fdf.groupby(['season', 'city'])['load_shedding_hours'].mean().reset_index()
            fig3 = px.bar(sea_city, x='season', y='load_shedding_hours', color='city',
                          barmode='group',
                          title="🌡️ Season × City Comparison",
                          color_discrete_map=CITY_COLORS)
            fig3.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig3, use_container_width=True)

        # Raw data download
        st.markdown("---")
        st.markdown("### 📥 Download Filtered Data")
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.dataframe(fdf.head(100).style.format({'load_shedding_hours': '{:.2f}',
                                                      'temperature': '{:.1f}'}),
                         height=280, use_container_width=True)
        with col_b:
            st.download_button(
                "📥 Download CSV",
                data=fdf.to_csv(index=False),
                file_name="load_shedding_filtered.csv",
                mime="text/csv",
            )
            st.metric("Filtered Rows", f"{len(fdf):,}")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 5 — AI CHATBOT
# ──────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("## 🤖 AI Load Shedding Chatbot")
    st.markdown("*Ask in English or Roman Urdu — e.g. 'Lahore mein average kitni load shedding hoti hai?'*")

    # Add chatbot to path
    chatbot_dir = os.path.join(BASE, 'chatbot')
    if chatbot_dir not in sys.path:
        sys.path.insert(0, chatbot_dir)

    try:
        from chatbot import get_response as bot_response
        chatbot_ok = True
    except Exception as e:
        chatbot_ok = False
        chatbot_err = str(e)

    # Init session history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [
            {'role': 'bot', 'text': (
                "Assalam-o-Alaikum! 👋 Main Pakistan Load Shedding Chatbot hoon.\n\n"
                "Mujhse ye pooch sakte ho:\n"
                "- *Karachi mein kal kitni load shedding hogi?*\n"
                "- *Sabse zyada load shedding kahan hoti hai?*\n"
                "- *Summer mein average kitni load shedding hoti hai?*\n"
                "- *Islamabad ka peak time kya hai?*"
            )}
        ]

    # Chat display
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                st.markdown(f"<div class='chat-label-user'>You</div>"
                            f"<div class='chat-user'>{msg['text']}</div>",
                            unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-label-bot'>⚡ Bot</div>"
                            f"<div class='chat-bot'>{msg['text']}</div>",
                            unsafe_allow_html=True)

    st.markdown("---")

    # Input row
    inp_col, btn_col = st.columns([5, 1])
    with inp_col:
        user_input = st.text_input(
            "Your message", placeholder="e.g. Lahore mein kitni load shedding hogi?",
            label_visibility="collapsed", key="chat_input"
        )
    with btn_col:
        send = st.button("Send ➤", key="send_btn")

    # Quick questions
    st.markdown("**Quick Questions:**")
    qc1, qc2, qc3, qc4 = st.columns(4)
    with qc1:
        if st.button("🏙️ Worst city?", key="q1"):
            user_input = "Which city has the worst load shedding?"
            send = True
    with qc2:
        if st.button("⏰ Peak hours?", key="q2"):
            user_input = "What are the peak hours for load shedding?"
            send = True
    with qc3:
        if st.button("🥵 Summer stats?", key="q3"):
            user_input = "Summer mein average kitni load shedding hoti hai?"
            send = True
    with qc4:
        if st.button("📊 All averages?", key="q4"):
            user_input = "Show average load shedding for all cities"
            send = True

    if send and user_input:
        st.session_state.chat_history.append({'role': 'user', 'text': user_input})
        if chatbot_ok:
            response = bot_response(user_input)
        else:
            response = f"❌ Chatbot load error: {chatbot_err}"
        st.session_state.chat_history.append({'role': 'bot', 'text': response})
        st.rerun()

    if st.button("🗑️ Clear Chat", key="clear_btn"):
        st.session_state.chat_history = [
            {'role': 'bot', 'text': "Chat cleared! Kuch aur poochein? 😊"}
        ]
        st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# TAB 6 — ADD DATA
# ──────────────────────────────────────────────────────────────────────────────
with tab6:
    st.markdown("## ➕ Add New Load Shedding Data")
    st.markdown("*Use this form to append real load shedding occurrences to the dataset.*")
    
    ad1, ad2 = st.columns(2)
    with ad1:
        new_date = st.date_input("📅 Date", value=datetime.now())
        
        st.markdown("**Location Details**")
        if st.checkbox("➕ New City (Enter Manually)"):
            add_city_sel = st.text_input("🏙️ New City Name")
        else:
            add_city_sel = st.selectbox("🏙️ City", CITIES, key='add_city_sel')
            
        if st.checkbox("➕ New Area (Enter Manually)"):
            add_area_sel = st.text_input("🏘️ New Area Name")
        else:
            avail_areas = CITY_AREAS.get(add_city_sel, [])
            if avail_areas:
                add_area_sel = st.selectbox("🏘️ Area", avail_areas, key='add_area_sel')
            else:
                add_area_sel = st.text_input("🏘️ Area Name")
                
        new_hour = st.slider("⏰ Hour", 0, 23, 12, format="%d:00", key='add_hour')
        
    with ad2:
        new_temp = st.slider("🌡️ Temperature (°C)", -5.0, 50.0, 30.0, key='add_temp')
        new_hours = st.number_input("⚡ Load Shedding Duration", min_value=0.0, max_value=24.0, value=2.0, step=0.5, key='add_hours', help="In Hours")
        
    if st.button("💾 Save Data", type="primary"):
        m = new_date.month
        if m in [3, 4]:      ns = 'Spring'
        elif m in [5, 6, 7]: ns = 'Summer'
        elif m in [8, 9, 10]:ns = 'Monsoon'
        else:                ns = 'Winter'
        
        ndow = new_date.strftime('%A')
        nweek = 1 if new_date.weekday() >= 5 else 0
        
        new_row = pd.DataFrame([{
            'date': new_date.strftime('%Y-%m-%d'),
            'city': add_city_sel,
            'area': add_area_sel,
            'hour': new_hour,
            'season': ns,
            'temperature': new_temp,
            'day_of_week': ndow,
            'is_weekend': nweek,
            'load_shedding_hours': round(new_hours, 2),
        }])
        
        try:
            new_row.to_csv(DATA_PATH, mode='a', header=False, index=False)
            st.success(f"✅ Data for {add_city_sel} ({add_area_sel}) saved successfully!")
            st.info("The dataset has been updated. You might want to retrain the ML model via the backend to reflect these changes in predictions.")
            load_data.clear() # Clears the streamlit cache function for dataset
            st.rerun()
        except Exception as e:
            st.error(f"Error saving data: {str(e)}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#4a5568; font-size:13px; padding:20px;'>
    ⚡ <strong style='color:#60a5fa;'>Pakistan Load Shedding Predictor & Analyzer</strong>
    &nbsp;|&nbsp; AI & Data Science Course &nbsp;|&nbsp; Made By: Mustafa Mukhtar
</div>
""", unsafe_allow_html=True)