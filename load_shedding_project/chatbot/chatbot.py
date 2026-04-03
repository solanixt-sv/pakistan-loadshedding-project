"""
Pakistan Load Shedding - NLP Chatbot
Rule-based keyword matching chatbot with ML model integration.
Works with Urdu/English mixed input (Roman Urdu supported).
"""

import re
import random
import pandas as pd
import numpy as np
import joblib
import os

# ── Load resources ────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_PATH   = os.path.join(_BASE, 'data', 'load_shedding_data.csv')
_MODEL_PATH  = os.path.join(_BASE, 'models', 'model.pkl')
_LE_CITY     = os.path.join(_BASE, 'models', 'le_city.pkl')
_LE_AREA     = os.path.join(_BASE, 'models', 'le_area.pkl')
_LE_SEASON   = os.path.join(_BASE, 'models', 'le_season.pkl')
_LE_DOW      = os.path.join(_BASE, 'models', 'le_dow.pkl')

try:
    _df      = pd.read_csv(_DATA_PATH)
    _model   = joblib.load(_MODEL_PATH)
    _le_city = joblib.load(_LE_CITY)
    _le_area = joblib.load(_LE_AREA)
    _le_sea  = joblib.load(_LE_SEASON)
    _le_dow  = joblib.load(_LE_DOW)
    _READY   = True
except Exception as e:
    _READY = False
    _df    = None

CITIES   = ['Karachi', 'Lahore', 'Islamabad', 'Peshawar', 'Quetta']
SEASONS  = ['Summer', 'Winter', 'Monsoon', 'Spring']
DAYS     = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# ── Keyword maps (Roman Urdu + English) ───────────────────────────────────────
CITY_ALIASES = {
    'karachi':    'Karachi',
    'lahore':     'Lahore',
    'islamabad':  'Islamabad',
    'peshawar':   'Peshawar',
    'quetta':     'Quetta',
}

SEASON_ALIASES = {
    'summer': 'Summer', 'garmi': 'Summer', 'garam': 'Summer',
    'winter': 'Winter', 'sardi': 'Winter', 'thanda': 'Winter',
    'monsoon': 'Monsoon', 'barsaat': 'Monsoon', 'barish': 'Monsoon',
    'spring': 'Spring', 'bahar': 'Spring',
}

INTENTS = {
    'predict':   r'predict|kitni|forecast|kal|اگلے|tomorrow|next',
    'worst':     r'worst|sabse zyada|maximum|max|highest|most',
    'best':      r'best|sabse kam|minimum|min|lowest|least',
    'average':   r'average|avg|mean|typically|generally|usually|normally',
    'peak':      r'peak|worst time|worst hour|peak hour|کون سا وقت|konsa waqt',
    'trend':     r'trend|month|mahina|time period|over time',
    'greeting':  r'^(hi|hello|hey|salam|assalam|adaab|hola)\b',
    'thanks':    r'thank|shukriya|thanks|shukria',
    'help':      r'help|kya pooch|what can|capabilities',
}

def _extract_city(text: str) -> str | None:
    text = text.lower()
    for alias, city in CITY_ALIASES.items():
        if alias in text:
            return city
    return None

def _extract_season(text: str) -> str | None:
    text = text.lower()
    for alias, season in SEASON_ALIASES.items():
        if alias in text:
            return season
    return None

def _extract_hour(text: str) -> int | None:
    m = re.search(r'\b(\d{1,2})\s*(am|pm|baje|AM|PM)?\b', text)
    if m:
        h = int(m.group(1))
        if m.group(2) and 'pm' in m.group(2).lower() and h < 12:
            h += 12
        return min(h, 23)
    return None

def _detect_intent(text: str) -> str:
    text = text.lower()
    for intent, pattern in INTENTS.items():
        if re.search(pattern, text, re.IGNORECASE):
            return intent
    return 'unknown'

def _get_season_now() -> str:
    from datetime import datetime
    m = datetime.now().month
    if m in [3, 4]:    return 'Spring'
    if m in [5, 6, 7]: return 'Summer'
    if m in [8, 9, 10]:return 'Monsoon'
    return 'Winter'

def _predict(city: str, hour: int = 14, season: str = None) -> float:
    if not _READY:
        return -1.0
    if season is None:
        season = _get_season_now()
    try:
        if _df is not None:
            # find a default area for this city
            default_area = _df[_df['city'] == city]['area'].iloc[0]
        else:
            default_area = "Saddar"
            
        city_enc   = _le_city.transform([city])[0]
        area_enc   = _le_area.transform([default_area])[0]
        season_enc = _le_sea.transform([season])[0]
        dow_enc    = _le_dow.transform(['Monday'])[0]
        from datetime import datetime
        is_weekend = 1 if datetime.now().weekday() >= 5 else 0
        from sklearn.preprocessing import LabelEncoder
        temp_map = {'Summer': 38, 'Winter': 12, 'Monsoon': 30, 'Spring': 25}
        temp = temp_map.get(season, 28)
        X = pd.DataFrame([[city_enc, area_enc, hour, season_enc, temp, is_weekend, dow_enc]],
                         columns=['city_enc', 'area_enc', 'hour', 'season_enc',
                                  'temperature', 'is_weekend', 'dow_enc'])
        return round(float(_model.predict(X)[0]), 2)
    except Exception:
        return -1.0

def _city_stats(city: str) -> dict:
    if _df is None:
        return {}
    sub = _df[_df['city'] == city]
    return {
        'avg':  round(sub['load_shedding_hours'].mean(), 2),
        'max':  round(sub['load_shedding_hours'].max(), 2),
        'min':  round(sub['load_shedding_hours'].min(), 2),
        'peak_hour': int(sub.groupby('hour')['load_shedding_hours'].mean().idxmax()),
    }

# ── Main chat function ────────────────────────────────────────────────────────
READY_RESPONSES = {
    'greeting': [
        "Assalam-o-Alaikum! 👋 Main Pakistan Load Shedding Chatbot hoon. City, season ya time ke barey mein poochein!",
        "Hello! 🌟 Load shedding ke barey mein kuch bhi poochein — main help karoon ga!",
    ],
    'thanks': [
        "Koi baat nahi! 😊 Aur kuch poochna ho toh zaroor poochein.",
        "You're welcome! Kisi bhi city ki load shedding info ke liye hamesha haazir hoon! ✅",
    ],
    'help': [
        """**Main ye sawaal samajh sakta hoon:**
- 🏙️ *City-wise prediction:* "Lahore mein kal kitni load shedding hogi?"
- ⏰ *Peak hours:* "Karachi ka peak time kya hai?"
- 📊 *Average:* "Islamabad mein average kitni load shedding hoti hai?"
- 🥵 *Worst city:* "Sabse zyada load shedding kahan hoti hai?"
- 🌡️ *Season:* "Summer mein kitni load shedding hoti hai?"
""",
    ],
}

def get_response(user_input: str) -> str:
    text     = user_input.strip()
    intent   = _detect_intent(text)
    city     = _extract_city(text)
    season   = _extract_season(text)
    hour     = _extract_hour(text) or 14

    if not _READY:
        return "⚠️ Model ya data load nahi hua. Pehle `generate_dataset.py` aur `train_model.py` chalao."

    # ── Greetings / meta ──────────────────────────────────────────────────────
    if intent == 'greeting':
        return random.choice(READY_RESPONSES['greeting'])
    if intent == 'thanks':
        return random.choice(READY_RESPONSES['thanks'])
    if intent == 'help':
        return random.choice(READY_RESPONSES['help'])

    # ── Predict ────────────────────────────────────────────────────────────────
    if intent == 'predict':
        if city:
            hrs = _predict(city, hour, season)
            season_str = season or _get_season_now()
            return (f"📊 **{city}** mein is {season_str} season mein hour **{hour}:00** par "
                    f"predicted load shedding: **{hrs} ghante** hai.")
        else:
            # predict for all cities
            lines = ["📊 **Sabhi cities ki predicted load shedding (aaj, current time):**\n"]
            cur_hour = __import__('datetime').datetime.now().hour
            for c in CITIES:
                h = _predict(c, cur_hour, season)
                lines.append(f"- **{c}**: {h} ghante")
            return "\n".join(lines)

    # ── Worst ─────────────────────────────────────────────────────────────────
    if intent == 'worst':
        if _df is not None:
            if season:
                sub = _df[_df['season'] == season]
                worst_city = sub.groupby('city')['load_shedding_hours'].mean().idxmax()
                avg = round(sub.groupby('city')['load_shedding_hours'].mean().max(), 2)
                return f"🔴 **{season}** mein sabse zyada load shedding **{worst_city}** mein hoti hai — average **{avg} ghante**."
            worst_city = _df.groupby('city')['load_shedding_hours'].mean().idxmax()
            avg = round(_df.groupby('city')['load_shedding_hours'].mean().max(), 2)
            return f"🔴 Overall sabse zyada load shedding **{worst_city}** mein hoti hai — average **{avg} ghante**."

    # ── Best ──────────────────────────────────────────────────────────────────
    if intent == 'best':
        if _df is not None:
            best_city = _df.groupby('city')['load_shedding_hours'].mean().idxmin()
            avg = round(_df.groupby('city')['load_shedding_hours'].mean().min(), 2)
            return f"🟢 Sabse kam load shedding **{best_city}** mein hoti hai — average **{avg} ghante**."

    # ── Average ───────────────────────────────────────────────────────────────
    if intent == 'average':
        if city:
            stats = _city_stats(city)
            return (f"📈 **{city}** ki average load shedding: **{stats['avg']} ghante/hour**\n"
                    f"- Max: {stats['max']} ghante\n- Min: {stats['min']} ghante")
        if season and _df is not None:
            avg = round(_df[_df['season'] == season]['load_shedding_hours'].mean(), 2)
            return f"📈 **{season}** mein average load shedding: **{avg} ghante** per reading."
        if _df is not None:
            lines = ["📈 **Har city ki average load shedding:**\n"]
            avgs = _df.groupby('city')['load_shedding_hours'].mean().round(2)
            for c, a in avgs.items():
                lines.append(f"- **{c}**: {a} ghante")
            return "\n".join(lines)

    # ── Peak hours ────────────────────────────────────────────────────────────
    if intent == 'peak':
        if city:
            stats = _city_stats(city)
            return (f"⏰ **{city}** ka peak hour: **{stats['peak_hour']}:00** — "
                    f"is waqt light jane ka sabse zyada chance hota hai.")
        if _df is not None:
            peak = int(_df.groupby('hour')['load_shedding_hours'].mean().idxmax())
            return f"⏰ Overall sabse zyada load shedding **{peak}:00 baje** hoti hai (evening peak)."

    # ── Trend ─────────────────────────────────────────────────────────────────
    if intent == 'trend' and _df is not None:
        _df['month'] = pd.to_datetime(_df['date']).dt.month
        month_avg = _df.groupby('month')['load_shedding_hours'].mean().round(2)
        worst_m = month_avg.idxmax()
        month_names = {1:'January',2:'February',3:'March',4:'April',5:'May',6:'June',
                       7:'July',8:'August',9:'September',10:'October',11:'November',12:'December'}
        return (f"📅 Sabse mushkil mahina **{month_names[worst_m]}** hai — "
                f"average {month_avg[worst_m]} ghante load shedding.\n"
                f"Summer (May-July) mein load shedding sabse zyada hoti hai.")

    # ── Fallback ──────────────────────────────────────────────────────────────
    return ("🤔 Mujhe samajh nahi aya. Kuch aisa poochein:\n"
            "- *'Lahore mein kal kitni load shedding hogi?'*\n"
            "- *'Sabse zyada load shedding kahan hoti hai?'*\n"
            "- *'Karachi ka peak time kya hai?'*\n"
            "- Type **help** for all options.")


if __name__ == '__main__':
    print("Pakistan Load Shedding Chatbot (type 'exit' to quit)\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in ('exit', 'quit', 'bye'):
            print("Bot: Allah Hafiz! 👋")
            break
        print(f"Bot: {get_response(q)}\n")
