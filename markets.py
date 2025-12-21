import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import hashlib
import time
from datetime import datetime, timedelta
import google.generativeai as genai
import random
import numpy as np
from sklearn.linear_model import LinearRegression

# --- 1. הגדרות מערכת ועיצוב ---
st.set_page_config(page_title="APEX Terminal", layout="wide", page_icon="💎")

def load_custom_css():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Assistant:wght@400;700&display=swap');
            .stApp { background-color: #0E1117; color: #E6EDF3; font-family: 'Assistant', sans-serif; direction: rtl; }
            h1, h2, h3 { color: #D4AF37 !important; text-align: right; }
            .stMetric { text-align: right !important; }
            /* כפתורים מעוצבים */
            .stButton button { width: 100%; border-radius: 8px; font-weight: bold; }
        </style>
    """, unsafe_allow_html=True)
load_custom_css()

# --- 2. חיבורים (DB & AI) ---
@st.cache_resource
def connect_to_db():
    try:
        if "gcp_service_account" not in st.secrets: return None
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        return client.open("APEX_Database")
    except: return None

def get_ai_response(messages, context_data):
    try:
        if "GOOGLE_API_KEY" not in st.secrets: return "⚠️ חסר מפתח AI"
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        model = genai.GenerativeModel('gemini-pro')
        
        # הוראה למודל לדבר בעברית ולהסביר כמו מורה
        sys_prompt = f"Context: {context_data}. You are APEX, a professional trading mentor. Explain simply in Hebrew."
        chat_history = [{'role': 'user', 'parts': [sys_prompt]}]
        
        for m in messages:
            role = 'user' if m['role']=='user' else 'model'
            chat_history.append({'role': role, 'parts': [m['content']]})
            
        return model.generate_content(chat_history).text
    except Exception as e: return f"Error: {str(e)}"

# --- 3. חישובים ונתונים ---
@st.cache_data(ttl=60)
def get_data(ticker, period, interval):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        return df, stock.info
    except: return pd.DataFrame(), {}

def add_indicators(df):
    if df.empty: return df
    # RSI
    df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().where(df['Close'].diff()>0, 0).rolling(14).mean() / -df['Close'].diff().where(df['Close'].diff()<0, 0).rolling(14).mean())))
    # ממוצעים
    df['SMA_50'] = df['Close'].rolling(50).mean()
    # בולינגר
    df['BB_Upper'] = df['Close'].rolling(20).mean() + (df['Close'].rolling(20).std() * 2)
    df['BB_Lower'] = df['Close'].rolling(20).mean() - (df['Close'].rolling(20).std() * 2)
    return df

# --- 4. פונקציות ליבה ---
def render_prediction(df):
    if len(df) < 30: return
    # הכנת נתונים לחיזוי
    df_p = df.copy().reset_index()
    df_p['DateNum'] = df_p['Date'].apply(lambda x: x.toordinal())
    X = df_p[['DateNum']]; y = df_p['Close']
    model = LinearRegression().fit(X, y)
    
    # חיזוי
    future_dates = [df_p['Date'].iloc[-1] + timedelta(days=i) for i in range(1, 31)]
    future_X = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    pred = model.predict(future_X)
    
    # תצוגה
    st.markdown("### 🔮 APEX Vision (צפי מגמה)")
    st.caption("הקו המקווקו מראה את כיוון המגמה לחודש הקרוב לפי אלגוריתם ליניארי.")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='מחיר בפועל', line=dict(color='#00C805')))
    fig.add_trace(go.Scatter(x=future_dates, y=pred, name='תחזית מגמה', line=dict(color='#D4AF37', dash='dot')))
    fig.update_layout(template="plotly_dark", height=300, margin=dict(t=10,b=10,l=0,r=0))
    st.plotly_chart(fig, use_container_width=True)

def add_trade(u, s, q, p):
    sh = connect_to_db()
    if not sh: return False
    try: sh.worksheet("trades").append_row([u, s, int(q), float(p), str(datetime.now())]); return True
    except: return False

def get_portfolio(u):
    sh = connect_to_db()
    if not sh: return pd.DataFrame()
    try:
        df = pd.DataFrame(sh.worksheet("trades").get_all_records())
        if df.empty: return pd.DataFrame()
        df['username'] = df['username'].astype(str)
        udf = df[df['username'] == str(u)].copy()
        if udf.empty: return pd.DataFrame()
        udf['quantity'] = pd.to_numeric(udf['quantity']); udf['price'] = pd.to_numeric(udf['price'])
        return udf.groupby('symbol').apply(lambda x: pd.Series({'Quantity': x['quantity'].sum(), 'AvgPrice': (x['quantity']*x['price']).sum()/x['quantity'].sum()})).reset_index()
    except: return pd.DataFrame()

# --- 5. האפליקציה הראשית ---
def main_app(username):
    # סרגל צד חכם
    with st.sidebar:
        st.title("💎 APEX PRO")
        st.caption(f"מחובר כ: {username}")
        st.markdown("---")
        st.markdown("### 🤖 העוזר האישי")
        if p := st.chat_input("שאל אותי משהו..."):
            with st.spinner("חושב..."):
                st.info(get_ai_response([{'role':'user', 'content':p}], "General Q&A"))
        
        st.markdown("---")
        with st.expander("❓ מקרא מהיר"):
