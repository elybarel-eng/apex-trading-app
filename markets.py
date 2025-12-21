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

# --- 1. הגדרות מערכת ---
st.set_page_config(page_title="APEX Terminal", layout="wide", page_icon="💎")

# --- 2. עיצוב ---
def load_custom_css():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
            .stApp { background-color: #0E1117; color: #E6EDF3; font-family: 'Inter', sans-serif; }
            section[data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
            h1, h2, h3 { color: #D4AF37 !important; letter-spacing: 0.5px; }
            .stButton button { background: linear-gradient(45deg, #D4AF37, #F4CF57); color: #000; font-weight: 700; border: none; border-radius: 6px; }
            [data-testid="stMetricValue"] { color: #E6EDF3 !important; font-weight: 700; }
            [data-testid="stMetricLabel"] { color: #A0A0A0 !important; }
        </style>
    """, unsafe_allow_html=True)
load_custom_css()

# --- 3. חיבורים ---
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
        chat_history = [{'role': 'user', 'parts': [f"Context: {context_data}. Be concise."]}]
        for m in messages:
            role = 'user' if m['role']=='user' else 'model'
            chat_history.append({'role': role, 'parts': [m['content']]})
        return model.generate_content(chat_history).text
    except Exception as e: return f"Error: {str(e)}"

# --- 4. נתונים ---
@st.cache_data(ttl=60)
def get_data(ticker, period, interval):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        return df, stock.info
    except: return pd.DataFrame(), {}

def add_indicators(df):
    if df.empty: return df
    df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().where(df['Close'].diff()>0, 0).rolling(14).mean() / -df['Close'].diff().where(df['Close'].diff()<0, 0).rolling(14).mean())))
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['BB_Upper'] = df['Close'].rolling(20).mean() + (df['Close'].rolling(20).std() * 2)
    df['BB_Lower'] = df['Close'].rolling(20).mean() - (df['Close'].rolling(20).std() * 2)
    return df

def render_prediction(df):
    if len(df) < 30: return
    df_p = df.copy().reset_index()
    df_p['DateNum'] = df_p['Date'].apply(lambda x: x.toordinal())
    X = df_p[['DateNum']]; y = df_p['Close']
    model = LinearRegression().fit(X, y)
    future_dates = [df_p['Date'].iloc[-1] + timedelta(days=i) for i in range(1, 31)]
    future_X = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    pred = model.predict(future_X)
    
    c1, c2 = st.columns([1, 3])
    c1.metric("צפי ל-30 יום", f"${pred[-1]:.2f}", f"{((pred[-1]-y.iloc[-1])/y.iloc[-1])*100:.2f}%")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='History', line=dict(color='#00C805')))
    fig.add_trace(go.Scatter(x=future_dates, y=pred, name='Forecast', line=dict(color='#D4AF37', dash='dot')))
    fig.update_layout(template="plotly_dark", height=300, margin=dict(t=10,b=10,l=0,r=0))
    c2.plotly_chart(fig, use_container_width=True)

# --- 5. פורטפוליו ---
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

# --- 6. אפליקציה ראשית ---
def main_app(username):
    with st.sidebar:
        st.title("💎 APEX PRO"); st.caption(f"User: {username}")
        if p := st.chat_input("Ask AI..."):
            st.info(get_ai_response([{'role':'user', 'content':p}], "General"))

    tabs = st.tabs(["📊 Market", "💼 Portfolio", "📡 Scanner", "🎓 Academy"])

    with tabs[0]: # MARKET
        if t := st.text_input("Symbol", "NVDA").upper():
            df, _ = get_data(t, "1y", "1d")
            if not df.empty:
                df = add_indicators(df)
                st.metric("Price", f"${df['Close'].iloc[-1]:.2f}", f"{df['Close'].pct_change().iloc[-1]*100:.2f}%")
                st.plotly_chart(go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], close=df['Close'], high=df['High'], low=df['Low'])]), use_container_width=True)
                render_prediction(df)

    with tabs[1]: # PORTFOLIO
        with st.expander("➕ Add Trade"):
            c1,c2,c3 = st.columns(3)
            s = c1.text_input("Sym").upper(); q = c2.number_input("Qty",1); pr = c3.number_input("Price",0.1)
            if st.button("Save"): 
                if add_trade(username, s, q, pr): st.success("Saved!"); st.rerun()
        
        df_p = get_portfolio(username)
        if not df_p.empty:
            df_p['Current'] = [yf.Ticker(x).fast_info['last_price'] for x in df_p['symbol']]
            df_p['Val'] = df_p['Quantity'] * df_p['Current']
            st.dataframe(df_p)

    with tabs[2]: # SCANNER
        if st.button("Scan"):
            res = []
            for t in ["AAPL","TSLA","NVDA","AMZN","GOOGL"]:
                d = yf.Ticker(t).history(period="1mo")
                if not d.empty: res.append({"Sym":t, "Price":d['Close'].iloc[-1]})
            st.dataframe(res)

    with tabs[3]: # ACADEMY - כאן כל החומר הלימודי!
        st.header("🎓 אקדמיית המסחר APEX")
        st.markdown("כאן תמצא את כל הידע הדרוש כדי להפוך מסוחר מתחיל למקצוען.")
        
        study_tabs = st.tabs(["📘 יסודות", "📈 ניתוח טכני", "🧠 פסיכולוגיה", "🧮 מחשבון"])
        
        with study_tabs[0]: # יסודות
            st.subheader("פרק א': שוק ההון למתחילים")
            with st.expander("מהי מניה?"):
                st.write("""
                מניה היא חלק בבעלות על חברה. כשאתה קונה מניה של אפל, אתה הופך להיות שותף (קטן מאוד) באפל.
                - **למה המניה עולה?** כי אנשים מאמינים שהחברה תרוויח יותר בעתיד.
                - **למה המניה יורדת?** כי אנשים חוששים שהרווחים ירדו.
                """)
            with st.expander("לונג (Long) מול שורט (Short)"):
                st.info("**Long:** קונים בזול, מחכים שהמחיר יעלה, מוכרים ביוקר.")
                st.error("**Short:** מוכרים מניה שאין לנו (בהלוואה), מחכים שהמחיר ירד, וקונים אותה חזרה בזול.")
            with st.expander("סוגי הוראות מסחר (Market vs Limit)"):
                st.write("""
                * **Market:** קנה עכשיו בכל מחיר שיש בשוק (מהיר אבל מסוכן).
                * **Limit:** קנה רק אם המחיר הוא X או נמוך יותר (בטוח יותר, אבל אולי לא תקבל את המניה).
                * **Stop Loss:** פקודה אוטומטית למכור אם הפסדת יותר מדי (חובה לכל סוחר!).
                """)

        with study_tabs[1]: # טכני
            st.subheader("פרק ב': הארגז הכלים הטכני")
            st.write("הגרפים לא משקרים. הנה הכלים שיעזרו לך לקרוא אותם:")
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### RSI (מדד העוצמה היחסית)")
                st.write("""
                מודד האם המניה "התעייפה".
                * **מעל 70:** קניית יתר (Overbought) - סיכוי לירידה 🔻.
                * **מתחת ל-30:** מכירת יתר (Oversold) - סיכוי לעלייה 💚.
                """)
            with c2:
                st.markdown("#### רצועות בולינגר (Bollinger Bands)")
                st.write("""
                מודד תנודתיות.
                * כשהמחיר נוגע ברצועה העליונה: הוא יקר יחסית.
                * כשהמחיר נוגע ברצועה התחתונה: הוא זול יחסית.
                * כשהרצועות מתכווצות: צפויה תנועה חדה בקרוב ("שקט שלפני הסערה").
                """)
            
            st.markdown("---")
            st.markdown("#### זיהוי מגמות (Trend)")
            st.write("החוק הכי חשוב: **Trend is your Friend**.")
            st.write("אל תנסה לתפוס סכין נופלת. קנה כשהמגמה בעלייה (Higher Highs) ומכור כשהיא בירידה (Lower Lows).")

        with study_tabs[2]: # פסיכולוגיה
            st.subheader("פרק ג': האויב שבפנים")
            st.warning("80% מההצלחה במסחר היא פסיכולוגיה, רק 20% טכניקה.")
            
            with st.expander("FOMO (פחד להחמיץ)"):
                st.write("""
                ההרגשה ש"כולם עושים כסף חוץ ממני" וגורמת לך לקנות בשיא.
                **הפתרון:** אם המניה כבר טסה 20% היום - פספסת. חכה להזדמנות הבאה. תמיד יש עוד רכבת.
                """)
            with st.expander("מסחר נקמה (Revenge Trading)"):
                st.write("""
                הפסדת כסף? הרצון הטבעי הוא "להחזיר את הכסף מהר" ולהגדיל את ההימור.
                **התוצאה:** מחיקת התיק.
                **הפתרון:** הפסדת? סגור את המחשב ולך לעשות ספורט. מחר יום חדש.
                """)
            with st.expander("ניהול סיכונים (חוק ה-1%)"):
                st.success("""
                לעולם אל תסכן יותר מ-1% מהתיק שלך בעסקה אחת.
                אם יש לך $10,000, המקסימום שאתה מרשה לעצמך להפסיד בעסקה אחת הוא $100.
                זה יבטיח שתשרוד גם רצף של הפסדים.
                """)

        with study_tabs[3]: # מחשבון
            st.subheader("מחשבון הריבית השמינית")
            st.write("ראה כמה הכסף שלך יכול לצמוח:")
            amount = st.number_input("סכום התחלתי (₪)", 10000, 1000000, 50000)
            monthly = st.number_input("הפקדה חודשית (₪)", 0, 50000, 1000)
            years = st.slider("למשך כמה שנים?", 1, 40, 20)
            rate = st.slider("תשואה שנתית ממוצעת (%)", 1, 15, 8)
            
            final_val = amount * (1+rate/100)**years
            # חישוב הפקדות חודשיות (מקורב)
            for i in range(years * 12):
                months_left = (years * 12) - i
                final_val += monthly * (1+rate/100)**(months_left/12)
            
            st.metric("שווי עתידי מוערך", f"₪{final_val:,.0f}")
            st.caption("* חישוב ריבית דריבית ממוצעת, ללא התחשבות באינפלציה או מס.")


# עקיפת מסך כניסה (Admin Mode)
main_app("Admin")
