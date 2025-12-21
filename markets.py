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
            .stButton button { width: 100%; border-radius: 8px; font-weight: bold; }
            .stMarkdown p { font-size: 1.1rem; }
        </style>
    """, unsafe_allow_html=True)
load_custom_css()

# --- 2. חיבורים ---
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
        # שימוש במודל החדש והתקין
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        sys_prompt = f"Context: {context_data}. You are APEX, a professional trading mentor. Explain simply in Hebrew."
        chat_history = [{'role': 'user', 'parts': [sys_prompt]}]
        for m in messages:
            role = 'user' if m['role']=='user' else 'model'
            chat_history.append({'role': role, 'parts': [m['content']]})
        return model.generate_content(chat_history).text
    except Exception as e: return f"Error: {str(e)}"

# --- 3. נתונים וחישובים ---
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
    
    st.markdown("### 🔮 APEX Vision (צפי מגמה)")
    st.caption("הקו המקווקו מראה את כיוון המגמה לחודש הקרוב.")
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

# --- 4. האפליקציה הראשית ---
def main_app(username):
    # סרגל צד
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
            st.write("**RSI:** מד חום למניה. מעל 70=רותח, מתחת ל-30=קפוא.")
            st.write("**SMA:** הקו הצהוב. אם המחיר מעליו = מגמת עלייה.")

    # לשוניות
    tabs = st.tabs(["📊 חדר מסחר", "💼 התיק שלי", "📡 סורק הזדמנויות", "🎓 אקדמיה"])

    # --- לשונית 1: חדר מסחר ---
    with tabs[0]:
        c1, c2 = st.columns([1,3])
        ticker = c1.text_input("חפש סימול מניה (למשל TSLA)", "NVDA").upper()
        
        if ticker:
            with st.spinner("מוריד נתונים..."):
                df, info = get_data(ticker, "1y", "1d")
            
            if not df.empty:
                df = add_indicators(df)
                last_price = df['Close'].iloc[-1]
                last_rsi = df['RSI'].iloc[-1]
                
                # כפתור ה-AI המיוחד
                if st.button(f"🤖 נתח לי את {ticker}", type="primary"):
                    with st.spinner("מנתח..."):
                        analysis = get_ai_response([{'role':'user', 'content':f"Analyze {ticker}. Price: {last_price}, RSI: {last_rsi}. Hebrew summary."}], "Analysis")
                        st.success(analysis)

                st.markdown("### נתוני זמן אמת")
                m1, m2, m3 = st.columns(3)
                m1.metric("מחיר אחרון", f"${last_price:.2f}", help="מחיר סגירה אחרון")
                m2.metric("RSI", f"{last_rsi:.1f}", delta_color="inverse" if last_rsi > 70 else "normal", help="מעל 70=יקר, מתחת ל-30=זול")
                m3.metric("שינוי יומי", f"{df['Close'].pct_change().iloc[-1]*100:.2f}%")

                fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='#D4AF37', width=2), name='ממוצע 50'))
                fig.update_layout(title=f"הגרף של {ticker}", template="plotly_dark", height=500)
                st.plotly_chart(fig, use_container_width=True)
                st.divider()
                render_prediction(df)

    # --- לשונית 2: התיק שלי ---
    with tabs[1]:
        st.header("ניהול תיק השקעות")
        with st.expander("➕ הוסף עסקה חדשה ידנית"):
            with st.form("trade_form"):
                c1,c2,c3 = st.columns(3)
                s = c1.text_input("סימול").upper()
                q = c2.number_input("כמות", 1)
                pr = c3.number_input("מחיר קנייה ($)", 0.1)
                if st.form_submit_button("שמור"): 
                    if add_trade(username, s, q, pr): st.success("נשמר!"); time.sleep(1); st.rerun()
        
        df_p = get_portfolio(username)
        if not df_p.empty:
            df_p['CurrentPrice'] = [yf.Ticker(x).fast_info['last_price'] for x in df_p['symbol']]
            df_p['TotalValue'] = df_p['Quantity'] * df_p['CurrentPrice']
            df_p['Profit'] = df_p['TotalValue'] - (df_p['Quantity'] * df_p['AvgPrice'])
            st.dataframe(df_p.style.format({"AvgPrice":"${:.2f}", "CurrentPrice":"${:.2f}", "TotalValue":"${:.2f}", "Profit":"${:.2f}"}), use_container_width=True)
            st.metric("רווח כולל", f"${df_p['Profit'].sum():,.2f}")
        else:
            st.warning("התיק ריק.")

    # --- לשונית 3: סורק ---
    with tabs[2]:
        st.header("🔍 סורק השוק")
        if st.button("הפעל סריקה"):
            res = []
            tickers = ["AAPL","TSLA","NVDA","AMZN","GOOGL","MSFT","AMD","META"]
            prog = st.progress(0)
            for i, t in enumerate(tickers):
                try:
                    d = yf.Ticker(t).history(period="3mo")
                    if not d.empty:
                        delta = d['Close'].diff()
                        up, down = delta.copy(), delta.copy()
                        up[up < 0] = 0; down[down > 0] = 0
                        rs = up.ewm(span=14).mean() / down.abs().ewm(span=14).mean()
                        rsi = 100 - 100 / (1 + rs)
                        last_rsi = rsi.iloc[-1]
                        stat = "🔥 רותח" if last_rsi > 70 else "❄️ קפוא" if last_rsi < 30 else "בינוני"
                        res.append({"מניה":t, "מחיר":f"${d['Close'].iloc[-1]:.2f}", "RSI":f"{last_rsi:.1f}", "סטטוס":stat})
                except: pass
                prog.progress((i+1)/len(tickers))
            st.dataframe(pd.DataFrame(res), use_container_width=True)

    # --- לשונית 4: אקדמיה (התוכן המלא!) ---
    with tabs[3]:
        st.header("🎓 אקדמיית APEX")
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
            for i in range(years * 12):
                months_left = (years * 12) - i
                final_val += monthly * (1+rate/100)**(months_left/12)
            
            st.metric("שווי עתידי מוערך", f"₪{final_val:,.0f}")
            st.caption("* חישוב ריבית דריבית ממוצעת, ללא התחשבות באינפלציה או מס.")

# הרצה במצב "עוקף כניסה" (Admin)
main_app("Admin")
