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

# --- 2. עיצוב יוקרתי ---
def load_custom_css():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
            .stApp { background-color: #0E1117; color: #E6EDF3; font-family: 'Inter', sans-serif; }
            section[data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
            h1, h2, h3 { color: #D4AF37 !important; letter-spacing: 0.5px; }
            p, li, label, .stMarkdown, a { color: #E6EDF3; }
            a { color: #D4AF37; text-decoration: none; font-weight: bold; }
            .stButton button { background: linear-gradient(45deg, #D4AF37, #F4CF57); color: #000; font-weight: 700; border: none; border-radius: 6px; }
            .stButton button:hover { transform: scale(1.02); box-shadow: 0 4px 12px rgba(212, 175, 55, 0.4); }
            [data-testid="stMetricValue"] { color: #E6EDF3 !important; font-weight: 700; }
            [data-testid="stMetricLabel"] { color: #A0A0A0 !important; }
        </style>
    """, unsafe_allow_html=True)
load_custom_css()

# --- 3. חיבורים (DB & AI) ---
@st.cache_resource
def connect_to_db():
    """חיבור לגוגל שיטס עם טיפול בשגיאות"""
    try:
        if "gcp_service_account" not in st.secrets:
            st.error("❌ שגיאה: המפתח 'gcp_service_account' חסר בקובץ secrets.toml")
            return None
        
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        return client.open("APEX_Database")
    except Exception as e:
        st.error(f"❌ שגיאת חיבור לגוגל שיטס: {e}")
        return None

def get_ai_response(messages, context_data):
    """שליחת בקשה ל-AI"""
    try:
        if "GOOGLE_API_KEY" not in st.secrets:
            return "⚠️ שגיאה: חסר מפתח AI בקובץ הסודות."
        
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        model = genai.GenerativeModel('gemini-pro')
        
        chat_history = [{'role': 'user', 'parts': [f"Context: {context_data}. You are a pro trader named APEX. Be short and sharp."]}]
        for m in messages:
            role = 'user' if m['role']=='user' else 'model'
            chat_history.append({'role': role, 'parts': [m['content']]})
            
        return model.generate_content(chat_history).text
    except Exception as e:
        return f"שגיאת AI: {str(e)}"

# --- 4. פונקציות מסחר וניתוח ---
@st.cache_data(ttl=60)
def get_data(ticker, period, interval):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        return df, stock.info
    except:
        return pd.DataFrame(), {}

def add_indicators(df):
    if df.empty: return df
    df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().where(df['Close'].diff() > 0, 0).rolling(14).mean() / -df['Close'].diff().where(df['Close'].diff() < 0, 0).rolling(14).mean())))
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    df['BB_Upper'] = df['Close'].rolling(20).mean() + (df['Close'].rolling(20).std() * 2)
    df['BB_Lower'] = df['Close'].rolling(20).mean() - (df['Close'].rolling(20).std() * 2)
    return df

def render_prediction(df, ticker):
    st.markdown("### 🔮 APEX Vision (AI Forecast)")
    if len(df) < 30: return

    df_p = df.copy().reset_index()
    df_p['DateNum'] = df_p['Date'].apply(lambda x: x.toordinal())
    
    X = df_p[['DateNum']]
    y = df_p['Close']
    model = LinearRegression().fit(X, y)
    
    future_dates = [df_p['Date'].iloc[-1] + timedelta(days=i) for i in range(1, 31)]
    future_X = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    pred = model.predict(future_X)
    
    chg = ((pred[-1] - y.iloc[-1]) / y.iloc[-1]) * 100
    
    c1, c2 = st.columns([1, 3])
    with c1:
        st.metric("צפי ל-30 יום", f"${pred[-1]:.2f}", f"{chg:.2f}%")
        st.caption(f"אמינות מגמה: {model.score(X, y)*100:.1f}%")

    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='History', line=dict(color='#00C805')))
        fig.add_trace(go.Scatter(x=future_dates, y=pred, name='Forecast', line=dict(color='#D4AF37', dash='dot')))
        fig.update_layout(template="plotly_dark", height=300, margin=dict(t=10,b=10,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)

# --- 5. ניהול משתמשים (כולל התיקון!) ---
def make_hashes(p):
    return hashlib.sha256(str.encode(p)).hexdigest()

def login_user(u, p):
    """פונקציית התחברות מתוקנת - מטפלת בבעיות המרת סוגי נתונים"""
    sh = connect_to_db()
    if not sh: return False
    try:
        ws = sh.worksheet("users")
        records = ws.get_all_records()
        df = pd.DataFrame(records)
        
        if df.empty: return False
        
        # --- התיקון הקריטי: המרה לטקסט ---
        # מוודאים שכל שמות המשתמשים הם טקסט (למקרה שמישהו נרשם עם מספר)
        df['username'] = df['username'].astype(str)
        u = str(u).strip() # ניקוי רווחים והמרה לטקסט
        
        # חיפוש המשתמש
        user_row = df[df['username'] == u]
        if user_row.empty: return False
        
        # השוואת סיסמאות
        stored_pass = str(user_row.iloc[0]['password'])
        input_pass = make_hashes(p)
        
        return stored_pass == input_pass
    except Exception as e:
        st.error(f"שגיאת התחברות: {e}")
        return False

def create_user(u, p):
    sh = connect_to_db()
    if not sh: return False
    try:
        ws = sh.worksheet("users")
        # המרה לטקסט גם בבדיקת הכפילויות
        existing_users = [str(x) for x in ws.col_values(1)]
        
        if str(u) in existing_users:
            return False # המשתמש קיים
            
        ws.append_row([str(u), make_hashes(p), str(datetime.now())])
        return True
    except Exception as e:
        st.error(f"שגיאת יצירה: {e}")
        return False

def add_trade(u, s, q, p):
    sh = connect_to_db()
    if not sh: return False
    try:
        sh.worksheet("trades").append_row([u, s, int(q), float(p), str(datetime.now())])
        return True
    except: return False

def get_portfolio(u):
    sh = connect_to_db()
    if not sh: return pd.DataFrame()
    try:
        records = sh.worksheet("trades").get_all_records()
        if not records: return pd.DataFrame()
        
        df = pd.DataFrame(records)
        # סינון לפי שם משתמש (כטקסט)
        df['username'] = df['username'].astype(str)
        udf = df[df['username'] == str(u)].copy()
        
        if udf.empty: return pd.DataFrame()
        
        # המרות סוגים לחישובים
        udf['quantity'] = pd.to_numeric(udf['quantity'])
        udf['price'] = pd.to_numeric(udf['price'])
        
        # סיכום לפי מניה
        ptf = udf.groupby('symbol').apply(
            lambda x: pd.Series({
                'Quantity': x['quantity'].sum(),
                'AvgPrice': (x['quantity'] * x['price']).sum() / x['quantity'].sum()
            })
        ).reset_index()
        return ptf[ptf['Quantity'] > 0]
    except: return pd.DataFrame()

# --- 6. האפליקציה הראשית ---
def main_app(username):
    # סרגל צד
    with st.sidebar:
        st.title("💎 APEX PRO")
        st.caption(f"User: {username}")
        
        with st.expander("💬 AI Chat", expanded=True):
            if "msgs" not in st.session_state: st.session_state.msgs = []
            for m in st.session_state.msgs[-3:]:
                st.markdown(f"**{'👤' if m['role']=='user' else '🤖'}**: {m['content']}")
            
            if p := st.chat_input("Ask market info..."):
                st.session_state.msgs.append({"role":"user", "content":p})
                r = get_ai_response(st.session_state.msgs, st.session_state.get('ctx', 'General'))
                st.session_state.msgs.append({"role":"assistant", "content":r})
                st.rerun()
                
        if st.button("LOGOUT", type="primary"):
            st.session_state.logged_in = False
            st.rerun()

    # לשוניות
    tabs = st.tabs(["📊 Market", "💼 Portfolio", "🕹️ Simulator", "📡 Scanner", "🎓 Academy"])

    # --- MARKET ---
    with tabs[0]:
        c1, c2 = st.columns([1,3])
        if t := c1.text_input("Symbol", "NVDA").upper():
            with st.spinner("Loading..."):
                df, info = get_data(t, "2y", "1d")
                if not df.empty:
                    df = add_indicators(df)
                    st.session_state.ctx = f"{t}: ${df['Close'].iloc[-1]:.2f}"
                    
                    with st.container(border=True):
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Price", f"${df['Close'].iloc[-1]:.2f}")
                        col2.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
                        col3.metric("Change", f"{df['Close'].pct_change().iloc[-1]*100:.2f}%")

                    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], close=df['Close'], high=df['High'], low=df['Low'])])
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='#D4AF37'), name='SMA 50'))
                    fig.update_layout(template="plotly_dark", height=500, title=f"{t} Chart")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.divider()
                    render_prediction(df, t)

    # --- PORTFOLIO ---
    with tabs[1]:
        st.header("My Vault")
        with st.expander("➕ Add Trade"):
            with st.form("trade"):
                c1,c2,c3 = st.columns(3)
                s = c1.text_input("Symbol").upper()
                q = c2.number_input("Qty", step=1, min_value=1)
                pr = c3.number_input("Price", min_value=0.1)
                if st.form_submit_button("Save"):
                    if add_trade(username, s, q, pr): st.success("Saved!"); time.sleep(1); st.rerun()
        
        df_p = get_portfolio(username)
        if not df_p.empty:
            # הוספת שווי נוכחי
            vals = []
            for sym in df_p['symbol']:
                try: vals.append(yf.Ticker(sym).fast_info['last_price'])
                except: vals.append(0)
            df_p['Current'] = vals
            df_p['Total Value'] = df_p['Quantity'] * df_p['Current']
            df_p['Profit'] = df_p['Total Value'] - (df_p['Quantity'] * df_p['AvgPrice'])
            
            st.dataframe(df_p.style.format({"AvgPrice":"${:.2f}", "Current":"${:.2f}", "Total Value":"${:.2f}", "Profit":"${:.2f}"}), use_container_width=True)
            st.metric("Total Equity", f"${df_p['Total Value'].sum():,.2f}")

    # --- SIMULATOR ---
    with tabs[2]:
        st.header("🕹️ Market Time Machine")
        if 'sim_t' not in st.session_state:
            st.session_state.sim_t = random.choice(["AAPL","TSLA","NVDA","AMZN","AMD"])
            data, _ = get_data(st.session_state.sim_t, "5y", "1d")
            st.session_state.sim_data = data
            st.session_state.sim_idx = random.randint(200, len(data)-100)
            st.session_state.sim_done = False
        
        idx = st.session_state.sim_idx
        vis = st.session_state.sim_data.iloc[idx-100:idx]
        
        fig = go.Figure(data=[go.Candlestick(x=vis.index, open=vis['Open'], close=vis['Close'])])
        fig.update_layout(title=f"Mystery Stock: {st.session_state.sim_t} (Hidden Date)", template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
        c1,c2,c3 = st.columns(3)
        if not st.session_state.sim_done:
            if c1.button("BUY 🐂", use_container_width=True): st.session_state.choice="LONG"; st.session_state.sim_done=True; st.rerun()
            if c2.button("SELL 🐻", use_container_width=True): st.session_state.choice="SHORT"; st.session_state.sim_done=True; st.rerun()
            if c3.button("SKIP ⏭️", use_container_width=True): del st.session_state.sim_t; st.rerun()
        else:
            future = st.session_state.sim_data['Close'].iloc[idx+30]
            start = st.session_state.sim_data['Close'].iloc[idx]
            pct = ((future-start)/start)*100
            win = (st.session_state.choice=="LONG" and pct>0) or (st.session_state.choice=="SHORT" and pct<0)
            
            if win: st.success(f"WIN! It moved {pct:.2f}%"); st.balloons()
            else: st.error(f"LOSS. It moved {pct:.2f}%")
            if st.button("Next Round"): del st.session_state.sim_t; st.rerun()

    # --- SCANNER ---
    with tabs[3]:
        st.header("📡 Live Scanner")
        if st.button("Scan Tech Giants"):
            tk = ["AAPL","MSFT","GOOGL","AMZN","TSLA","NVDA"]
            res = []
            for t in tk:
                try:
                    d = yf.Ticker(t).history(period="3mo")
                    if not d.empty:
                        # חישוב RSI מהיר
                        delta = d['Close'].diff()
                        up, down = delta.copy(), delta.copy()
                        up[up < 0] = 0
                        down[down > 0] = 0
                        rs = up.ewm(span=14).mean() / down.abs().ewm(span=14).mean()
                        rsi = 100 - 100 / (1 + rs)
                        
                        r_val = rsi.iloc[-1]
                        stat = "HOT 🔥" if r_val > 70 else "COLD ❄️" if r_val < 30 else "OK"
                        res.append({"Symbol":t, "Price":f"${d['Close'].iloc[-1]:.2f}", "RSI":f"{r_val:.1f}", "Status":stat})
                except: pass
            st.dataframe(pd.DataFrame(res), use_container_width=True)

    # --- ACADEMY ---
    with tabs[4]:
        st.title("🎓 Academy")
        c1, c2 = st.columns(2)
        with c1:
            st.write("### מחשבון ריבית")
            init = st.number_input("סכום התחלה", 10000)
            rate = st.slider("תשואה %", 1, 15, 8)
            yrs = st.slider("שנים", 1, 40, 20)
            final = init * (1+rate/100)**yrs
            st.metric("תוצאה", f"₪{final:,.0f}")
        with c2:
            st.write("### מנטור AI")
            q = st.text_input("מה תרצה ללמוד?")
            if q and st.button("למד אותי"):
                st.info(get_ai_response([{'role':'user', 'content':f"Teach me: {q}"}], "Education"))


# --- לוגיקת כניסה ---
if 'logged_in' not in st.session_state: st.session_state.logged_in=False; st.session_state.username=''

if not st.session_state.logged_in:
    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        st.title("💎 APEX LOGIN")
        t1,t2 = st.tabs(["Login", "Sign Up"])
        
        with t1:
            u = st.text_input("User")
            p = st.text_input("Password", type="password")
            if st.button("Enter", use_container_width=True):
                if login_user(u, p):
                    st.session_state.logged_in = True
                    st.session_state.username = str(u)
                    st.rerun()
                else: st.error("Wrong user/pass")
        
        with t2:
            nu = st.text_input("New User")
            np = st.text_input("New Password", type="password")
            if st.button("Create Account", use_container_width=True):
                if create_user(nu, np): st.success("Created! Now Login.")
                else: st.error("User taken")

else:
    main_app(st.session_state.username)
