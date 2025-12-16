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
            [data-testid="stPopover"] > button {
                background-color: #161B22; color: #D4AF37; border: 2px solid #D4AF37;
                border-radius: 50%; width: 60px; height: 60px; font-size: 28px;
                box-shadow: 0 0 15px rgba(212, 175, 55, 0.2); position: fixed; bottom: 30px; right: 30px; z-index: 9999;
            }
            [data-testid="stMetricValue"] { color: #E6EDF3; font-weight: 700; }
            [data-testid="stExpander"] { border: 1px solid #30363D; border-radius: 8px; background-color: #161B22; }
            [data-testid="stDataFrame"] { border: 1px solid #30363D; border-radius: 8px; }
        </style>
    """, unsafe_allow_html=True)
load_custom_css()

# --- 3. חיבורים (DB & AI) ---
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
        if "GOOGLE_API_KEY" not in st.secrets: return "⚠️ Error: AI Key Missing."
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"Role: Trading Mentor. Context: {context_data}. Question: {messages[-1]['content']}. Keep it short & professional."
        chat = [{'role': 'user', 'parts': [prompt]}]
        for m in messages: chat.append({'role': 'user' if m['role']=='user' else 'model', 'parts': [m['content']]})
        return model.generate_content(chat).text
    except Exception as e: return f"AI Error: {e}"

# --- 4. נתונים וניתוח ---
@st.cache_data(ttl=60)
def get_data(ticker, period, interval):
    try:
        stock = yf.Ticker(ticker)
        return stock.history(period=period, interval=interval), stock.info
    except: return pd.DataFrame(), {}

def add_indicators(df):
    if df.empty: return df
    df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().where(df['Close'].diff() > 0, 0).rolling(14).mean() / -df['Close'].diff().where(df['Close'].diff() < 0, 0).rolling(14).mean())))
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    low14, high14 = df['Low'].rolling(14).min(), df['High'].rolling(14).max()
    df['K_Percent'] = 100 * ((df['Close'] - low14) / (high14 - low14))
    return df

def render_prediction(df, ticker):
    st.markdown("### 🔮 AI Price Projector (30 Days)")
    st.info("מודל רגרסיה ליניארית המזהה את המגמה. לא המלצה למסחר.")
    df_p = df.copy().reset_index()
    df_p['DateNum'] = df_p['Date'].apply(lambda x: x.toordinal())
    X, y = df_p[['DateNum']], df_p['Close']
    model = LinearRegression().fit(X, y)
    future_dates = [df_p['Date'].iloc[-1] + timedelta(days=i) for i in range(1, 31)]
    future_X = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    pred = model.predict(future_X)
    chg = ((pred[-1] - y.iloc[-1]) / y.iloc[-1]) * 100
    
    c1, c2 = st.columns([1, 3])
    c1.metric("צפי ל-30 יום", f"${pred[-1]:.2f}", f"{chg:.2f}%")
    c1.metric("אמינות מגמה", f"{model.score(X, y)*100:.1f}%")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='History', line=dict(color='#00C805')))
    fig.add_trace(go.Scatter(x=future_dates, y=pred, name='AI Forecast', line=dict(color='#D4AF37', dash='dot')))
    fig.update_layout(template="plotly_dark", height=350, margin=dict(t=10,b=10,l=0,r=0))
    c2.plotly_chart(fig, use_container_width=True)

# --- 5. ניהול משתמשים ---
def make_hashes(p): return hashlib.sha256(str.encode(p)).hexdigest()
def login_user(u, p):
    try:
        sh = connect_to_db()
        if not sh: return False
        df = pd.DataFrame(sh.worksheet("users").get_all_records())
        return not df.empty and u in df['username'].values and make_hashes(p) == df[df['username']==u]['password'].values[0]
    except: return False
def create_user(u, p):
    try:
        sh = connect_to_db()
        if not sh: return False
        ws = sh.worksheet("users")
        if u in ws.col_values(1): return False
        ws.append_row([u, make_hashes(p), str(datetime.now())]); return True
    except: return False
def add_trade(u, s, q, p):
    try: connect_to_db().worksheet("trades").append_row([u, s, q, p, str(datetime.now())]); return True
    except: return False
def get_portfolio(u):
    try:
        df = pd.DataFrame(connect_to_db().worksheet("trades").get_all_records())
        if df.empty: return df
        udf = df[df['username'] == u]
        if udf.empty: return pd.DataFrame()
        return udf.groupby('symbol').apply(lambda x: pd.Series({'Quantity': x['quantity'].sum(), 'AvgPrice': (x['quantity']*x['price']).sum()/x['quantity'].sum()})).reset_index()
    except: return pd.DataFrame()

# --- 6. האפליקציה הראשית ---
def main_app(username):
    st.sidebar.markdown("## 💎 APEX PRO")
    st.sidebar.caption(f"Operator: {username} | Status: Online")
    if st.sidebar.button("LOGOUT"): st.session_state.logged_in=False; st.rerun()

    # AI Chat
    with st.sidebar:
        with st.popover("💬 AI Assistant", use_container_width=True):
            if "msgs" not in st.session_state: st.session_state.msgs = []
            for m in st.session_state.msgs: st.markdown(f"**{'You' if m['role']=='user' else 'APEX'}:** {m['content']}")
            if p := st.chat_input("Ask me..."):
                st.session_state.msgs.append({"role":"user", "content":p})
                with st.spinner("..."): r = get_ai_response(st.session_state.msgs, st.session_state.get('ctx', 'General'))
                st.session_state.msgs.append({"role":"assistant", "content":r}); st.rerun()

    tabs = st.tabs(["📊 Market", "💼 Portfolio", "🕹️ Simulator", "📡 Scanner", "🎓 Academy"])

    with tabs[0]: # Market
        c1, c2 = st.columns([1,3])
        if t := c1.text_input("Symbol", "NVDA").upper():
            with st.spinner("Loading..."):
                df, info = get_data(t, "2y", "1d")
                if not df.empty:
                    df = add_indicators(df)
                    cur = df.iloc[-1]
                    st.session_state.ctx = f"{t}: ${cur['Close']:.2f}, RSI:{cur['RSI']:.1f}"
                    with st.container(border=True):
                        cols = st.columns(4)
                        cols[0].metric("Price", f"${cur['Close']:.2f}")
                        cols[1].metric("RSI", f"{cur['RSI']:.1f}", help=">70 High, <30 Low")
                        cols[2].metric("ATR", f"{cur['ATR']:.2f}")
                        cols[3].metric("PE Ratio", f"{info.get('trailingPE',0):.1f}")
                    
                    fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='#D4AF37'), name='SMA 50'))
                    fig.update_layout(title=f"{t} Analysis", template="plotly_dark", height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    st.divider()
                    render_prediction(df, t)
                    st.divider()
                    st.markdown("### 🏢 Fundamentals")
                    c1,c2,c3 = st.columns(3)
                    c1.metric("Market Cap", f"${info.get('marketCap',0)/1e9:.1f}B")
                    c2.metric("Target", f"${info.get('targetMeanPrice', 0)}")
                    c3.metric("Margin", f"{info.get('profitMargins',0)*100:.1f}%")
                    st.caption(info.get('longBusinessSummary', ''))
                else: st.error("Not Found")

    with tabs[1]: # Portfolio
        st.header(f"Cloud Vault: {username}")
        with st.expander("➕ Add Trade"):
            with st.form("trd"):
                c1,c2,c3 = st.columns(3)
                s = c1.text_input("Sym").upper(); q = c2.number_input("Qty", step=1); p = c3.number_input("Price", min_value=0.1)
                if st.form_submit_button("Sync"): 
                    if add_trade(username, s, q, p): st.success("Saved"); st.rerun()
        df_p = get_portfolio(username)
        if not df_p.empty: st.dataframe(df_p, use_container_width=True)

    with tabs[2]: # Simulator
        st.header("🕹️ Time Machine")
        if 'sim_t' not in st.session_state:
            st.session_state.sim_t = random.choice(["AAPL","TSLA","NVDA","AMZN"])
            d, _ = get_data(st.session_state.sim_t, "2y", "1d")
            st.session_state.sim_d = add_indicators(d)
            st.session_state.sim_c = random.randint(100, len(d)-30)
            st.session_state.sim_rev = False
        
        d, c = st.session_state.sim_d, st.session_state.sim_c
        vis = d.iloc[:c]
        fig = go.Figure(data=[go.Candlestick(x=vis.index, open=vis['Open'], close=vis['Close'])])
        fig.update_layout(title=f"Mystery: {st.session_state.sim_t}", template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        if not st.session_state.sim_rev:
            c1,c2,c3 = st.columns(3)
            if c1.button("BUY"): st.session_state.ch="LONG"; st.session_state.sim_rev=True; st.rerun()
            if c2.button("SELL"): st.session_state.ch="SHORT"; st.session_state.sim_rev=True; st.rerun()
            if c3.button("SKIP"): del st.session_state['sim_t']; st.rerun()
        else:
            pct = ((d.iloc[c+20]['Close']-d.iloc[c]['Close'])/d.iloc[c]['Close'])*100
            win = (st.session_state.ch=="LONG" and pct>0) or (st.session_state.ch=="SHORT" and pct<0)
            st.success(f"WIN! {pct:.2f}%") if win else st.error(f"LOSS. {pct:.2f}%")
            if st.button("Again"): del st.session_state['sim_t']; st.rerun()

    with tabs[3]: # Scanner
        st.header("📡 Radar")
        if st.button("Scan Tech"):
            tk = ["AAPL","MSFT","GOOGL","AMZN","TSLA","NVDA"]
            res = []
            dt = yf.download(tk, period="3mo", progress=False)['Close']
            for t in tk:
                s = dt[t].dropna()
                rsi = 100-(100/(1+(s.diff().where(s.diff()>0,0).rolling(14).mean()/-s.diff().where(s.diff()<0,0).rolling(14).mean()).iloc[-1]))
                res.append({"T":t, "RSI":f"{rsi:.1f}", "Stat": "HOT" if rsi>70 else "COLD" if rsi<30 else "OK"})
            st.dataframe(pd.DataFrame(res), use_container_width=True)

    with tabs[4]: # Academy
        st.title("🎓 APEX University")
        at = st.tabs(["📚 ספר לימוד", "🏛️ אוניברסיטה", "🧮 מחשבון", "🌐 מקורות"])
        with at[0]: st.info("פרקים: יסודות, טכני, פסיכולוגיה.")
        with at[1]: st.info("פקולטות: מאקרו, נגזרים, ניהול סיכונים.")
        with at[2]: 
            c1,c2 = st.columns(2)
            init = c1.number_input("התחלה", 10000); mon = c1.number_input("חודשי", 1500)
            rate = c2.slider("תשואה %", 2, 15, 8); yrs = c2.slider("שנים", 5, 40, 20)
            final = init * (1+rate/100)**yrs
            st.metric("צפי סופי", f"₪{final:,.0f}")
        with at[3]: st.markdown("[Bloomberg](https://bloomberg.com) | [TradingView](https://tradingview.com)")

# --- Login System ---
if 'logged_in' not in st.session_state: st.session_state.logged_in=False; st.session_state.username=''
if not st.session_state.logged_in:
    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        st.title("💎 APEX LOGIN")
        t1,t2 = st.tabs(["Login","Sign Up"])
        with t1:
            u=st.text_input("User"); p=st.text_input("Pass", type="password")
            if st.button("Enter"):
                if login_user(u,p): st.session_state.logged_in=True; st.session_state.username=u; st.rerun()
                else: st.error("Denied")
        with t2:
            nu=st.text_input("New User"); np=st.text_input("New Pass", type="password")
            if st.button("Create"):
                if create_user(nu,np): st.success("Created")
                else: st.error("Taken")
else: main_app(st.session_state.username)