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

# ==========================================
# 1. הגדרות מערכת ועיצוב (System & UI)
# ==========================================
st.set_page_config(page_title="APEX Terminal", layout="wide", page_icon="💎")

def load_custom_css():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Heebo:wght@400;700&display=swap');
            
            /* עיצוב כללי - רקע כהה ויוקרתי */
            .stApp { background-color: #0E1117; color: #E6EDF3; font-family: 'Heebo', sans-serif; direction: rtl; }
            
            /* כותרות בזהב */
            h1, h2, h3 { color: #D4AF37 !important; text-align: right; font-weight: 700; text-shadow: 0px 0px 10px rgba(212, 175, 55, 0.3); }
            
            /* טקסטים ומדדים */
            p, label, .stMarkdown { text-align: right; font-size: 1.05rem; }
            .stMetric { text-align: right !important; direction: ltr; }
            [data-testid="stMetricValue"] { color: #E6EDF3 !important; font-weight: 700; }
            [data-testid="stMetricLabel"] { color: #D4AF37 !important; }
            
            /* כפתורים מיוחדים */
            .stButton button { 
                background: linear-gradient(135deg, #D4AF37 0%, #F4CF57 100%); 
                color: #000; 
                font-weight: 800; 
                border: none; 
                border-radius: 8px; 
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(212, 175, 55, 0.3);
            }
            .stButton button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(212, 175, 55, 0.5); }
            
            /* טבלאות */
            [data-testid="stDataFrame"] { border: 1px solid #30363D; border-radius: 8px; }
        </style>
    """, unsafe_allow_html=True)
load_custom_css()

# ==========================================
# 2. חיבורים חיצוניים (Database & AI)
# ==========================================
@st.cache_resource
def connect_to_db():
    """חיבור מאובטח לגוגל שיטס"""
    try:
        if "gcp_service_account" not in st.secrets: return None
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        return client.open("APEX_Database")
    except Exception as e:
        print(f"DB Connection Error: {e}")
        return None

def get_ai_response(messages, context_data):
    """מנוע הבינה המלאכותית - המוח של האפליקציה"""
    try:
        if "GOOGLE_API_KEY" not in st.secrets: return "⚠️ חסר מפתח AI בקובץ הסודות."
        
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        # שימוש במודל החדש והמהיר
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        system_prompt = f"""
        Role: You are APEX, an elite trading mentor and analyst.
        Context: {context_data}
        Language: Hebrew (עברית).
        Style: Professional, sharp, concise. No financial advice disclaimers needed inside the analysis.
        Task: Analyze the data and give clear insights.
        """
        
        chat_history = [{'role': 'user', 'parts': [system_prompt]}]
        for m in messages:
            role = 'user' if m['role']=='user' else 'model'
            chat_history.append({'role': role, 'parts': [m['content']]})
            
        return model.generate_content(chat_history).text
    except Exception as e: return f"שגיאת AI: {str(e)}"

# ==========================================
# 3. מנוע נתונים ואינדיקטורים (Data Engine)
# ==========================================
@st.cache_data(ttl=60)
def get_data(ticker, period, interval):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        return df, stock.info
    except: return pd.DataFrame(), {}

def add_indicators(df):
    """חישוב כל המדדים הטכניים החשובים"""
    if df.empty: return df
    
    # 1. RSI (מומנטום)
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 2. ממוצעים נעים (Trend)
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()
    
    # 3. רצועות בולינגר (Volatility)
    df['BB_Upper'] = df['Close'].rolling(20).mean() + (df['Close'].rolling(20).std() * 2)
    df['BB_Lower'] = df['Close'].rolling(20).mean() - (df['Close'].rolling(20).std() * 2)
    
    # 4. MACD (חדש! ביקשת איכות)
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    return df

def render_prediction(df):
    """מודל חיזוי ליניארי"""
    if len(df) < 30: return
    
    df_p = df.copy().reset_index()
    df_p['DateNum'] = df_p['Date'].apply(lambda x: x.toordinal())
    X = df_p[['DateNum']]; y = df_p['Close']
    
    model = LinearRegression().fit(X, y)
    
    future_dates = [df_p['Date'].iloc[-1] + timedelta(days=i) for i in range(1, 31)]
    future_X = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    pred = model.predict(future_X)
    
    # חישוב אחוז שינוי חזוי
    current = y.iloc[-1]
    target = pred[-1]
    change = ((target - current) / current) * 100
    color = "#00C805" if change > 0 else "#FF3333"
    
    st.markdown("### 🔮 APEX Vision AI")
    st.caption("מודל רגרסיה לזיהוי כיוון המגמה ב-30 הימים הקרובים")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("מחיר יעד (30 יום)", f"${target:.2f}", f"{change:.2f}%", delta_color="normal")
        st.write(f"**אמינות מודל:** {model.score(X, y)*100:.0f}%")
        
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='מחיר היסטורי', line=dict(color='#888')))
        fig.add_trace(go.Scatter(x=future_dates, y=pred, name='תחזית AI', line=dict(color=color, width=3, dash='dot')))
        fig.update_layout(template="plotly_dark", height=300, margin=dict(t=10,b=10,l=0,r=0), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 4. ניהול משתמשים ופורטפוליו
# ==========================================
def make_hashes(p): return hashlib.sha256(str.encode(p)).hexdigest()

def login_user(u, p):
    sh = connect_to_db()
    if not sh: return False
    try:
        df = pd.DataFrame(sh.worksheet("users").get_all_records())
        if df.empty: return False
        df['username'] = df['username'].astype(str)
        user_row = df[df['username'] == str(u).strip()]
        if user_row.empty: return False
        return str(user_row.iloc[0]['password']) == make_hashes(p)
    except: return False

def create_user(u, p):
    sh = connect_to_db()
    if not sh: return False
    try:
        ws = sh.worksheet("users")
        existing = [str(x) for x in ws.col_values(1)]
        if str(u) in existing: return False
        ws.append_row([str(u), make_hashes(p), str(datetime.now())])
        return True
    except: return False

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
        df = pd.DataFrame(sh.worksheet("trades").get_all_records())
        if df.empty: return pd.DataFrame()
        df['username'] = df['username'].astype(str)
        udf = df[df['username'] == str(u)].copy()
        if udf.empty: return pd.DataFrame()
        
        # חישובים
        udf['quantity'] = pd.to_numeric(udf['quantity'])
        udf['price'] = pd.to_numeric(udf['price'])
        
        ptf = udf.groupby('symbol').apply(
            lambda x: pd.Series({
                'Quantity': x['quantity'].sum(),
                'AvgPrice': (x['quantity'] * x['price']).sum() / x['quantity'].sum()
            })
        ).reset_index()
        return ptf[ptf['Quantity'] > 0]
    except: return pd.DataFrame()

# ==========================================
# 5. האפליקציה הראשית (Main App)
# ==========================================
def main_app(username):
    # --- סרגל צד (Sidebar) ---
    with st.sidebar:
        st.title("💎 APEX PRO")
        st.caption(f"מחובר כ: {username}")
        st.markdown("---")
        
        st.markdown("### 🧠 המנטור שלך")
        if user_q := st.chat_input("שאל על השוק, אסטרטגיות או מושגים..."):
            with st.spinner("מגבש תשובה..."):
                ans = get_ai_response([{'role':'user', 'content':user_q}], "General Mentor Chat")
                st.info(ans)
        
        st.markdown("---")
        if st.button("יציאה מהמערכת (Logout)"):
            st.session_state.logged_in = False
            st.rerun()

    # --- לשוניות תוכן (Tabs) ---
    tabs = st.tabs(["📊 חדר מסחר", "💼 הכספת (תיק)", "📡 הראדאר", "🕹️ סימולטור", "🎓 אקדמיה"])

    # --- לשונית 1: חדר מסחר (Market) ---
    with tabs[0]:
        col_search, col_info = st.columns([1, 3])
        ticker = col_search.text_input("חפש סימול (למשל TSLA)", "NVDA").upper()
        
        if ticker:
            with st.spinner(f"מוריד נתונים עבור {ticker}..."):
                df, info = get_data(ticker, "2y", "1d")
            
            if not df.empty:
                df = add_indicators(df)
                last = df.iloc[-1]
                
                # כותרת ונתונים בזמן אמת
                st.markdown(f"## {ticker} - {info.get('shortName', ticker)}")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("מחיר אחרון", f"${last['Close']:.2f}", help="מחיר סגירה אחרון")
                m2.metric("שינוי יומי", f"{df['Close'].pct_change().iloc[-1]*100:.2f}%", 
                          delta_color="normal")
                
                rsi_val = last['RSI']
                rsi_state = "🔥 יקר" if rsi_val > 70 else "❄️ זול" if rsi_val < 30 else "בינוני"
                m3.metric("RSI", f"{rsi_val:.1f}", rsi_state, delta_color="off")
                
                m4.metric("Market Cap", f"${info.get('marketCap',0)/1e9:.1f}B")

                # כפתור הניתוח המרכזי
                if st.button(f"🤖 נתח את {ticker} עם בינה מלאכותית", type="primary", use_container_width=True):
                    with st.spinner("ה-AI קורא את הגרף..."):
                        prompt = f"Analyze {ticker}. Price: {last['Close']}, RSI: {rsi_val}, BB_Upper: {last['BB_Upper']}, BB_Lower: {last['BB_Lower']}. Trend: {'Up' if last['Close']>last['SMA_50'] else 'Down'}. Summarize concisely."
                        analysis = get_ai_response([{'role':'user', 'content':prompt}], "Technical Analysis")
                        st.success(analysis)

                # הגרף הגדול
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='מחיר'))
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='rgba(255,255,255,0.2)', width=1), name='B.Band Upper'))
                fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='rgba(255,255,255,0.2)', width=1), name='B.Band Lower'))
                fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='#D4AF37', width=2), name='SMA 50'))
                fig.update_layout(template="plotly_dark", height=500, xaxis_rangeslider_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
                render_prediction(df)

    # --- לשונית 2: הכספת (Portfolio) ---
    with tabs[1]:
        st.header(f"הכספת של {username}")
        
        c1, c2 = st.columns([1, 2])
        with c1:
            with st.container(border=True):
                st.subheader("הוספת פעולה")
                with st.form("add_trade"):
                    s = st.text_input("סימול (Symbol)").upper()
                    q = st.number_input("כמות (Quantity)", 1)
                    p = st.number_input("מחיר קנייה ($)", 0.0)
                    if st.form_submit_button("רשום בספרים"):
                        if add_trade(username, s, q, p): 
                            st.toast("✅ העסקה נרשמה!")
                            time.sleep(1); st.rerun()
                        else: st.error("שגיאת רישום")

        with c2:
            df_p = get_portfolio(username)
            if not df_p.empty:
                # חישוב רווח בזמן אמת
                current_prices = []
                for sym in df_p['symbol']:
                    try: current_prices.append(yf.Ticker(sym).fast_info['last_price'])
                    except: current_prices.append(0)
                
                df_p['Current'] = current_prices
                df_p['Value'] = df_p['Quantity'] * df_p['Current']
                df_p['Cost'] = df_p['Quantity'] * df_p['AvgPrice']
                df_p['Profit ($)'] = df_p['Value'] - df_p['Cost']
                df_p['Profit (%)'] = (df_p['Profit ($)'] / df_p['Cost']) * 100
                
                total_equity = df_p['Value'].sum()
                total_profit = df_p['Profit ($)'].sum()
                
                # מדדים מסכמים
                m1, m2 = st.columns(2)
                m1.metric("שווי תיק כולל", f"${total_equity:,.2f}")
                m2.metric("רווח/הפסד פתוח", f"${total_profit:,.2f}", delta=total_profit)
                
                st.dataframe(
                    df_p[['symbol', 'Quantity', 'AvgPrice', 'Current', 'Profit ($)', 'Profit (%)']]
                    .style.format({"AvgPrice":"${:.2f}", "Current":"${:.2f}", "Profit ($)":"${:.2f}", "Profit (%)":"{:.2f}%"})
                    .background_gradient(subset=['Profit (%)'], cmap='RdYlGn'),
                    use_container_width=True
                )
            else:
                st.info("הכספת ריקה. התחל לסחור!")

    # --- לשונית 3: הראדאר (Scanner) ---
    with tabs[2]:
        st.header("📡 הראדאר: איתור הזדמנויות")
        st.markdown("סריקה חיה של מניות הטכנולוגיה הגדולות לאיתור מצבי קיצון.")
        
        if st.button("סרוק את השוק עכשיו"):
            tickers = ["AAPL","MSFT","GOOGL","AMZN","TSLA","NVDA","META","AMD","NFLX","INTC"]
            results = []
            
            bar = st.progress(0)
            for i, t in enumerate(tickers):
                try:
                    d = yf.Ticker(t).history(period="3mo")
                    if not d.empty:
                        # חישוב RSI מקוצר
                        delta = d['Close'].diff()
                        up, down = delta.copy(), delta.copy()
                        up[up<0]=0; down[down>0]=0
                        rs = up.ewm(span=14).mean() / down.abs().ewm(span=14).mean()
                        rsi = 100 - 100/(1+rs)
                        val = rsi.iloc[-1]
                        
                        status = "OK"
                        if val > 70: status = "🔥 רותח (Overbought)"
                        elif val < 30: status = "❄️ קפוא (Oversold)"
                        
                        results.append({
                            "Symbol": t, 
                            "Price": f"${d['Close'].iloc[-1]:.2f}", 
                            "RSI": f"{val:.1f}", 
                            "Signal": status
                        })
                except: pass
                bar.progress((i+1)/len(tickers))
            
            st.dataframe(pd.DataFrame(results), use_container_width=True)

    # --- לשונית 4: סימולטור (Time Machine) ---
    with tabs[3]:
        st.header("🕹️ מכונת הזמן")
        st.markdown("בחן את האינסטינקטים שלך: האם אתה יכול לזהות את המגמה?")
        
        if 'sim_data' not in st.session_state:
            st.session_state.sim_ticker = random.choice(["AAPL","TSLA","NVDA","AMZN","AMD"])
            data, _ = get_data(st.session_state.sim_ticker, "3y", "1d")
            st.session_state.sim_full = data
            st.session_state.sim_idx = random.randint(200, len(data)-60)
            st.session_state.sim_done = False
        
        idx = st.session_state.sim_idx
        # הצגת העבר עד לנקודה שנבחרה
        vis_df = st.session_state.sim_full.iloc[idx-100:idx]
        
        fig = go.Figure(data=[go.Candlestick(x=vis_df.index, open=vis_df['Open'], close=vis_df['Close'])])
        fig.update_layout(title="מניה מסתורית X", template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
        
        c1, c2, c3 = st.columns(3)
        if not st.session_state.sim_done:
            if c1.button("קנה (LONG) 🐂", use_container_width=True): 
                st.session_state.choice = "LONG"
                st.session_state.sim_done = True
                st.rerun()
            if c2.button("מכור (SHORT) 🐻", use_container_width=True): 
                st.session_state.choice = "SHORT"
                st.session_state.sim_done = True
                st.rerun()
            if c3.button("דלג ⏭️", use_container_width=True):
                del st.session_state.sim_data
                st.rerun()
        else:
            # בדיקת תוצאות
            future_price = st.session_state.sim_full['Close'].iloc[idx+20] # בדיקה עוד 20 יום
            start_price = st.session_state.sim_full['Close'].iloc[idx]
            pct = ((future_price - start_price) / start_price) * 100
            
            win = (st.session_state.choice=="LONG" and pct>0) or (st.session_state.choice=="SHORT" and pct<0)
            
            st.markdown(f"### התוצאה: המניה עשתה **{pct:.2f}%** ב-20 הימים הבאים.")
            st.markdown(f"המניה הייתה: **{st.session_state.sim_ticker}** בתאריך {vis_df.index[-1].strftime('%Y-%m-%d')}")
            
            if win: 
                st.success("🎉 ניצחון! קראת את המפה נכון.")
                st.balloons()
            else: 
                st.error("💀 הפסד. לא נורא, לומדים.")
            
            if st.button("סיבוב נוסף 🔄"):
                del st.session_state.sim_data
                st.rerun()

    # --- לשונית 5: אקדמיה (Academy) - החומר המלא! ---
    with tabs[4]:
        st.header("🎓 האוניברסיטה של APEX")
        
        sub_tabs = st.tabs(["פסיכולוגיה", "ניתוח טכני", "מחשבון ריבית"])
        
        with sub_tabs[0]:
            st.subheader("האויב שבפנים: פסיכולוגיה של מסחר")
            
            with st.expander("😱 FOMO (הפחד להחמיץ)", expanded=True):
                st.write("""
                **מה זה:** ההרגשה שכולם מתעשרים חוץ ממך. זה קורה כשמניה עולה ב-20% ביום, ואתה קונה בשיא רק כדי להיות "חלק מהחגיגה".
                **הסכנה:** בדרך כלל אתה קונה בדיוק כשהמקצוענים מוכרים.
                **הפתרון:** אם המניה ברחה, שחרר אותה. תמיד תבוא עוד רכבת. אל תרדוף אחרי המחיר.
                """)
                
            with st.expander("😡 מסחר נקמה (Revenge Trading)"):
                st.write("""
                **מה זה:** הפסדת כסף בעסקה? האינסטינקט הוא להיכנס מיד לעסקה גדולה יותר כדי "להחזיר את ההפסד" מהר.
                **התוצאה:** בדרך כלל זה נגמר במחיקת התיק כולו. אתה פועל מכעס, לא מהיגיון.
                **הפתרון:** הפסדת? סגור את המחשב. לך לשתות מים, עשה ספורט. תחזור מחר כשאתה רגוע.
                """)
                
            with st.expander("🛡️ ניהול סיכונים (חוק ה-1%)"):
                st.success("""
                זה החוק החשוב ביותר: **לעולם אל תסכן יותר מ-1% מהתיק שלך בעסקה אחת.**
                לדוגמה: אם יש לך $10,000, ההפסד המקסימלי בעסקה (אם הסטופ נתפס) לא יעלה על $100.
                זה מבטיח שגם רצף של 10 הפסדים לא יחסל אותך.
                """)

        with sub_tabs[1]:
            st.subheader("ארגז הכלים הטכני")
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### RSI (מדד העוצמה)")
                # --- התיקון הקריטי כאן למטה: שימוש במרכאות משולשות ---
                st.info("""כמו מד סל"ד באוטו. אם הוא מעל 70, המנוע 'צועק' (קניית יתר) ועשוי לעצור. אם מתחת ל-30, הוא 'נח' (מכירת יתר) ועשוי לזנק.""")
            with c2:
                st.markdown("#### רצועות בולינגר")
                st.info("גבולות הגזרה של המחיר. כשהמחיר פורץ את הרצועה העליונה הוא יקר סטטיסטית, וכשהוא פורץ את התחתונה הוא זול.")
                
            st.markdown("#### תבניות נרות יפניים")
            st.write("""
            * **פטיש (Hammer):** נר עם זנב ארוך למטה וגוף קטן למעלה. סימן שהקונים חוזרים (היפוך למעלה).
            * **כוכב נופל (Shooting Star):** זנב ארוך למעלה וגוף קטן למטה. סימן שהמוכרים משתלטים (היפוך למטה).
            """)

        with sub_tabs[2]:
            st.subheader("🧮 מחשבון הפלא (ריבית דריבית)")
            st.write("תראה איך זמן מנצח כסף.")
            
            col_calc1, col_calc2 = st.columns(2)
            with col_calc1:
                start_money = st.number_input("סכום התחלתי (₪)", 10000, 1000000, 50000)
                monthly_add = st.number_input("הפקדה חודשית (₪)", 0, 50000, 2000)
            with col_calc2:
                years = st.slider("שנים", 1, 40, 20)
                rate = st.slider("תשואה שנתית ממוצעת (%)", 1, 15, 8)
            
            # חישוב מדויק
            future_val = start_money * ((1 + rate/100) ** years)
            for i in range(1, years * 12 + 1):
                # חישוב ערך עתידי של כל הפקדה חודשית
                time_remaining = (years * 12 - i) / 12
                future_val += monthly_add * ((1 + rate/100) ** time_remaining)
            
            st.metric("שווי עתידי מוערך", f"₪{future_val:,.0f}")
            st.progress(min(100, int(rate*5)))


# ==========================================
# 6. מסך כניסה והרשמה (Login Flow)
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

if not st.session_state.logged_in:
    # מרכוז מסך הכניסה
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("<h1 style='text-align: center;'>💎 APEX</h1>", unsafe_allow_html=True)
        
        login_tab, signup_tab = st.tabs(["כניסה למערכת", "הרשמה חדשה"])
        
        with login_tab:
            with st.form("login"):
                u = st.text_input("שם משתמש")
                p = st.text_input("סיסמה", type="password")
                if st.form_submit_button("התחבר", use_container_width=True):
                    if login_user(u, p):
                        st.session_state.logged_in = True
                        st.session_state.username = str(u).strip()
                        st.rerun()
                    else:
                        st.error("שם משתמש או סיסמה שגויים")
        
        with signup_tab:
            with st.form("signup"):
                new_u = st.text_input("בחר שם משתמש")
                new_p = st.text_input("בחר סיסמה", type="password")
                if st.form_submit_button("צור חשבון", use_container_width=True):
                    if len(new_p) < 4:
                        st.warning("הסיסמה קצרה מדי")
                    elif create_user(new_u, new_p):
                        st.success("החשבון נוצר! כעת עבור ללשונית כניסה.")
                    else:
                        st.error("שם המשתמש תפוס")
else:
    # הפעלת האפליקציה הראשית אם המשתמש מחובר
    main_app(st.admin)

