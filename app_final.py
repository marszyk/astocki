import streamlit as st
import streamlit_authenticator as stauth
import akshare as ak
import pandas as pd
import numpy as np
import mplfinance as mpf
import ta
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from snownlp import SnowNLP

# ======================
# 页面配置（美化）
# ======================
st.set_page_config(
    page_title="AI股票分析系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 隐藏Streamlit默认样式
hide_style = """
<style>
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
.block-container {padding:1rem 1rem !important;}
</style>
"""
st.markdown(hide_style, unsafe_allow_html=True)

# ======================
# 密码登录配置
# ======================
credentials = {
    "usernames": {
        "admin": {
            "name": "管理员",
            "password": "$2b$12$PQ8XnQdVbLZcQ7iQVJxQOOGx.gsqp/hUeF1d9kE4m5X0nVhMhw5ZG"
        }
    }
}

authenticator = stauth.Authenticate(
    credentials,
    "stock_cookie",
    "secret_key",
    30
)

name, authentication_status, username = authenticator.login("登录", "main")

if authentication_status:
    authenticator.logout("退出登录", "sidebar")
    st.sidebar.title(f"欢迎 {name}")

    # ======================
    # 主界面开始
    # ======================
    st.title("📊 AI股票分析系统 — 专业版")
    st.markdown("### 市场 → 板块 → 个股 → 预测 → 舆情 全链路分析")

    END_DATE = datetime.now()
    START_DATE = END_DATE - timedelta(days=365*5)

    # ======================
    # 1. 大盘总览
    # ======================
    with st.container(border=True):
        st.subheader("1️⃣ 大盘总览")
        col1, col2, col3 = st.columns(3)

        def get_index(symbol):
            df = ak.stock_zh_index_daily_em(symbol=symbol)
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["date"] >= START_DATE].copy()
            return df

        with col1:
            sh = get_index("sh000001")
            st.metric("上证指数", f"{sh['close'].iloc[-1]:.2f}")
            st.line_chart(sh.set_index("date")["close"], height=200)
        with col2:
            sz = get_index("sz399001")
            st.metric("深证成指", f"{sz['close'].iloc[-1]:.2f}")
            st.line_chart(sz.set_index("date")["close"], height=200)
        with col3:
            cyb = get_index("sz399006")
            st.metric("创业板指", f"{cyb['close'].iloc[-1]:.2f}")
            st.line_chart(cyb.set_index("date")["close"], height=200)

    # ======================
    # 2. 板块轮动AI
    # ======================
    with st.container(border=True):
        st.subheader("2️⃣ 🧠 AI板块轮动雷达")
        @st.cache_data(ttl=3600)
        def get_sector_rank():
            df = ak.stock_sector_follow()
            df = df.rename(columns={"板块":"sector","涨跌幅":"pct","主力净流入":"main_net","成交额":"amount"})
            df["pct"] = pd.to_numeric(df["pct"].astype(str).str.replace("%",""), errors="coerce")
            df["main_net"] = pd.to_numeric(df["main_net"], errors="coerce")
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
            df["sector_score"] = df["pct"] * 0.6 + (df["main_net"] / df["amount"] * 100) * 40
            return df.sort_values("sector_score", ascending=False)

        sector_df = get_sector_rank()
        st.dataframe(sector_df.head(10)[["sector","pct","main_net","amount","sector_score"]], use_container_width=True)

    # ======================
    # 3. AI选股排名
    # ======================
    with st.container(border=True):
        st.subheader("3️⃣ 🎯 AI全市场选股排名")
        @st.cache_data(ttl=3600)
        def get_ai_stock_rank(n=50):
            df = ak.stock_rank_cybs_ths().head(n)
            df["代码"] = df["代码"].astype(str).str.zfill(6)
            scores, reasons = [], []
            for code in df["代码"]:
                try:
                    k = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq").tail(120)
                    close = k["收盘"]
                    ma20 = ta.trend.sma_indicator(close,20).iloc[-1]
                    ma60 = ta.trend.sma_indicator(close,60).iloc[-1]
                    rsi = ta.momentum.rsi(close,14).iloc[-1]
                    s = 0
                    rs = []
                    if close.iloc[-1]>ma20 and ma20>ma60:
                        s+=40
                        rs.append("多头")
                    if 40<rsi<65:
                        s+=20
                        rs.append("健康")
                    scores.append(min(100,max(0,s)))
                    reasons.append("｜".join(rs))
                except:
                    scores.append(0)
                    reasons.append("数据异常")
            df["ai_score"]=scores
            df["ai_tag"]=reasons
            return df.sort_values("ai_score", ascending=False)

        rank_df = get_ai_stock_rank(50)
        st.dataframe(rank_df[["代码","名称","涨跌幅","ai_score","ai_tag"]], use_container_width=True)

    # ======================
    # 4. 个股预测+策略
    # ======================
    with st.container(border=True):
        st.subheader("4️⃣ 🔮 个股AI预测 & 策略")
        code = st.text_input("股票代码", value="000001")

        @st.cache_data(ttl=3600)
        def get_data(code):
            df = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")
            df["date"] = pd.to_datetime(df["日期"])
            df = df[df["date"] >= START_DATE].reset_index(drop=True)
            df.rename(columns={"开盘":"open","最高":"high","最低":"low","收盘":"close","成交量":"volume"},inplace=True)
            close = df["close"]
            df["ma5"] = ta.trend.sma_indicator(close,5)
            df["ma20"] = ta.trend.sma_indicator(close,20)
            df["ma60"] = ta.trend.sma_indicator(close,60)
            df["rsi"] = ta.momentum.rsi(close,14)
            df["ret"] = df["close"].pct_change()
            df["label"] = (df["ret"].shift(-1) > 0).astype(int)
            return df.dropna()

        df = get_data(code)
        features = ["ma5","ma20","ma60","rsi","volume"]
        X, y = df[features], df["label"]
        split = int(len(df)*0.8)
        model = LinearRegression()
        model.fit(X.iloc[:split], y.iloc[:split])
        prob = max(0, min(1, model.predict([df.iloc[-1][features]])[0]))

        c1,c2,c3 = st.columns(3)
        c1.metric("AI上涨概率", f"{prob:.1%}")
        c2.metric("回测准确率", f"{accuracy_score(y.iloc[split:], model.predict(X.iloc[split:])>0.5):.1%}")
        c3.metric("现价", f"{df['close'].iloc[-1]:.2f}")

        recent = df.tail(60)
        st.info(f"支撑：{recent['low'].min():.2f} ｜ 压力：{recent['high'].max():.2f}")

        df_plot = df.set_index("date").tail(120)
        fig, _ = mpf.plot(df_plot, type="candle", volume=True, style="yahoo", returnfig=True)
        st.pyplot(fig)

    # ======================
    # 5. 舆情NLP
    # ======================
    with st.container(border=True):
        st.subheader("5️⃣ 📰 AI舆情分析")
        @st.cache_data(ttl=3600)
        def get_news(code):
            try:
                df = ak.stock_news_em(symbol=code).head(15)
                df["sentiment"] = df["内容"].apply(lambda x: SnowNLP(str(x)).sentiments)
                return df
            except:
                return pd.DataFrame()

        news = get_news(code)
        if not news.empty:
            avg_sen = news["sentiment"].mean()
            st.markdown(f"**舆情综合评分：{avg_sen:.1%}**")
            if avg_sen > 0.6:
                st.success("✅ 舆情偏正面")
            elif avg_sen < 0.4:
                st.error("🔻 舆情偏负面")
            else:
                st.info("📊 舆情中性")
            st.dataframe(news[["发布时间","标题","sentiment"]], use_container_width=True)

    st.success("✅ AI股票分析系统 专业版 已完整运行")

elif authentication_status == False:
    st.error("用户名或密码错误")
elif authentication_status == None:
    st.warning("请输入用户名和密码")
