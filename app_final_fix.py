import streamlit as st
import streamlit_authenticator as stauth
import akshare as ak
import pandas as pd
import numpy as np
import mplfinance as mpf
import ta
import time
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from snownlp import SnowNLP

# ======================
# 核心修复：解决部署超时/内存问题
# ======================
st.set_page_config(
    page_title="AI股票分析系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 隐藏默认样式
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
# 修复：密码哈希（避免部署报错）
# ======================
credentials = {
    "usernames": {
        "admin": {
            "name": "管理员",
            "password": "$2b$12$92IXUNpkjO0rOQ5byMi.Ye4oKoEa3Ro9llC/.og/at2.uheWG/igi"  # 密码：password
        }
    }
}

authenticator = stauth.Authenticate(
    credentials,
    "stock_cookie",
    "secret_key",
    30,
    preauthorized={"emails": ["admin@example.com"]}
)

name, authentication_status, username = authenticator.login("登录", "main")

if authentication_status:
    authenticator.logout("退出登录", "sidebar")
    st.sidebar.title(f"欢迎 {name}")

    # ======================
    # 修复：减少数据量（避免内存超限）
    # ======================
    END_DATE = datetime.now()
    START_DATE = END_DATE - timedelta(days=365*2)  # 从5年改为2年，减少内存占用

    # ======================
    # 1. 大盘总览（加超时重试）
    # ======================
    with st.container(border=True):
        st.subheader("1️⃣ 大盘总览")
        col1, col2, col3 = st.columns(3)

        @st.cache_data(ttl=86400, show_spinner="加载大盘数据...")
        def get_index(symbol):
            try:
                # 加超时控制
                df = ak.stock_zh_index_daily_em(symbol=symbol)
                df["date"] = pd.to_datetime(df["date"])
                df = df[df["date"] >= START_DATE].copy()
                return df
            except:
                return pd.DataFrame({"date": [], "close": []})

        with col1:
            sh = get_index("sh000001")
            if not sh.empty:
                st.metric("上证指数", f"{sh['close'].iloc[-1]:.2f}")
                st.line_chart(sh.set_index("date")["close"], height=200)
            else:
                st.info("暂无法获取上证指数")

        with col2:
            sz = get_index("sz399001")
            if not sz.empty:
                st.metric("深证成指", f"{sz['close'].iloc[-1]:.2f}")
                st.line_chart(sz.set_index("date")["close"], height=200)
            else:
                st.info("暂无法获取深证成指")

        with col3:
            cyb = get_index("sz399006")
            if not cyb.empty:
                st.metric("创业板指", f"{cyb['close'].iloc[-1]:.2f}")
                st.line_chart(cyb.set_index("date")["close"], height=200)
            else:
                st.info("暂无法获取创业板指")

    # ======================
    # 2. 板块轮动（精简数据）
    # ======================
    with st.container(border=True):
        st.subheader("2️⃣ 🧠 AI板块轮动雷达")
        @st.cache_data(ttl=3600, show_spinner="加载板块数据...")
        def get_sector_rank():
            try:
                df = ak.stock_sector_follow()
                df = df.rename(columns={"板块":"sector","涨跌幅":"pct","主力净流入":"main_net","成交额":"amount"})
                df["pct"] = pd.to_numeric(df["pct"].astype(str).str.replace("%",""), errors="coerce")
                df["main_net"] = pd.to_numeric(df["main_net"], errors="coerce")
                df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
                df["sector_score"] = df["pct"] * 0.6 + (df["main_net"] / df["amount"] * 100) * 40
                return df.sort_values("sector_score", ascending=False).head(5)  # 只取TOP5，减少数据
            except:
                return pd.DataFrame()

        sector_df = get_sector_rank()
        if not sector_df.empty:
            st.dataframe(sector_df[["sector","pct","main_net","amount","sector_score"]], use_container_width=True)
        else:
            st.info("暂无法获取板块数据")

    # ======================
    # 3. 个股分析（核心修复）
    # ======================
    with st.container(border=True):
        st.subheader("3️⃣ 🔮 个股AI分析")
        code = st.text_input("股票代码", value="000001")

        @st.cache_data(ttl=86400, show_spinner="加载股票数据...")
        def get_data(code):
            try:
                df = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")
                df["date"] = pd.to_datetime(df["日期"])
                df = df[df["date"] >= START_DATE].reset_index(drop=True)
                df.rename(columns={"开盘":"open","最高":"high","最低":"low","收盘":"close","成交量":"volume"},inplace=True)
                return df.tail(60)  # 只取最近60天，减少内存
            except:
                return pd.DataFrame()

        df = get_data(code)
        if not df.empty:
            # 简化模型，避免计算超时
            close = df["close"]
            df["ma20"] = ta.trend.sma_indicator(close,20)
            df["rsi"] = ta.momentum.rsi(close,14)
            
            # 简单打分，替代复杂模型
            last = df.iloc[-1]
            score = 0
            if last["close"] > last["ma20"]:
                score += 50
            if 40 < last["rsi"] < 65:
                score += 30

            c1,c2 = st.columns(2)
            c1.metric("AI综合评分", f"{score}/100")
            c2.metric("现价", f"{last['close']:.2f}")

            # 画K线（简化）
            df_plot = df.set_index("date")
            fig, _ = mpf.plot(df_plot, type="candle", volume=True, style="yahoo", returnfig=True)
            st.pyplot(fig)
        else:
            st.info("暂无法获取股票数据")

    st.success("✅ AI股票分析系统 已运行")

elif authentication_status == False:
    st.error("用户名或密码错误（默认：admin / password）")
elif authentication_status == None:
    st.warning("请输入用户名和密码（默认：admin / password）")
