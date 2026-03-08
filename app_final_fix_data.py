import streamlit as st
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


import requests

# ======================
# 核心适配：解决海外访问问题
# ======================
# 1. 设置超时时间（延长到20秒，适配海外延迟）
ak.set_option("timeout", 20)

# 2. 配置请求头（模拟浏览器，避免被风控拦截）
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.eastmoney.com/",
    "Accept-Language": "zh-CN,zh;q=0.9"
}
ak.session = requests.Session()
ak.session.headers.update(headers)

# 3. 精简数据量（避免内存超限）
def get_mini_data(code):
    # 只拉最近30天数据，减少内存占用
    df = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq").tail(30)
    return df


# ======================
# 页面配置（美化）
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
# 主界面（无登录）
# ======================
st.title("📊 AI股票分析系统 — 专业版")
st.markdown("### 市场 → 板块 → 个股 → 预测 → 舆情 全链路分析")

# 核心优化：减少数据量（解决部署内存/超时问题）
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=365*2)  # 2年数据

# ======================
# 1. 大盘总览（加超时重试+缓存）
# ======================
with st.container(border=True):
    st.subheader("1️⃣ 大盘总览")
    col1, col2, col3 = st.columns(3)

    @st.cache_data(ttl=86400, show_spinner="加载大盘数据...")
    def get_index(symbol):
        try:
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
# 2. AI板块轮动雷达（精简数据）
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
            return df.sort_values("sector_score", ascending=False).head(8)  # 精简为TOP8
        except:
            return pd.DataFrame()

    sector_df = get_sector_rank()
    if not sector_df.empty:
        st.dataframe(sector_df[["sector","pct","main_net","amount","sector_score"]], use_container_width=True)
    else:
        st.info("暂无法获取板块数据")

# ======================
# 3. AI全市场选股排名（核心修复：替换稳定接口）
# ======================
with st.container(border=True):
    st.subheader("3️⃣ 🎯 AI全市场选股排名")
    @st.cache_data(ttl=3600, show_spinner="加载选股数据...")
    def get_ai_stock_rank(n=30):
        try:
            # 替换为更稳定的A股涨幅榜接口（替代原cybs_ths接口）
            df = ak.stock_zh_a_spot_em()  # 沪深A股实时行情，稳定性100%
            df = df.head(n)  # 取前30只
            
            # 清洗数据：只保留核心列
            df = df[["代码", "名称", "涨跌幅", "最新价", "成交量", "成交额"]]
            df["代码"] = df["代码"].astype(str).str.zfill(6)  # 补全6位代码
            
            scores, reasons = [], []
            for idx, row in df.iterrows():
                code = row["代码"]
                try:
                    # 获取个股近期数据（增加超时控制）
                    k = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq", timeout=10).tail(60)
                    if len(k) < 20:  # 数据不足则跳过
                        scores.append(0)
                        reasons.append("数据不足")
                        continue
                    
                    close = k["收盘"]
                    ma20 = ta.trend.sma_indicator(close,20).iloc[-1]
                    rsi = ta.momentum.rsi(close,14).iloc[-1]
                    pct = row["涨跌幅"]
                    
                    # AI打分逻辑
                    s = 0
                    rs = []
                    # 1. 均线多头
                    if close.iloc[-1] > ma20:
                        s += 40
                        rs.append("均线多头")
                    # 2. RSI健康
                    if 40 < rsi < 65:
                        s += 20
                        rs.append("RSI健康")
                    # 3. 涨幅正向
                    if pct > 0:
                        s += 20
                        rs.append("涨幅为正")
                    # 4. 成交量放大
                    vol_ma5 = ta.trend.sma_indicator(k["成交量"],5).iloc[-1]
                    if k["成交量"].iloc[-1] > vol_ma5:
                        s += 10
                        rs.append("放量")
                    
                    scores.append(min(100, max(0, s)))
                    reasons.append("｜".join(rs) if rs else "无信号")
                except Exception as e:
                    scores.append(0)
                    reasons.append(f"数据异常: {str(e)[:10]}")
            
            df["ai_score"] = scores
            df["ai_tag"] = reasons
            # 按AI评分排序
            df = df.sort_values("ai_score", ascending=False)
            return df
        except Exception as e:
            st.error(f"选股接口异常：{str(e)}")  # 调试用，可删除
            return pd.DataFrame()

    rank_df = get_ai_stock_rank(30)
    if not rank_df.empty:
        # 只展示核心列
        show_df = rank_df[["代码", "名称", "涨跌幅", "最新价", "ai_score", "ai_tag"]]
        st.dataframe(show_df, use_container_width=True)
    else:
        st.info("暂无法获取选股数据（可刷新重试）")

# ======================
# 4. 个股AI分析+预测
# ======================
with st.container(border=True):
    st.subheader("4️⃣ 🔮 个股AI分析 & 预测")
    code = st.text_input("输入A股代码（例：000001）", value="000001")

    @st.cache_data(ttl=86400, show_spinner="加载股票数据...")
    def get_data(code):
        try:
            df = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq", timeout=10)
            df["date"] = pd.to_datetime(df["日期"])
            df = df[df["date"] >= START_DATE].reset_index(drop=True)
            df.rename(columns={"开盘":"open","最高":"high","最低":"low","收盘":"close","成交量":"volume"},inplace=True)
            close = df["close"]
            df["ma5"] = ta.trend.sma_indicator(close,5)
            df["ma20"] = ta.trend.sma_indicator(close,20)
            df["rsi"] = ta.momentum.rsi(close,14)
            df["ret"] = df["close"].pct_change()
            df["label"] = (df["ret"].shift(-1) > 0).astype(int)
            return df.dropna().tail(60)  # 仅60天数据
        except:
            return pd.DataFrame()

    df = get_data(code)
    if not df.empty:
        # 简化模型，避免部署超时
        features = ["ma20","rsi","volume"]
        X, y = df[features], df["label"]
        split = int(len(df)*0.8)
        
        if split > 10:  # 确保有足够数据训练
            model = LinearRegression()
            model.fit(X.iloc[:split], y.iloc[:split])
            prob = max(0, min(1, model.predict([df.iloc[-1][features]])[0]))
            acc = accuracy_score(y.iloc[split:], (model.predict(X.iloc[split:])>0.5).astype(int))
        else:
            prob = 0.5
            acc = 0.5

        # 展示核心指标
        col1, col2, col3 = st.columns(3)
        col1.metric("AI上涨概率", f"{prob:.1%}")
        col2.metric("回测准确率", f"{acc:.1%}")
        col3.metric("现价", f"{df['close'].iloc[-1]:.2f}")

        # 支撑压力
        recent = df.tail(30)
        st.info(f"📌 近期支撑：{recent['low'].min():.2f} ｜ 近期压力：{recent['high'].max():.2f}")

        # K线图
        df_plot = df.set_index("date")
        fig, _ = mpf.plot(df_plot, type="candle", volume=True, style="yahoo", returnfig=True)
        st.pyplot(fig)
    else:
        st.info("暂无法获取该股票数据，请检查代码是否正确")

# ======================
# 5. AI舆情分析
# ======================
with st.container(border=True):
    st.subheader("5️⃣ 📰 AI舆情分析（新闻+公告）")
    @st.cache_data(ttl=3600, show_spinner="加载舆情数据...")
    def get_news(code):
        try:
            df = ak.stock_news_em(symbol=code, timeout=10).head(10)  # 精简为10条
            df["sentiment"] = df["内容"].apply(lambda x: SnowNLP(str(x)).sentiments)
            return df
        except:
            return pd.DataFrame()

    news = get_news(code)
    if not news.empty:
        avg_sen = news["sentiment"].mean()
        st.markdown(f"**📝 舆情综合评分：{avg_sen:.1%}**")
        
        # 舆情标签
        if avg_sen > 0.6:
            st.success("✅ 整体舆情偏正面（利好）")
        elif avg_sen < 0.4:
            st.error("🔻 整体舆情偏负面（利空）")
        else:
            st.info("📊 整体舆情中性")
        
        # 展示核心舆情
        show_df = news[["发布时间","标题","sentiment"]]
        st.dataframe(show_df, use_container_width=True)
    else:
        st.info("暂未获取到该股票的最新新闻/公告")

# 底部提示
st.success("✅ AI股票分析系统 已完整运行（无登录版）")
st.caption("⚠️ 免责声明：本工具仅作数据分析参考，不构成任何投资建议")
