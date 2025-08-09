import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import io

# ================== SETTINGS ==================
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "LINKUSDT", "ADAUSDT"]
DEFAULT_TIMEFRAMES = ["15m", "1h", "4h"]
ROLLING_DAYS = {
    "15m": 2,
    "1h": 7,
    "4h": 30
}
# ===============================================

st.set_page_config(page_title="Bitget → Excel Logger", page_icon="📊", layout="wide")

# ----------------- Functions ------------------
def get_bitget_klines(symbol, interval, limit=200):
    """Fetch latest kline data from Bitget."""
    url = f"https://api.bitget.com/api/v2/market/candles?symbol={symbol}&granularity={interval}&limit={limit}"
    r = requests.get(url)
    if r.status_code != 200:
        st.error(f"Error fetching {symbol} {interval}: {r.text}")
        return pd.DataFrame()
    data = r.json().get("data", [])
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "volume", "turnover"])
    df["time_utc"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.astype({
        "open": float, "high": float, "low": float, "close": float, "volume": float
    })
    return df.sort_values("time_utc").reset_index(drop=True)

def compute_indicators(df):
    """Add RSI, moving averages, highs/lows, % change."""
    # RSI
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    # MAs
    df["MA20"] = df["close"].rolling(20).mean()
    df["MA50"] = df["close"].rolling(50).mean()
    # High/Low
    df["High_20"] = df["high"].rolling(20).max()
    df["Low_20"] = df["low"].rolling(20).min()
    # % Change from High/Low
    df["%_from_High20"] = (df["close"] - df["High_20"]) / df["High_20"] * 100
    df["%_from_Low20"] = (df["close"] - df["Low_20"]) / df["Low_20"] * 100
    return df

def trim_rolling_history(df, timeframe):
    """Trim to rolling days for each timeframe."""
    if "time_utc" not in df.columns:
        return df
    df["time_utc"] = pd.to_datetime(df["time_utc"], errors="coerce")
    cutoff = df["time_utc"].max() - timedelta(days=ROLLING_DAYS.get(timeframe, 30))
    return df[df["time_utc"] >= cutoff].copy()

def merge_history(new_df, old_df):
    """Merge old and new data without duplicates."""
    if old_df is None or old_df.empty:
        return new_df
    combined = pd.concat([old_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["time_utc"], keep="last")
    combined = combined.sort_values("time_utc").reset_index(drop=True)
    return combined

def build_excel(dfs):
    """Build Excel with charts for each sheet."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter", datetime_format="yyyy-mm-dd HH:MM") as writer:
        for (symbol, tf), df in dfs.items():
            df = trim_rolling_history(df, tf)
            sheet_name = f"{symbol}_{tf}"
            df.to_excel(writer, sheet_name=sheet_name, index=False)

            # Add chart
            workbook = writer.book
            worksheet = writer.sheets[sheet_name]
            chart = workbook.add_chart({"type": "line"})
            last_row = len(df) + 1
            chart.add_series({
                "name": "Close",
                "categories": [sheet_name, 1, 0, last_row, 0],
                "values": [sheet_name, 1, 4, last_row, 4],
            })
            chart.set_title({"name": f"{symbol} {tf}"})
            worksheet.insert_chart("L2", chart)
    buf.seek(0)
    return buf
# -----------------------------------------------

st.title("📊 Bitget Data Logger with History & Trimming")

# Sidebar selections
symbols = st.sidebar.multiselect("Select symbols", DEFAULT_SYMBOLS, DEFAULT_SYMBOLS)
timeframes = st.sidebar.multiselect("Select timeframes", DEFAULT_TIMEFRAMES, DEFAULT_TIMEFRAMES)
uploaded_file = st.sidebar.file_uploader("Upload previous Excel to append (optional)", type=["xlsx"])

existing_data = {}
if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    for sh in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sh)
            # Extract symbol/timeframe from sheet name
            parts = sh.split("_")
            tf = parts[-1]
            symbol = "_".join(parts[:-1])
            existing_data[(symbol, tf)] = df
        except Exception as e:
            st.warning(f"Could not read sheet {sh}: {e}")

if st.button("Fetch & Build Excel"):
    all_dfs = {}
    for sym in symbols:
        for tf in timeframes:
            st.write(f"Fetching {sym} {tf}...")
            df = get_bitget_klines(sym, tf)
            if df.empty:
                continue
            df = compute_indicators(df)
            old_df = existing_data.get((sym, tf))
            df = merge_history(df, old_df)
            all_dfs[(sym, tf)] = df

    if all_dfs:
        excel_bytes = build_excel(all_dfs)
        st.download_button(
            label="📥 Download Excel",
            data=excel_bytes,
            file_name=f"bitget_data_{datetime.now().strftime('%Y%m%d-%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.warning("No data fetched.")
