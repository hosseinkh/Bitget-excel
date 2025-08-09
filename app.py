import streamlit as st
import pandas as pd
import numpy as np
import ccxt
import time
from datetime import datetime
import io

# ================== SETTINGS ==================
DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "LINK/USDT", "ADA/USDT"]
DEFAULT_TIMEFRAMES = ["15m", "1h", "4h"]
# ===============================================

st.set_page_config(page_title="Bitget → Excel Logger", page_icon="📊", layout="wide")

# ----------------- Helpers ------------------
def to_naive_utc(dt_series: pd.Series) -> pd.Series:
    """Ensure timestamps are timezone-naive (Excel safe)."""
    s = pd.to_datetime(dt_series, errors="coerce")
    try:
        return s.dt.tz_localize(None)
    except Exception:
        return s

# CCXT setup
_EXCHANGE = ccxt.bitget({"enableRateLimit": True})
CCXT_TF = {"15m": "15m", "1h": "1h", "4h": "4h"}

def candles_needed(days: int, timeframe: str) -> int:
    mins = {"15m": 15, "1h": 60, "4h": 240}[timeframe]
    return int((days * 24 * 60) / mins) + 10

def get_bitget_klines(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """
    Fetch a window of OHLCV using ccxt, paginating until we have 'days' worth.
    Returns ascending by time with columns: time_utc, open, high, low, close, volume
    """
    tf = CCXT_TF[timeframe]
    need = candles_needed(days, timeframe)
    out = []
    since_ms = int((pd.Timestamp.utcnow() - pd.Timedelta(days=days)).timestamp() * 1000)
    limit = 500

    while len(out) < need:
        batch = _EXCHANGE.fetch_ohlcv(symbol, timeframe=tf, since=since_ms, limit=limit)
        if not batch:
            break
        out.extend(batch)
        since_ms = batch[-1][0] + 1
        time.sleep(_EXCHANGE.rateLimit / 1000.0)
        if len(batch) < 5:
            break

    if not out:
        return pd.DataFrame()

    df = pd.DataFrame(out, columns=["ts", "open", "high", "low", "close", "volume"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["time_utc"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_localize(None)
    return df.sort_values("time_utc").reset_index(drop=True)

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI(14), MA20, MA50, High/Low(20), % distances."""
    if df.empty:
        return df.copy()

    out = df.copy()
    delta = out["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    out["RSI"] = 100 - (100 / (1 + rs))
    out["MA20"] = out["close"].rolling(20).mean()
    out["MA50"] = out["close"].rolling(50).mean()
    out["High_20"] = out["high"].rolling(20).max()
    out["Low_20"] = out["low"].rolling(20).min()
    out["%_from_High20"] = (out["close"] - out["High_20"]) / out["High_20"] * 100
    out["%_from_Low20"] = (out["close"] - out["Low_20"]) / out["Low_20"] * 100
    return out

def build_excel(dfs: dict) -> bytes:
    """Build an Excel with one sheet per (symbol, timeframe), with Close charts."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter", datetime_format="yyyy-mm-dd HH:MM") as writer:
        for (symbol, tf), df in dfs.items():
            df = df.copy()
            if "time_utc" in df.columns:
                df["time_utc"] = to_naive_utc(df["time_utc"])
            sheet = f"{symbol.replace('/', '')}_{tf}"
            df.to_excel(writer, sheet_name=sheet, index=False)

            workbook = writer.book
            worksheet = writer.sheets[sheet]
            chart = workbook.add_chart({"type": "line"})
            last_row = len(df) + 1
            chart.add_series({
                "name": "Close",
                "categories": [sheet, 1, 0, last_row, 0],
                "values": [sheet, 1, 4, last_row, 4],
            })
            chart.set_title({"name": f"{symbol} {tf} Close"})
            chart.set_legend({"position": "bottom"})
            worksheet.insert_chart("L2", chart)

    buf.seek(0)
    return buf

# -----------------------------------------------

st.title("📊 Bitget → Excel (Fresh History)")

# Sidebar controls
symbols = st.sidebar.multiselect("Select symbols", DEFAULT_SYMBOLS, DEFAULT_SYMBOLS)
timeframes = st.sidebar.multiselect("Select timeframes", DEFAULT_TIMEFRAMES, DEFAULT_TIMEFRAMES)

st.sidebar.markdown("### History window (days)")
lookback_15m = st.sidebar.number_input("15m lookback (days)", min_value=1, max_value=14, value=3)
lookback_1h = st.sidebar.number_input("1h lookback (days)", min_value=2, max_value=60, value=21)
lookback_4h = st.sidebar.number_input("4h lookback (days)", min_value=7, max_value=180, value=90)
LOOKBACK_BY_TF = {"15m": lookback_15m, "1h": lookback_1h, "4h": lookback_4h}

# Main fetch button
if st.button("🚀 Fetch & Build Excel"):
    all_dfs: dict[tuple[str, str], pd.DataFrame] = {}
    for sym in symbols:
        for tf in timeframes:
            st.write(f"Fetching **{sym} {tf}** …")
            df_new = get_bitget_klines(sym, tf, days=LOOKBACK_BY_TF[tf])
            if df_new.empty:
                st.warning(f"No data for {sym} {tf}")
                continue
            df_new = compute_indicators(df_new)
            all_dfs[(sym, tf)] = df_new

    if not all_dfs:
        st.warning("No data fetched.")
    else:
        excel_bytes = build_excel(all_dfs)
        st.download_button(
            label="📥 Download Excel",
            data=excel_bytes,
            file_name=f"bitget_data_{datetime.now().strftime('%Y%m%d-%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
