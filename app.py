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
    "15m": 2,   # keep last 2 days of 15m
    "1h": 7,    # keep last 7 days of 1h
    "4h": 30,   # keep last 30 days of 4h
}
# ===============================================

st.set_page_config(page_title="Bitget → Excel Logger", page_icon="📊", layout="wide")

# ----------------- Helpers ------------------
def to_naive_utc(dt_series: pd.Series) -> pd.Series:
    """
    Ensure timestamps are timezone-naive (required by Excel).
    Accepts tz-aware or naive; returns naive (no tz).
    """
    s = pd.to_datetime(dt_series, errors="coerce")
    # If tz-aware, drop tz; if already naive, this is a no-op
    try:
        return s.dt.tz_localize(None)
    except Exception:
        return s

def get_bitget_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    """
    Fetch latest kline data from Bitget and return ascending by time.
    """
    url = f"https://api.bitget.com/api/v2/market/candles?symbol={symbol}&granularity={interval}&limit={limit}"
    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        st.error(f"Error fetching {symbol} {interval}: {r.text}")
        return pd.DataFrame()
    data = r.json().get("data", [])
    if not data:
        return pd.DataFrame()

    # Bitget returns [ts, open, high, low, close, volume, turnover]
    df = pd.DataFrame(data, columns=["ts", "open", "high", "low", "close", "volume", "turnover"])

    # Ensure numeric dtypes
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Make a timezone-naive UTC column compatible with Excel
    # (start with utc=True to avoid local-time ambiguity, then strip tz)
    df["time_utc"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df["time_utc"] = df["time_utc"].dt.tz_localize(None)

    df = df.sort_values("time_utc").reset_index(drop=True)
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add RSI(14), MA20, MA50, rolling High/Low(20) and % distances.
    """
    if df.empty:
        return df.copy()

    out = df.copy()

    # RSI(14)
    delta = out["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    out["RSI"] = 100 - (100 / (1 + rs))

    # MAs
    out["MA20"] = out["close"].rolling(20).mean()
    out["MA50"] = out["close"].rolling(50).mean()

    # High/Low(20)
    out["High_20"] = out["high"].rolling(20).max()
    out["Low_20"] = out["low"].rolling(20).min()

    # % from High/Low
    out["%_from_High20"] = (out["close"] - out["High_20"]) / out["High_20"] * 100
    out["%_from_Low20"] = (out["close"] - out["Low_20"]) / out["Low_20"] * 100

    return out

def trim_rolling_history(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Keep only the last N days (per timeframe) so files stay small.
    """
    if df.empty or "time_utc" not in df.columns:
        return df
    tmp = df.copy()
    tmp["time_utc"] = to_naive_utc(tmp["time_utc"])
    cutoff = tmp["time_utc"].max() - timedelta(days=ROLLING_DAYS.get(timeframe, 30))
    return tmp[tmp["time_utc"] >= cutoff].copy()

def merge_history(new_df: pd.DataFrame, old_df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Append old + new, drop duplicates by time, sort ascending.
    Force timestamps to be timezone-naive.
    """
    if old_df is None or old_df.empty:
        out = new_df.copy()
    else:
        # Normalize old file columns (in case)
        old = old_df.copy()
        if "time_utc" in old.columns:
            old["time_utc"] = to_naive_utc(old["time_utc"])
        out = pd.concat([old, new_df], ignore_index=True)

    # Ensure naive timestamps and drop dups
    if "time_utc" in out.columns:
        out["time_utc"] = to_naive_utc(out["time_utc"])
        out = out.drop_duplicates(subset=["time_utc"], keep="last")

    return out.sort_values("time_utc").reset_index(drop=True)

def build_excel(dfs: dict) -> bytes:
    """
    Build an Excel with one sheet per (symbol, timeframe).
    Also inserts a simple Close-price line chart on each sheet.
    """
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter", datetime_format="yyyy-mm-dd HH:MM") as writer:
        for (symbol, tf), df in dfs.items():
            # final safety: timestamps must be naive
            if "time_utc" in df.columns:
                df = df.copy()
                df["time_utc"] = to_naive_utc(df["time_utc"])

            df = trim_rolling_history(df, tf)

            sheet = f"{symbol}_{tf}"
            df.to_excel(writer, sheet_name=sheet, index=False)

            # Add a quick Close chart
            workbook = writer.book
            worksheet = writer.sheets[sheet]
            chart = workbook.add_chart({"type": "line"})
            last_row = len(df) + 1  # header is row 0 in Excel writer

            # time_utc is col 0; close is col 4 after our to_excel order
            chart.add_series({
                "name": "Close",
                "categories": [sheet, 1, 0, last_row, 0],
                "values":     [sheet, 1, 4, last_row, 4],
            })
            chart.set_title({"name": f"{symbol} {tf} Close"})
            chart.set_legend({"position": "bottom"})
            worksheet.insert_chart("L2", chart)

    buf.seek(0)
    return buf
# -----------------------------------------------

st.title("📊 Bitget → Excel (History, Append, Trimming)")

# Sidebar
symbols = st.sidebar.multiselect("Select symbols", DEFAULT_SYMBOLS, DEFAULT_SYMBOLS)
timeframes = st.sidebar.multiselect("Select timeframes", DEFAULT_TIMEFRAMES, DEFAULT_TIMEFRAMES)
uploaded_file = st.sidebar.file_uploader("Upload previous Excel to append (optional)", type=["xlsx"])

# Load existing Excel (append mode)
existing_data: dict[tuple[str, str], pd.DataFrame] = {}
if uploaded_file:
    try:
        xls = pd.ExcelFile(uploaded_file)
        for sh in xls.sheet_names:
            try:
                df_old = pd.read_excel(xls, sheet_name=sh)
                # Expect sheet like "BTCUSDT_15m"
                if "_" in sh:
                    tf = sh.split("_")[-1]
                    sym = sh[: -(len(tf) + 1)]
                    # force naive ts in existing file too
                    if "time_utc" in df_old.columns:
                        df_old["time_utc"] = to_naive_utc(df_old["time_utc"])
                    existing_data[(sym, tf)] = df_old
            except Exception as e:
                st.warning(f"Could not read sheet {sh}: {e}")
    except Exception as e:
        st.error(f"Failed to open uploaded Excel: {e}")

if st.button("🚀 Fetch & Build Excel"):
    all_dfs: dict[tuple[str, str], pd.DataFrame] = {}

    for sym in symbols:
        for tf in timeframes:
            st.write(f"Fetching **{sym} {tf}** …")
            df_new = get_bitget_klines(sym, tf)
            if df_new.empty:
                st.warning(f"No data for {sym} {tf}")
                continue

            df_new = compute_indicators(df_new)

            df_old = existing_data.get((sym, tf))
            df_merged = merge_history(df_new, df_old)

            all_dfs[(sym, tf)] = df_merged

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
