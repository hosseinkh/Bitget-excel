# app.py
import io
import time
from datetime import datetime, timezone
from typing import List, Dict, Tuple

import pandas as pd
import numpy as np
import streamlit as st
import ccxt

# ----------------------------
# UI / Page setup
# ----------------------------
st.set_page_config(page_title="Bitget → Excel (RSI, Volume, % Change)", page_icon="📈", layout="wide")
st.title("📊 Bitget → Excel (RSI, Volume, % Change)")
st.caption("Select symbols and timeframes, choose history window, then click **Fetch & Download Excel**.")

# ----------------------------
# Helpers
# ----------------------------
@st.cache_data(show_spinner=False)
def list_spot_markets() -> List[str]:
    """Return all Bitget spot USDT pairs like BTC/USDT, SOL/USDT, etc."""
    ex = ccxt.bitget({"enableRateLimit": True})
    markets = ex.load_markets()
    return sorted([m for m, info in markets.items() if info.get("spot") and m.endswith("/USDT")])

def timeframe_minutes(tf: str) -> int:
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    if tf.endswith("d"):
        return int(tf[:-1]) * 60 * 24
    raise ValueError(f"Unsupported timeframe: {tf}")

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-style RSI using EMA smoothing."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)

def compute_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rsi14"] = rsi(df["close"], 14).round(2)
    df["pct_change_1"] = df["close"].pct_change().fillna(0).round(4)
    # distance to 30-candle low (context)
    ll30 = df["low"].rolling(30, min_periods=1).min()
    df["dist_to_ll30_%"] = ((df["close"] - ll30) / ll30 * 100).round(2)
    return df

def fetch_ohlcv(ex: ccxt.Exchange, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """
    Fetch OHLCV for symbol/timeframe going back 'days'.
    Returns DataFrame with time, open, high, low, close, volume in UTC.
    """
    tf_min = timeframe_minutes(timeframe)
    # conservative limit calculation
    est_needed = int(np.ceil(days * 24 * 60 / tf_min)) + 10
    limit = min(3000, max(100, est_needed))
    since_ms = int((time.time() - days * 86400) * 1000)

    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
    if not ohlcv:
        return pd.DataFrame()

    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop(columns=["ts"]).set_index("time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.reset_index()

def build_excel(dfs: Dict[Tuple[str, str], pd.DataFrame]) -> bytes:
    """Create an Excel file in memory with one sheet per (symbol, timeframe)."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for (symbol, tf), df in dfs.items():
            if df.empty:
                continue
            sheet_name = f"{symbol.replace('/','')}_{tf}"
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    return output.getvalue()

# ----------------------------
# Controls
# ----------------------------
ALL_SYMBOLS = list_spot_markets()
DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "LINK/USDT", "ADA/USDT"]

with st.expander("Settings", expanded=True):
    symbols = st.multiselect(
        "Select symbols",
        options=ALL_SYMBOLS,
        default=[s for s in DEFAULT_SYMBOLS if s in ALL_SYMBOLS],
        help="Add/remove any Bitget spot USDT pairs.",
    )
    extra = st.text_input("Add more symbols (comma separated, e.g. INJ/USDT, XRP/USDT)").strip()
    if extra:
        added = [x.strip().upper() for x in extra.split(",") if x.strip()]
        added = [x for x in added if x.endswith("/USDT")]
        symbols = sorted(set(symbols + added))

    TF_OPTIONS = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    timeframes = st.multiselect(
        "Select timeframes",
        options=TF_OPTIONS,
        default=["15m", "1h", "4h"],
    )

    days = st.number_input(
        "History window (days) for each selected timeframe",
        min_value=1, max_value=60, value=3, step=1,
        help="How many days of candles per timeframe to fetch (e.g., 3 days of 15m/1h/4h)."
    )

# ----------------------------
# Action
# ----------------------------
col1, col2 = st.columns([1,2])
with col1:
    start = st.button("🚀 Fetch & Download Excel", type="primary", use_container_width=True)

if start:
    if not symbols or not timeframes:
        st.error("Please select at least one symbol and one timeframe.")
        st.stop()

    st.info(f"Fetching data from Bitget for {len(symbols)} symbols × {len(timeframes)} TFs…")
    ex = ccxt.bitget({"enableRateLimit": True})
    results: Dict[Tuple[str, str], pd.DataFrame] = {}

    progress = st.progress(0.0)
    total = len(symbols) * len(timeframes)
    done = 0

    for sym in symbols:
        for tf in timeframes:
            done += 1
            progress.progress(done / total)
            with st.spinner(f"Fetching {sym} {tf} …"):
                try:
                    raw = fetch_ohlcv(ex, sym, tf, days)
                    if raw.empty:
                        st.warning(f"No data for {sym} {tf}")
                        continue
                    df = compute_columns(raw)
                    results[(sym, tf)] = df
                except Exception as e:
                    st.error(f"Error fetching {sym} {tf}: {e}")

    if not results:
        st.warning("Nothing fetched. Check selections and try again.")
        st.stop()

    # Build Excel
    excel_bytes = build_excel(results)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    filename = f"bitget_data_{ts}.xlsx"

    st.success("Done! Download your Excel below 👇")
    st.download_button(
        "⬇️ Download Excel",
        data=excel_bytes,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    st.caption("Each sheet is named like `BTCUSDT_15m`, `ETHUSDT_1h`, etc. Columns include OHLCV, RSI(14), % change, and distance to 30-candle low.")
