# app.py
# Bitget -> Excel (Full list selectors + RSI, % Change)
# Works on Streamlit Cloud. No API key needed (public OHLCV).

from __future__ import annotations

import io
import math
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
import ccxt

# ---------------------------
# Streamlit page config
# ---------------------------
st.set_page_config(
    page_title="Bitget → Excel (Fetch + Indicators)",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Bitget → Excel (Fetch + Indicators)")
st.caption(
    "Fetch historical candles from Bitget, compute RSI & % change, "
    "and download a single Excel with one sheet per (symbol, timeframe)."
)

# ---------------------------
# Helpers
# ---------------------------

@st.cache_data(show_spinner=False)
def get_exchange() -> ccxt.Exchange:
    # enableRateLimit helps avoid 429s
    ex = ccxt.bitget({"enableRateLimit": True})
    ex.load_markets()
    return ex

@st.cache_data(show_spinner=False)
def list_spot_usdt_symbols() -> List[str]:
    ex = get_exchange()
    symbols = []
    for sym, info in ex.markets.items():
        # Keep spot USDT pairs (e.g., BTC/USDT)
        if info.get("spot") and sym.endswith("/USDT"):
            symbols.append(sym)
    symbols.sort()
    return symbols

@st.cache_data(show_spinner=False)
def list_supported_timeframes() -> List[str]:
    ex = get_exchange()
    # ccxt exposes exchange.timeframes for supported OHLCV frames
    # If empty, fallback to common ones.
    tf_map = ex.timeframes or {}
    if tf_map:
        # return the keys in a friendly order
        order = ["1m","3m","5m","15m","30m","1h","2h","4h","6h","8h","12h","1d","3d","1w","1M"]
        return [t for t in order if t in tf_map]
    return ["15m", "1h", "4h", "1d"]

def tf_to_ms(tf: str) -> int:
    # Convert timeframe string to milliseconds
    unit = tf[-1]
    n = int(tf[:-1])
    if unit == "m":
        return n * 60_000
    if unit == "h":
        return n * 60 * 60_000
    if unit == "d":
        return n * 24 * 60 * 60_000
    if unit == "w":
        return n * 7 * 24 * 60 * 60_000
    if unit == "M":
        # approximate month as 30d for paging purposes
        return n * 30 * 24 * 60 * 60_000
    raise ValueError(f"Unsupported timeframe: {tf}")

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    down = (-delta).clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    rs = up / down
    return 100 - (100 / (1 + rs))

def fetch_ohlcv_all(
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
    limit_per_call: int = 1000,
) -> pd.DataFrame:
    """Page through OHLCV from 'since_ms' until 'until_ms'."""
    ex = get_exchange()
    all_rows: List[List[float]] = []
    tf_ms = tf_to_ms(timeframe)
    cursor = since_ms
    safety = 0
    while cursor < until_ms and safety < 800:  # safety bound
        try:
            batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit_per_call)
        except Exception as e:
            # 404 or symbol/tf not supported, etc.
            raise RuntimeError(f"fetch_ohlcv failed for {symbol} {timeframe}: {e}")
        if not batch:
            break
        all_rows.extend(batch)
        # Advance cursor to last candle + 1 tf to avoid overlap
        last_ts = batch[-1][0]
        cursor = max(cursor + tf_ms, last_ts + tf_ms)
        safety += 1

    if not all_rows:
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

    df = pd.DataFrame(all_rows, columns=["timestamp","open","high","low","close","volume"])
    # Convert to timezone-naive datetime for Excel
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).tz_convert(None)
    df = df[["time","open","high","low","close","volume"]]
    df = df.sort_values("time").reset_index(drop=True)
    return df

def build_excel(dfs: Dict[Tuple[str,str], pd.DataFrame]) -> bytes:
    """Create a single Excel file; each sheet is SYMBOL_TF."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for (symbol, tf), df in dfs.items():
            if df.empty:
                continue
            # Add indicators
            df2 = df.copy()
            df2["pct_change"] = df2["close"].pct_change() * 100
            df2["rsi_14"] = compute_rsi(df2["close"], period=14)
            sheet_name = f"{symbol.replace('/','')}_{tf}"
            # Excel sheet names max 31 chars
            if len(sheet_name) > 31:
                sheet_name = sheet_name[:31]
            df2.to_excel(writer, sheet_name=sheet_name, index=False)
    return output.getvalue()

# ---------------------------
# Sidebar controls
# ---------------------------

with st.sidebar:
    st.header("Settings")

    # Symbols selector (populated from Bitget spot USDT list)
    all_symbols = list_spot_usdt_symbols()
    default_syms = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "LINK/USDT", "ADA/USDT"]
    default_syms = [s for s in default_syms if s in all_symbols]

    symbols = st.multiselect(
        "Select symbols",
        options=all_symbols,
        default=default_syms,
        placeholder="Type to search (e.g., XRP/USDT)…",
        help="All Bitget spot /USDT pairs are listed. You can type to filter.",
    )

    # Optional manual add
    manual_sym = st.text_input("Add symbol manually (optional, e.g., XRP/USDT)")
    if manual_sym:
        manual_sym = manual_sym.strip().upper()
        if manual_sym and manual_sym not in symbols:
            symbols.append(manual_sym)

    # Timeframes selector (populated from exchange.timeframes)
    all_tfs = list_supported_timeframes()
    default_tfs = [t for t in ["15m", "1h", "4h"] if t in all_tfs]

    timeframes = st.multiselect(
        "Select timeframes",
        options=all_tfs,
        default=default_tfs,
        help="Timeframes supported by Bitget via ccxt.",
    )

    # Lookback window per timeframe (days)
    st.subheader("History window (days)")
    lb_15m = st.number_input("15m lookback (days)", min_value=1, max_value=14, value=3, step=1)
    lb_1h  = st.number_input("1h lookback (days)",  min_value=1, max_value=60, value=21, step=1)
    lb_4h  = st.number_input("4h lookback (days)",  min_value=1, max_value=180, value=90, step=1)

    # Map each tf to its chosen lookback (days). If a tf isn't listed, default to 7d.
    tf_lookback_days: Dict[str, int] = {tf: 7 for tf in timeframes}
    if "15m" in timeframes: tf_lookback_days["15m"] = lb_15m
    if "1h"  in timeframes: tf_lookback_days["1h"]  = lb_1h
    if "4h"  in timeframes: tf_lookback_days["4h"]  = lb_4h

# ---------------------------
# Main action
# ---------------------------

if st.button("🚀 Fetch & Build Excel", use_container_width=True):
    if not symbols:
        st.warning("Please select at least one symbol.")
        st.stop()
    if not timeframes:
        st.warning("Please select at least one timeframe.")
        st.stop()

    results: Dict[Tuple[str,str], pd.DataFrame] = {}
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    for sym in symbols:
        st.markdown(f"**Fetching `{sym}`…**")
        for tf in timeframes:
            st.write(f"Fetching **{sym} {tf}** …")
            days = tf_lookback_days.get(tf, 7)
            since_ms = now_ms - days * 24 * 60 * 60 * 1000
            try:
                df = fetch_ohlcv_all(sym, tf, since_ms, now_ms)
            except Exception as e:
                st.error(f"Error fetching {sym} {tf}: {e}")
                df = pd.DataFrame(columns=["time","open","high","low","close","volume"])

            if df.empty:
                st.warning(f"No data for {sym} {tf}")
            else:
                # quick stats preview
                last = df.iloc[-1]
                st.write(
                    f"Rows: {len(df)} | Last: {last['time']} "
                    f"Close: {last['close']:.4f} | Volume: {last['volume']:.4f}"
                )
                results[(sym, tf)] = df

    if not results:
        st.error("Nothing to write. Check symbols/timeframes and try again.")
        st.stop()

    st.success(f"Fetched {len(results)} sheet(s). Building Excel…")
    excel_bytes = build_excel(results)

    fname = f"bitget_excel_{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.xlsx"
    st.download_button(
        label="⬇️ Download Excel",
        data=excel_bytes,
        file_name=fname,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    st.info("Tip: Add this page to your Home Screen for one-tap access.")
