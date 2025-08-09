# app.py
import io
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import ccxt

# ----------------------------
# Page
# ----------------------------
st.set_page_config(page_title="Bitget → Excel (RSI, Volume, % Change)", page_icon="📈", layout="wide")
st.title("📊 Bitget → Excel (RSI, Volume, % Change)")
st.caption("Pick symbols & timeframes, choose history window, then **Fetch & Build Excel**.")

# ----------------------------
# Helpers
# ----------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def list_spot_markets() -> List[str]:
    """
    Load all Bitget spot USDT pairs. If the API fails (rate limit / network),
    return a static fallback so the selector never shows 'No results'.
    """
    fallback = [
        "BTC/USDT","ETH/USDT","SOL/USDT","LINK/USDT","ADA/USDT","XRP/USDT","INJ/USDT",
        "AVAX/USDT","ATOM/USDT","DOGE/USDT","TRX/USDT","OP/USDT","APT/USDT","AR/USDT",
    ]
    try:
        ex = ccxt.bitget({"enableRateLimit": True, "timeout": 20000})
        markets = ex.load_markets()
        syms = sorted([m for m, info in markets.items() if info.get("spot") and m.endswith("/USDT")])
        return syms if syms else fallback
    except Exception:
        return fallback

def timeframe_minutes(tf: str) -> int:
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    if tf.endswith("d"):
        return int(tf[:-1]) * 1440
    raise ValueError(f"Unsupported timeframe: {tf}")

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1/period, adjust=False).mean()
    avg_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rsi14"] = rsi(df["close"], 14).round(2)
    df["pct_change_1"] = df["close"].pct_change().fillna(0).round(4)
    ll30 = df["low"].rolling(30, min_periods=1).min()
    df["dist_to_ll30_%"] = ((df["close"] - ll30) / ll30 * 100).round(2)
    return df

def fetch_ohlcv(ex: ccxt.Exchange, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """
    Fetch OHLCV for symbol/timeframe going back 'days' days.
    Returns columns: time (UTC), open, high, low, close, volume.
    """
    tf_min = timeframe_minutes(timeframe)
    est = int(np.ceil(days * 24 * 60 / tf_min)) + 10
    limit = min(3000, max(100, est))
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
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        for (symbol, tf), df in dfs.items():
            if df.empty:
                continue
            sheet = f"{symbol.replace('/','')}_{tf}"
            df.to_excel(writer, sheet_name=sheet, index=False)
    return bio.getvalue()

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
        help="Multi-select any Bitget spot USDT pairs.",
    )

    extra = st.text_input("Add more symbols (comma separated, e.g. INJ/USDT, XRP/USDT)").strip()
    if extra:
        add = [x.strip().upper() for x in extra.split(",") if x.strip()]
        add = [x for x in add if x.endswith("/USDT")]
        symbols = sorted(set(symbols + add))

    TF_OPTIONS = ["1m","5m","15m","30m","1h","4h","1d"]
    timeframes = st.multiselect(
        "Select timeframes",
        options=TF_OPTIONS,
        default=["15m","1h","4h"],
    )

    days = st.number_input(
        "History window (days) for each selected timeframe",
        min_value=1, max_value=60, value=3, step=1,
        help="Example: 3 days of 15m/1h/4h candles."
    )

# ----------------------------
# Action
# ----------------------------
go = st.button("🚀 Fetch & Build Excel", type="primary", use_container_width=True)

if go:
    if not symbols:
        st.error("Select at least one symbol.")
        st.stop()
    if not timeframes:
        st.error("Select at least one timeframe.")
        st.stop()

    st.info(f"Fetching {len(symbols)} symbols × {len(timeframes)} timeframes from Bitget…")
    ex = ccxt.bitget({"enableRateLimit": True, "timeout": 20000})
    results: Dict[Tuple[str, str], pd.DataFrame] = {}

    total = len(symbols) * len(timeframes)
    p = st.progress(0.0)
    done = 0

    for sym in symbols:
        for tf in timeframes:
            done += 1
            p.progress(done / total)
            with st.spinner(f"Fetching {sym} {tf}…"):
                try:
                    raw = fetch_ohlcv(ex, sym, tf, days)
                    if raw.empty:
                        st.warning(f"No data for {sym} {tf}")
                        continue
                    results[(sym, tf)] = compute_indicators(raw)
                except Exception as e:
                    st.error(f"Error fetching {sym} {tf}: {e}")

    if not results:
        st.warning("Nothing fetched. Try different selections or increase history window.")
        st.stop()

    data = build_excel(results)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    fname = f"bitget_data_{ts}.xlsx"

    st.success("Done! Download your Excel below 👇")
    st.download_button(
        "⬇️ Download Excel",
        data=data,
        file_name=fname,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    st.caption("Each sheet is named `SYMBOLTF` (e.g., `BTCUSDT_15m`). Columns include OHLCV, RSI(14), % change, and distance to the 30-candle low.")
