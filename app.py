# app.py
# Bitget → Excel (Historical candles + RSI, % change, Volume)
# Works on Streamlit Cloud. No API keys needed for public OHLCV.

from __future__ import annotations
import io
from typing import Dict, List, Tuple
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np
import streamlit as st
import ccxt


# ----------------------------- UI / Page config ----------------------------- #
st.set_page_config(
    page_title="Bitget → Excel (Full history + Indicators)",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Bitget → Excel (Full History + Indicators)")
st.caption(
    "Fetch historical candles from Bitget with **ccxt**, compute RSI & % change, "
    "and download a single Excel with one sheet per (symbol, timeframe)."
)


# ----------------------------- Helpers (cached) ----------------------------- #
@st.cache_data(show_spinner=False)
def get_exchange() -> ccxt.bitget:
    ex = ccxt.bitget({"enableRateLimit": True})
    ex.load_markets()
    return ex


@st.cache_data(show_spinner=False)
def get_bitget_symbols() -> List[str]:
    ex = get_exchange()
    # Spot USDT pairs only, sorted.
    return sorted([s for s in ex.symbols if s.endswith("/USDT") and ":" not in s])


@st.cache_data(show_spinner=False)
def get_bitget_timeframes() -> List[str]:
    ex = get_exchange()
    if getattr(ex, "timeframes", None):
        return sorted(ex.timeframes.keys())
    # Fallback (ccxt normally exposes timeframes)
    return ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "1w", "1M"]


# ----------------------------- Indicators ---------------------------------- #
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    down = -delta.clip(upper=0).ewm(alpha=1/period, adjust=False).mean()
    rs = up / down
    return 100 - (100 / (1 + rs))


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rsi"] = rsi(out["close"])
    out["pct_change"] = out["close"].pct_change() * 100.0
    return out


# ----------------------------- Data fetching -------------------------------- #
def millis(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def fetch_ohlcv_full(
    ex: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    days: int,
    limit_per_call: int = 1000,
) -> pd.DataFrame:
    """
    Pulls a rolling window of OHLCV for `days` back from now.
    Returns a DataFrame with columns: time, open, high, low, close, volume.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    since = millis(start)

    frames = []
    while True:
        candles = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit_per_call)
        if not candles:
            break

        chunk = pd.DataFrame(
            candles, columns=["time", "open", "high", "low", "close", "volume"]
        )
        frames.append(chunk)

        # Move forward: next "since" just after last returned candle
        last_ts = candles[-1][0]
        # If last candle didn't advance (rare), break to avoid infinite loop
        if last_ts <= since:
            break
        since = last_ts + 1

        # Stop if we already crossed "now"
        if last_ts >= millis(end):
            break

    if not frames:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["time"])
    # Make datetime **timezone-naive** for Excel (fixes the Excel tz error)
    dt_utc = pd.to_datetime(df["time"], unit="ms", utc=True)
    df["time"] = dt_utc.tz_convert(None)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values("time").reset_index(drop=True)
    return df


# ----------------------------- Excel builder -------------------------------- #
def to_excel_bytes(dfs: Dict[Tuple[str, str], pd.DataFrame]) -> bytes:
    """
    Builds a single Excel (one sheet per (symbol, timeframe)).
    Ensures all datetimes are timezone-naive before writing.
    """
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter", datetime_format="yyyy-mm-dd hh:mm:ss") as writer:
        for (symbol, tf), df in dfs.items():
            if df.empty:
                continue
            safe = df.copy()
            # Double-ensure tz-naive:
            if pd.api.types.is_datetime64_any_dtype(safe["time"]):
                # If tz-aware, remove tz; if already naive, this is a no-op
                if getattr(safe["time"].dt, "tz", None) is not None:
                    safe["time"] = safe["time"].dt.tz_localize(None)
            sheet = f"{symbol.replace('/','')}_{tf}"
            # Excel sheet name limit 31 chars:
            sheet = (sheet[:28] + "...") if len(sheet) > 31 else sheet
            safe.to_excel(writer, sheet_name=sheet, index=False)
    return buf.getvalue()


# ----------------------------- Sidebar (selectors) -------------------------- #
with st.sidebar:
    st.header("Settings")
    # Symbols
    symbols = st.multiselect(
        "Select symbols (type to add more, e.g., XRP/USDT)",
        options=get_bitget_symbols(),
        default=["BTC/USDT", "ETH/USDT", "SOL/USDT", "LINK/USDT", "ADA/USDT"],
    )

    # Timeframes
    available_tfs = get_bitget_timeframes()
    default_tfs = [tf for tf in ["15m", "1h", "4h"] if tf in available_tfs]
    timeframes = st.multiselect(
        "Select timeframes",
        options=available_tfs,
        default=default_tfs or available_tfs[:3],
    )

    st.markdown("### History window (days)")
    lookbacks: Dict[str, int] = {}
    for tf in timeframes:
        default_days = 3 if tf == "15m" else 21 if tf == "1h" else 90 if tf == "4h" else 30
        lookbacks[tf] = int(
            st.number_input(f"{tf} lookback (days)", min_value=1, max_value=365, value=default_days, step=1)
        )


# ----------------------------- Main action ---------------------------------- #
if st.button("🚀 Fetch & Build Excel", type="primary", use_container_width=True):
    if not symbols or not timeframes:
        st.error("Please select at least one symbol and one timeframe.")
        st.stop()

    ex = get_exchange()
    results: Dict[Tuple[str, str], pd.DataFrame] = {}

    progress = st.progress(0.0)
    tasks = len(symbols) * len(timeframes)
    done = 0

    for sym in symbols:
        st.subheader(sym)
        cols = st.columns(len(timeframes))
        for i, tf in enumerate(timeframes):
            with cols[i]:
                st.write(f"Fetching **{tf}** …")
                try:
                    df = fetch_ohlcv_full(ex, sym, tf, lookbacks.get(tf, 30))
                    if df.empty:
                        st.warning(f"No data for {sym} {tf}")
                        continue
                    df = add_indicators(df)
                    results[(sym, tf)] = df
                    st.success(f"{len(df):,} rows")
                except Exception as e:
                    st.error(f"Error fetching {sym} {tf}: {e}")
                finally:
                    done += 1
                    progress.progress(done / tasks)

    if not results:
        st.error("No data collected. Please adjust selections and try again.")
        st.stop()

    # Build Excel
    try:
        excel_bytes = to_excel_bytes(results)
    except Exception as e:
        st.error(f"Error while building Excel: {e}")
        st.stop()

    # Download button
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"bitget_data_{ts}.xlsx"
    st.success("✅ Excel file ready.")
    st.download_button(
        "⬇️ Download Excel",
        data=excel_bytes,
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

# Helpful footer
st.caption(
    "Tip: If a dropdown shows *No results*, your options list would be empty. "
    "This app fetches symbols/timeframes **before** rendering, so you should always see results now."
)
