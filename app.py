import io
import time
from datetime import datetime

import ccxt
import numpy as np
import pandas as pd
import streamlit as st

# ================== SETTINGS ==================
DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "LINK/USDT", "ADA/USDT"]
DEFAULT_TIMEFRAMES = ["15m", "1h", "4h"]
# ===============================================

st.set_page_config(page_title="Bitget → Excel (with Indicators)", page_icon="📊", layout="wide")
st.title("📊 Bitget → Excel (Fresh History + Indicators)")

# ----------------- Helpers ------------------
def to_naive_utc(dt_series: pd.Series) -> pd.Series:
    """Ensure timestamps are timezone-naive (Excel safe)."""
    s = pd.to_datetime(dt_series, errors="coerce")
    try:
        return s.dt.tz_localize(None)
    except Exception:
        return s

# CCXT setup (load markets once)
_EXCHANGE = ccxt.bitget({"enableRateLimit": True})
_EXCHANGE.load_markets()

CCXT_TF = {"15m": "15m", "1h": "1h", "4h": "4h"}

def resolve_symbol(s: str) -> str:
    """Normalize user input (BTCUSDT -> BTC/USDT) and resolve to Bitget market key."""
    s = s.upper().replace("-", "/")
    if "/" not in s and s.endswith("USDT"):
        s = s[:-4] + "/USDT"
    if s in _EXCHANGE.markets:
        return s
    for m in _EXCHANGE.markets.keys():
        if m.replace(":USDT", "") == s or m.replace("/", "") == s.replace("/", ""):
            return m
    raise ValueError(f"Symbol not found on Bitget: {s}")

def candles_needed(days: int, timeframe: str) -> int:
    mins = {"15m": 15, "1h": 60, "4h": 240}[timeframe]
    return int((days * 24 * 60) / mins) + 10  # small buffer

def get_bitget_klines(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """
    Fetch a window of OHLCV using ccxt, paginating until we have ~'days' worth.
    Returns ascending by time with columns: time_utc, open, high, low, close, volume
    """
    tf = CCXT_TF[timeframe]
    need = candles_needed(days, timeframe)
    out = []
    since_ms = int((pd.Timestamp.utcnow() - pd.Timedelta(days=days)).timestamp() * 1000)
    limit = 500

    while len(out) < need:
        try:
            batch = _EXCHANGE.fetch_ohlcv(symbol, timeframe=tf, since=since_ms, limit=limit)
        except Exception as e:
            st.error(f"Error fetching {symbol} {tf}: {e}")
            return pd.DataFrame()
        if not batch:
            break
        out.extend(batch)
        since_ms = batch[-1][0] + 1
        time.sleep(_EXCHANGE.rateLimit / 1000.0)
        if len(batch) < 5:  # no more data
            break

    if not out:
        return pd.DataFrame()

    df = pd.DataFrame(out, columns=["ts", "open", "high", "low", "close", "volume"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["time_utc"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_localize(None)
    df = df.sort_values("time_utc").reset_index(drop=True)
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add indicators: RSI(14), EMA(8/21), SMA(20/50), ATR(14), Bollinger(20,2), MACD(12,26,9),
    plus a few helper signal flags.
    """
    if df.empty:
        return df.copy()

    out = df.copy()

    # --- RSI(14) (simple Wilder-like approx with rolling mean) ---
    delta = out["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    out["RSI"] = 100 - (100 / (1 + rs))

    # --- EMAs & SMAs ---
    out["EMA8"] = out["close"].ewm(span=8, adjust=False).mean()
    out["EMA21"] = out["close"].ewm(span=21, adjust=False).mean()
    out["MA20"] = out["close"].rolling(20).mean()
    out["MA50"] = out["close"].rolling(50).mean()

    # --- High/Low 20 + distances ---
    out["High_20"] = out["high"].rolling(20).max()
    out["Low_20"] = out["low"].rolling(20).min()
    out["%_from_High20"] = (out["close"] - out["High_20"]) / out["High_20"] * 100
    out["%_from_Low20"] = (out["close"] - out["Low_20"]) / out["Low_20"] * 100

    # --- ATR(14) ---
    prev_close = out["close"].shift(1)
    tr1 = out["high"] - out["low"]
    tr2 = (out["high"] - prev_close).abs()
    tr3 = (out["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    out["ATR14"] = tr.ewm(alpha=1/14, adjust=False).mean()

    # --- Bollinger Bands (20, 2) ---
    std20 = out["close"].rolling(20).std()
    out["BB_mid"] = out["MA20"]
    out["BB_upper"] = out["MA20"] + 2 * std20
    out["BB_lower"] = out["MA20"] - 2 * std20
    # Percent-B / where price sits in bands
    out["%B"] = (out["close"] - out["BB_lower"]) / (out["BB_upper"] - out["BB_lower"])

    # --- MACD (12, 26, 9) ---
    ema12 = out["close"].ewm(span=12, adjust=False).mean()
    ema26 = out["close"].ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_hist"] = out["MACD"] - out["MACD_signal"]

    # --- Simple helper "signals" ---
    out["EMA8_above_EMA21"] = out["EMA8"] > out["EMA21"]
    out["RSI_zone"] = np.select(
        [out["RSI"] >= 70, out["RSI"] <= 30],
        ["overbought", "oversold"],
        default="neutral",
    )
    # MACD cross: True on the bar where cross happens
    macd_cross_up = (out["MACD"].shift(1) < out["MACD_signal"].shift(1)) & (out["MACD"] > out["MACD_signal"])
    macd_cross_dn = (out["MACD"].shift(1) > out["MACD_signal"].shift(1)) & (out["MACD"] < out["MACD_signal"])
    out["MACD_cross"] = np.where(macd_cross_up, "bull_cross", np.where(macd_cross_dn, "bear_cross", ""))

    return out

def build_excel(dfs: dict) -> bytes:
    """Build an Excel with one sheet per (symbol, timeframe), plus Close/EMA chart."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter", datetime_format="yyyy-mm-dd HH:MM") as writer:
        for (symbol, tf), df in dfs.items():
            df = df.copy()
            if "time_utc" in df.columns:
                df["time_utc"] = to_naive_utc(df["time_utc"])
            sheet = f"{symbol.replace('/', '')}_{tf}"
            df.to_excel(writer, sheet_name=sheet, index=False)

            # Build a chart with Close + EMA8 + EMA21
            workbook = writer.book
            ws = writer.sheets[sheet]

            def col_idx(col_name: str) -> int:
                return df.columns.get_loc(col_name)

            last_row = len(df) + 1
            x_col = col_idx("time_utc")
            close_col = col_idx("close")
            ema8_col = col_idx("EMA8")
            ema21_col = col_idx("EMA21")

            chart = workbook.add_chart({"type": "line"})
            chart.add_series({
                "name": "Close",
                "categories": [sheet, 1, x_col, last_row, x_col],
                "values": [sheet, 1, close_col, last_row, close_col],
            })
            if "EMA8" in df.columns:
                chart.add_series({
                    "name": "EMA8",
                    "categories": [sheet, 1, x_col, last_row, x_col],
                    "values": [sheet, 1, ema8_col, last_row, ema8_col],
                })
            if "EMA21" in df.columns:
                chart.add_series({
                    "name": "EMA21",
                    "categories": [sheet, 1, x_col, last_row, x_col],
                    "values": [sheet, 1, ema21_col, last_row, ema21_col],
                })
            chart.set_title({"name": f"{symbol} {tf} – Close & EMAs"})
            chart.set_legend({"position": "bottom"})
            ws.insert_chart("L2", chart)

    buf.seek(0)
    return buf

# ----------------- Sidebar controls ------------------
symbols = st.sidebar.multiselect("Select symbols", DEFAULT_SYMBOLS, DEFAULT_SYMBOLS)
# Auto-fix for missing slash format (BTCUSDT -> BTC/USDT)
symbols = [s if "/" in s else s.replace("USDT", "/USDT") for s in symbols]

timeframes = st.sidebar.multiselect("Select timeframes", DEFAULT_TIMEFRAMES, DEFAULT_TIMEFRAMES)

st.sidebar.markdown("### History window (days)")
lookback_15m = st.sidebar.number_input("15m lookback (days)", min_value=1, max_value=14, value=3)
lookback_1h  = st.sidebar.number_input("1h lookback (days)",  min_value=2, max_value=60, value=21)
lookback_4h  = st.sidebar.number_input("4h lookback (days)",  min_value=7, max_value=180, value=90)
LOOKBACK_BY_TF = {"15m": lookback_15m, "1h": lookback_1h, "4h": lookback_4h}

# ----------------- Main button ------------------
if st.button("🚀 Fetch & Build Excel"):
    all_dfs: dict[tuple[str, str], pd.DataFrame] = {}
    for sym in symbols:
        # Resolve to an exchange-supported market key
        try:
            sym_resolved = resolve_symbol(sym)
        except Exception as e:
            st.error(str(e))
            continue

        for tf in timeframes:
            st.write(f"Fetching **{sym_resolved} {tf}** …")
            df_raw = get_bitget_klines(sym_resolved, tf, days=LOOKBACK_BY_TF[tf])
            if df_raw.empty:
                st.warning(f"No data for {sym_resolved} {tf}")
                continue
            df = compute_indicators(df_raw)
            all_dfs[(sym_resolved, tf)] = df

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
