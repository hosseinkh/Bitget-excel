# app.py
import io
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, Tuple, List, Optional

import pandas as pd
import numpy as np
import streamlit as st
import ccxt

# ---------------------------
# UI defaults
# ---------------------------
DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "LINK/USDT", "ADA/USDT"]
DEFAULT_TFS = ["15m", "1h", "4h"]
RSI_PERIOD = 14
CCXT_LIMIT = 1000  # max candles per CCXT fetch

st.set_page_config(page_title="Bitget → Excel (Full History + Indicators)",
                   page_icon="📊", layout="wide")
st.title("📊 Bitget → Excel (Full History + Indicators)")
st.caption("Fetch historical candles from Bitget, compute RSI & % change, and download a single Excel with one sheet per (symbol, timeframe).")

# ---------------------------
# Helpers
# ---------------------------

def minutes_for_tf(tf: str) -> int:
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("h"):
        return int(tf[:-1]) * 60
    if tf.endswith("d"):
        return int(tf[:-1]) * 60 * 24
    raise ValueError(f"Unsupported timeframe: {tf}")

def since_ms_for_lookback(days: int) -> int:
    # UTC now minus N days
    return int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

def rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))

def detz_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all datetime are timezone-unaware (Excel requirement)."""
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    for col in df.columns:
        s = df[col]
        if hasattr(s.dtype, "tz") and getattr(s.dtype, "tz", None) is not None:
            df[col] = s.dt.tz_localize(None)
        elif pd.api.types.is_datetime64_any_dtype(s):
            df[col] = pd.to_datetime(s, errors="ignore")
    return df

def fetch_ohlcv_bitget(symbol: str, timeframe: str, since_ms: int, limit_per_call: int = CCXT_LIMIT) -> pd.DataFrame:
    """
    Robust OHLCV pull with pagination via CCXT.
    Returns DataFrame with columns: time, open, high, low, close, volume
    """
    ex = ccxt.bitget()
    ex.enableRateLimit = True

    all_rows: List[List[float]] = []
    cursor = since_ms
    max_iters = 1000  # safety

    for _ in range(max_iters):
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit_per_call)
        if not batch:
            break
        all_rows.extend(batch)
        # next since = last timestamp + 1ms to avoid duplicates
        cursor = int(batch[-1][0]) + 1
        # light throttle
        time.sleep(ex.rateLimit / 1000.0 if hasattr(ex, "rateLimit") else 0.2)
        # stop if we didn't move forward (shouldn't happen)
        if len(batch) < limit_per_call:
            break

    if not all_rows:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_rows, columns=["time", "open", "high", "low", "close", "volume"])
    # Convert ms → UTC naive datetime (Excel-safe)
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True).dt.tz_localize(None)
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pct_change"] = out["close"].pct_change() * 100.0
    out["rsi"] = rsi(out["close"], RSI_PERIOD)
    # nice ordering
    cols = ["time", "open", "high", "low", "close", "volume", "pct_change", "rsi"]
    return out[cols]

def build_excel(dfs: Dict[Tuple[str, str], pd.DataFrame]) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        for (sym, tf), df in dfs.items():
            if df.empty:
                continue
            df = detz_df(df)
            sheet = f"{sym.replace('/','')}_{tf}"
            df.to_excel(writer, sheet_name=sheet, index=False)
    return bio.getvalue()

def read_excel_to_dict(uploaded) -> Dict[Tuple[str, str], pd.DataFrame]:
    """Read existing Excel (if provided) back to per-(symbol,tf) dataframes."""
    if uploaded is None:
        return {}
    existing: Dict[Tuple[str, str], pd.DataFrame] = {}
    xls = pd.ExcelFile(uploaded)
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        # best-effort parse of name: e.g., BTCUSDT_1h
        if "_" in sheet:
            sym_part, tf_part = sheet.rsplit("_", 1)
            # restore symbol format BTCUSDT → BTC/USDT
            sym = f"{sym_part[:-4]}/{sym_part[-4:]}" if sym_part.endswith("USDT") else sym_part
            # ensure datetime parsing
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"], errors="coerce")
            existing[(sym, tf_part)] = df
    return existing

def append_and_dedupe(existing: Dict[Tuple[str, str], pd.DataFrame],
                      new: Dict[Tuple[str, str], pd.DataFrame]) -> Dict[Tuple[str, str], pd.DataFrame]:
    out: Dict[Tuple[str, str], pd.DataFrame] = {}
    keys = set(existing.keys()) | set(new.keys())
    for k in keys:
        e = existing.get(k, pd.DataFrame())
        n = new.get(k, pd.DataFrame())
        if e.empty:
            out[k] = n
        elif n.empty:
            out[k] = e
        else:
            df = pd.concat([e, n], ignore_index=True)
            if "time" in df.columns:
                df.drop_duplicates(subset=["time"], keep="last", inplace=True)
                df.sort_values("time", inplace=True)
            out[k] = df.reset_index(drop=True)
    return out

# ---------------------------
# Sidebar / Controls
# ---------------------------

with st.sidebar:
    st.subheader("Select symbols")
    # multiselect with typeahead; you can also type new ones like "XRP/USDT"
    symbols: List[str] = st.multiselect(
        "Symbols (type to add more, e.g. XRP/USDT)",
        options=sorted(DEFAULT_SYMBOLS),
        default=DEFAULT_SYMBOLS,
        help="You can type any Bitget spot pair like XRP/USDT, INJ/USDT, AVAX/USDT…",
    )
    manual_symbol = st.text_input("Add symbol manually (e.g., XRP/USDT)")
    if manual_symbol.strip():
        if manual_symbol.strip().upper() not in [s.upper() for s in symbols]:
            symbols.append(manual_symbol.strip().upper())

    st.subheader("Select timeframes")
    tfs: List[str] = st.multiselect(
        "Timeframes",
        options=DEFAULT_TFS,
        default=DEFAULT_TFS
    )

    st.subheader("History window (days)")
    lb_15m = st.number_input("15m lookback (days)", value=3, min_value=1, max_value=60, step=1)
    lb_1h  = st.number_input("1h lookback (days)",  value=21, min_value=1, max_value=180, step=1)
    lb_4h  = st.number_input("4h lookback (days)",  value=90, min_value=1, max_value=365, step=1)

    uploaded_file = st.file_uploader("Upload previous Excel to append (optional)", type=["xlsx"])

# Sanity: don’t run with empty selections
if not symbols or not tfs:
    st.info("Select at least one symbol and one timeframe to continue.")
    st.stop()

# map timeframe → lookback days
tf_lookback_days: Dict[str, int] = {}
for tf in tfs:
    if tf == "15m":
        tf_lookback_days[tf] = lb_15m
    elif tf == "1h":
        tf_lookback_days[tf] = lb_1h
    elif tf == "4h":
        tf_lookback_days[tf] = lb_4h
    else:
        # Fallback: default 7 days for any custom tf
        tf_lookback_days[tf] = 7

# ---------------------------
# Main button / Execution
# ---------------------------

if st.button("🚀 Fetch & Build Excel", type="primary"):
    st.write("Starting fetch…")
    results: Dict[Tuple[str, str], pd.DataFrame] = {}

    for sym in symbols:
        for tf in tfs:
            st.markdown(f"**Fetching `{sym}` {tf} …**")
            lookback_days = tf_lookback_days.get(tf, 7)
            since_ms = since_ms_for_lookback(lookback_days)
            try:
                raw = fetch_ohlcv_bitget(sym, tf, since_ms)
            except Exception as e:
                st.error(f"Error fetching {sym} {tf}: {e}")
                continue

            if raw.empty:
                st.warning(f"No data for {sym} {tf}")
                continue

            df = compute_indicators(raw)
            df = detz_df(df)  # guarantee Excel-safe
            results[(sym, tf)] = df

            with st.expander(f"Preview {sym} {tf}"):
                st.dataframe(df.tail(20), use_container_width=True)

    if not results:
        st.warning("No data fetched. Check symbols/timeframes and try again.")
        st.stop()

    # Append option
    existing = read_excel_to_dict(uploaded_file)
    if existing:
        st.info("Appending to previously uploaded Excel and de-duplicating on `time`…")
        results = append_and_dedupe(existing, results)

    # Build Excel
    st.write("Building Excel…")
    excel_bytes = build_excel(results)

    # Filename with timestamp
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = f"bitget_data_{ts}.xlsx"
    st.download_button("⬇️ Download Excel", data=excel_bytes, file_name=fname, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.success("Done! Tip: add this page to your Home Screen for one-tap access.")
