# app.py

import io
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
import ccxt


# -------------------------
# Config / constants
# -------------------------
DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "LINK/USDT", "ADA/USDT"]
DEFAULT_TFS = ["15m", "1h", "4h"]
RSI_PERIOD = 14
PER_CALL_LIMIT = 1000  # ccxt fetch_ohlcv limit per call


# -------------------------
# Small helpers
# -------------------------
@st.cache_data(show_spinner=False)
def list_spot_usdt_symbols() -> List[str]:
    """List USDT spot symbols available on Bitget via CCXT."""
    ex = ccxt.bitget()
    markets = ex.load_markets()
    syms = sorted(
        [
            m
            for m, info in markets.items()
            if info.get("spot") and m.endswith("/USDT")
        ]
    )
    return syms


def rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = up / dn
    return 100 - (100 / (1 + rs))


def fetch_ohlcv_full(
    ex: ccxt.Exchange, symbol: str, timeframe: str, days: int, limit_per_call: int = PER_CALL_LIMIT
) -> pd.DataFrame:
    """Fetch historical candles for the requested window."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    since = int(start.timestamp() * 1000)

    frames: List[pd.DataFrame] = []
    while True:
        candles = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit_per_call)
        if not candles:
            break

        chunk = pd.DataFrame(candles, columns=["time", "open", "high", "low", "close", "volume"])
        frames.append(chunk)

        last_ts = candles[-1][0]
        if last_ts <= since:
            break
        since = last_ts + 1
        if last_ts >= int(end.timestamp() * 1000):
            break

    if not frames:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])

    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["time"])

    # Convert to timezone-naive datetime for Excel safety
    dt_utc = pd.to_datetime(df["time"], unit="ms", utc=True)
    df["time"] = dt_utc.tz_convert(None)

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["close"]).sort_values("time").reset_index(drop=True)

    # Set a DatetimeIndex during computations (fix for pandas time ops)
    df = df.set_index(pd.DatetimeIndex(df["time"]))
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute indicators while a DatetimeIndex is set, then return a flat table."""
    # Ensure we have a DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index(pd.DatetimeIndex(df["time"]))

    out = df.copy()
    out["rsi"] = rsi(out["close"])
    out["pct_change"] = out["close"].pct_change() * 100.0

    # Back to column layout for Excel
    out = out.reset_index(drop=False).rename(columns={"index": "time"})
    return out


def build_excel(sheets: Dict[Tuple[str, str], pd.DataFrame]) -> bytes:
    """Create an Excel file with one sheet per (symbol, timeframe)."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        for (symbol, tf), df in sheets.items():
            sheet_name = f"{symbol.replace('/','_')}_{tf}"
            # Ensure 'time' is timezone-naive and present as a column
            if "time" in df.columns:
                df = df.copy()
                df["time"] = pd.to_datetime(df["time"])
                if getattr(df["time"].dt, "tz", None) is not None:
                    df["time"] = df["time"].dt.tz_localize(None)
            else:
                # if index is datetime, expose it as column
                if isinstance(df.index, pd.DatetimeIndex):
                    df = df.reset_index().rename(columns={"index": "time"})

            df.to_excel(writer, sheet_name=sheet_name, index=False)

    return buffer.getvalue()


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Bitget → Excel (Full History + Indicators)", layout="wide")
st.title("📊 Bitget → Excel (Full History + Indicators)")
st.caption("Fetch historical candles from Bitget, compute RSI & % change, and export — one sheet per (symbol, timeframe).")

colL, colR = st.columns([1, 3])

with colL:
    st.subheader("Settings")

    all_symbols = list_spot_usdt_symbols()
    selected = st.multiselect(
        "Select symbols (type to add more, e.g., XRP/USDT)",
        options=all_symbols,
        default=[s for s in DEFAULT_SYMBOLS if s in all_symbols],
        placeholder="Start typing a symbol…",
    )

    # Allow manual entry of a custom pair, if not listed
    manual = st.text_input("Or add custom symbol (exact CCXT format, e.g., XRP/USDT)", "")
    if manual and manual not in selected:
        selected.append(manual)

    tfs = st.multiselect(
        "Select timeframes",
        options=DEFAULT_TFS,
        default=DEFAULT_TFS,
    )

    st.markdown("### History window (days)")
    lb_15m = st.number_input("15m lookback (days)", min_value=1, max_value=30, value=3, step=1)
    lb_1h  = st.number_input("1h lookback (days)",  min_value=1, max_value=120, value=21, step=1)
    lb_4h  = st.number_input("4h lookback (days)",  min_value=1, max_value=365, value=90, step=1)

with colR:
    if st.button("🚀 Fetch & Build Excel", type="primary"):
        if not selected or not tfs:
            st.warning("Pick at least one symbol and one timeframe.")
        else:
            ex = ccxt.bitget()
            lookback_by_tf = {"15m": lb_15m, "1h": lb_1h, "4h": lb_4h}

            results: Dict[Tuple[str, str], pd.DataFrame] = {}
            prog = st.progress(0.0)
            total = len(selected) * len(tfs)
            done = 0

            for sym in selected:
                st.markdown(f"### **{sym}**")
                cols = st.columns(len(tfs))
                for i, tf in enumerate(tfs):
                    with cols[i]:
                        st.write(f"Fetching **{tf}** …")
                        try:
                            days = lookback_by_tf.get(tf, 7)
                            raw = fetch_ohlcv_full(ex, sym, tf, days=days)
                            if raw.empty:
                                st.warning(f"No data for {sym} {tf}")
                            else:
                                df = add_indicators(raw)
                                results[(sym, tf)] = df
                                st.success(f"Fetched {len(df):,} rows")
                                st.dataframe(df.tail(5), use_container_width=True)
                        except Exception as e:
                            st.error(f"Error fetching {sym} {tf}: {e}")
                        done += 1
                        prog.progress(done / total)

            if results:
                excel_bytes = build_excel(results)
                ts = datetime.now().strftime("%Y%m%d-%H%M")
                fname = f"bitget_data_{ts}.xlsx"
                st.download_button(
                    "⬇️ Download Excel",
                    data=excel_bytes,
                    file_name=fname,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            else:
                st.info("Nothing to export yet — fix errors above or adjust selections.")
