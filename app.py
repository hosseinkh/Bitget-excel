# app.py

# app.py
import io
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
import ccxt

# ------------------ Config ------------------
DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "LINK/USDT", "ADA/USDT"]
DEFAULT_TFS     = ["15m", "1h", "4h"]
RSI_PERIOD      = 14
PER_CALL_LIMIT  = 1000  # ccxt per-call candle limit

# ------------------ Helpers ------------------
@st.cache_data(show_spinner=False)
def list_spot_usdt_symbols() -> List[str]:
    ex = ccxt.bitget()
    markets = ex.load_markets()
    return sorted([m for m, info in markets.items() if info.get("spot") and m.endswith("/USDT")])

def rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = up / dn
    return 100 - (100 / (1 + rs))

def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has a true DatetimeIndex that is timezone-naive, unique, and sorted.
    Works whether 'time' is ms since epoch or datetime.
    """
    if "time" in df.columns:
        # Accept ints/ms, strings, or datetimes
        t = pd.to_datetime(df["time"], errors="coerce", utc=True)
        # make timezone-naive for Excel later; index ops don't need tz-aware
        t = t.tz_convert(None)
        df = df.copy()
        df["time"] = t
        idx = pd.DatetimeIndex(t)
        df = df.set_index(idx)
    elif not isinstance(df.index, pd.DatetimeIndex):
        # Fallback if only index is present
        idx = pd.to_datetime(df.index, errors="coerce", utc=True).tz_convert(None)
        df = df.copy()
        df.index = pd.DatetimeIndex(idx)

    # Drop any NaT index rows, enforce monotonic increasing, unique
    df = df[~df.index.isna()]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df

def fetch_ohlcv_full(
    ex: ccxt.Exchange, symbol: str, timeframe: str, days: int, limit_per_call: int = PER_CALL_LIMIT
) -> pd.DataFrame:
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    since = int(start.timestamp() * 1000)

    chunks: List[pd.DataFrame] = []
    while True:
        candles = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit_per_call)
        if not candles:
            break
        cdf = pd.DataFrame(candles, columns=["time","open","high","low","close","volume"])
        chunks.append(cdf)
        last_ts = candles[-1][0]
        if last_ts <= since:
            break
        since = last_ts + 1
        if last_ts >= int(end.timestamp() * 1000):
            break

    if not chunks:
        return pd.DataFrame(columns=["time","open","high","low","close","volume"])

    df = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["time"])
    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)

    # Guarantee a proper DatetimeIndex
    df = _ensure_dt_index(df)
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_dt_index(df)
    out = df.copy()
    out["rsi"] = rsi(out["close"])
    out["pct_change"] = out["close"].pct_change() * 100.0
    # Flatten back to columns
    out = out.reset_index().rename(columns={"index": "time"})
    return out

def build_excel(sheets: Dict[Tuple[str, str], pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        for (symbol, tf), df in sheets.items():
            sheet = f"{symbol.replace('/','_')}_{tf}"
            dfx = df.copy()

            # Ensure timezone-naive 'time' column for Excel
            if "time" in dfx.columns:
                t = pd.to_datetime(dfx["time"], errors="coerce")
                # if any tz-aware sneaks in, strip it
                try:
                    if getattr(t.dt, "tz", None) is not None:
                        t = t.dt.tz_localize(None)
                except Exception:
                    pass
                dfx["time"] = t
            else:
                if isinstance(dfx.index, pd.DatetimeIndex):
                    dfx = dfx.reset_index().rename(columns={"index":"time"})

            dfx.to_excel(w, sheet_name=sheet, index=False)
    return buf.getvalue()

# ------------------ UI ------------------
st.set_page_config(page_title="Bitget → Excel (Full History + Indicators)", layout="wide")
st.title("📊 Bitget → Excel (Full History + Indicators)")
st.caption("Fetch Bitget OHLCV, compute RSI & % change, export one sheet per (symbol, timeframe).")

left, right = st.columns([1,3])

with left:
    st.subheader("Settings")
    all_syms = list_spot_usdt_symbols()
    symbols = st.multiselect(
        "Select symbols (type to add more, e.g., XRP/USDT)",
        options=all_syms,
        default=[s for s in DEFAULT_SYMBOLS if s in all_syms],
        placeholder="Start typing a pair…",
    )
    manual = st.text_input("Or add custom symbol (exact CCXT format)", "")
    if manual and manual not in symbols:
        symbols.append(manual)

    tfs = st.multiselect("Select timeframes", options=DEFAULT_TFS, default=DEFAULT_TFS)

    st.markdown("### History window (days)")
    lb_15m = st.number_input("15m lookback (days)", 1, 30, 3, 1)
    lb_1h  = st.number_input("1h lookback (days)",  1, 180, 21, 1)
    lb_4h  = st.number_input("4h lookback (days)",  1, 365, 90, 1)

with right:
    if st.button("🚀 Fetch & Build Excel", type="primary"):
        if not symbols or not tfs:
            st.warning("Pick at least one symbol and one timeframe.")
        else:
            ex = ccxt.bitget()
            lookback = {"15m": lb_15m, "1h": lb_1h, "4h": lb_4h}
            results: Dict[Tuple[str,str], pd.DataFrame] = {}

            total = len(symbols) * len(tfs)
            done = 0
            prog = st.progress(0.0)

            for sym in symbols:
                st.markdown(f"### **{sym}**")
                cols = st.columns(len(tfs))
                for i, tf in enumerate(tfs):
                    with cols[i]:
                        st.write(f"Fetching **{tf}** …")
                        try:
                            days = lookback.get(tf, 7)
                            raw = fetch_ohlcv_full(ex, sym, tf, days)
                            if raw.empty:
                                st.warning(f"No data for {sym} {tf}")
                            else:
                                # compute indicators without any resampling
                                df = add_indicators(raw)
                                results[(sym, tf)] = df
                                st.success(f"Fetched {len(df):,} rows")
                                st.dataframe(df.tail(5), use_container_width=True)
                        except Exception as e:
                            # Show exact message to identify source next time
                            st.error(f"{type(e).__name__}: {e}")
                        finally:
                            done += 1
                            prog.progress(done/total)

            if results:
                excel = build_excel(results)
                ts = datetime.now().strftime("%Y%m%d-%H%M")
                name = f"bitget_data_{ts}.xlsx"
                st.download_button("⬇️ Download Excel", data=excel, file_name=name,
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                st.info("Nothing to export yet. Fix errors above or adjust selections.")
