# app.py
# Bitget → Excel (Full Candles + Indicators + Live Snapshot)
# v3.3 — index-free; correct tz handling with tz_localize(None); live snapshot on last row

from __future__ import annotations
import io
from typing import Dict, List, Tuple, Union

import pandas as pd
import numpy as np
import streamlit as st
import ccxt

# --------------------------
# Settings / defaults
# --------------------------
DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "LINK/USDT", "ADA/USDT"]
DEFAULT_TFS = ["15m", "1h", "4h"]
RSI_PERIOD = 14
MAX_LIMIT = 1500  # safety cap per request
APP_VERSION = "v3.3 (index-free + tz_localize fix + live snapshot)"

st.set_page_config(page_title="Bitget → Excel (Full Candles + Indicators)",
                   page_icon="📊", layout="wide")

st.title("📊 Bitget → Excel (Full Candles + Indicators + Live Snapshot)")
st.caption(f"{APP_VERSION} — Latest snapshot + historical candles with RSI & candle stats. Timestamps are UTC and timezone-naive (Excel-safe).")

# --------------------------
# Helpers
# --------------------------
@st.cache_resource(show_spinner=False)
def get_exchange():
    ex = ccxt.bitget({"enableRateLimit": True, "timeout": 20000})
    ex.load_markets()
    return ex

@st.cache_data(show_spinner=False)
def list_spot_usdt_symbols() -> List[str]:
    ex = get_exchange()
    return sorted([s for s, info in ex.markets.items() if info.get("spot") and s.endswith("/USDT")])

def tf_to_minutes(tf: str) -> int:
    tf = tf.strip().lower()
    if tf.endswith("m"): return int(tf[:-1])
    if tf.endswith("h"): return int(tf[:-1]) * 60
    if tf.endswith("d"): return int(tf[:-1]) * 60 * 24
    raise ValueError(f"Unsupported timeframe: {tf}")

def force_range_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index(drop=True)
    return df

# ---------- Time conversion (SAFE for Excel) ----------
def ms_to_naive_utc(ms: Union[pd.Series, int, float]) -> Union[pd.Series, pd.Timestamp]:
    """Convert milliseconds → UTC tz-aware → drop tz → tz-naive."""
    obj = pd.to_datetime(ms, unit="ms", utc=True)
    if isinstance(obj, pd.Series):
        return obj.dt.tz_localize(None)
    if isinstance(obj, pd.Timestamp):
        return obj.tz_localize(None)
    return obj  # fallback

def any_to_naive_utc(x) -> pd.Timestamp:
    """Accept ms or ISO string; return tz-naive UTC Timestamp."""
    if isinstance(x, (int, float)):
        return ms_to_naive_utc(x)  # returns Timestamp
    if isinstance(x, str):
        ts = pd.to_datetime(x, utc=True, errors="coerce")
        if isinstance(ts, pd.Timestamp):
            return ts.tz_localize(None)
    # fallback: current UTC as naive
    return pd.Timestamp.utcnow()

# ---------- Indicators (index-free) ----------
def rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    series = series.reset_index(drop=True)  # ensure RangeIndex
    d = series.diff()
    up = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = up / dn
    return 100 - (100 / (1 + rs))

# ---------- Data fetch (index-free) ----------
def fetch_ohlcv(ex: ccxt.Exchange, symbol: str, timeframe: str,
                lookback_days: int) -> pd.DataFrame:
    """Return candles with tz-naive 'timestamp' column and RangeIndex."""
    minutes = tf_to_minutes(timeframe)
    candles_needed = int((lookback_days * 24 * 60) / minutes) + 5
    candles_needed = max(50, min(candles_needed, MAX_LIMIT))

    now_utc = pd.Timestamp.utcnow()
    since = int((now_utc - pd.Timedelta(days=lookback_days)).timestamp() * 1000)

    raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=candles_needed)
    if not raw:
        cols = ["timestamp","open","high","low","close","volume"]
        return pd.DataFrame(columns=cols).pipe(force_range_index)

    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = ms_to_naive_utc(df["timestamp"])  # tz-naive
    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = (df
          .drop_duplicates(subset=["timestamp"])
          .dropna(subset=["close"])
          .sort_values("timestamp")
          .reset_index(drop=True))
    return force_range_index(df)

def fetch_latest_details(ex: ccxt.Exchange, symbol: str) -> dict:
    """Live snapshot: last, bid, ask, spread, and snapshot timestamp (UTC-naive)."""
    t = ex.fetch_ticker(symbol)
    last = t.get("last") or t.get("close") or t.get("bid") or t.get("ask")
    bid = t.get("bid")
    ask = t.get("ask")
    ts_val = t.get("timestamp") or t.get("datetime")
    ts = any_to_naive_utc(ts_val)

    spread = np.nan
    if isinstance(bid, (int, float)) and isinstance(ask, (int, float)):
        spread = float(ask) - float(bid)

    return {
        "current_price": float(last) if last is not None else np.nan,
        "current_bid": float(bid) if bid is not None else np.nan,
        "current_ask": float(ask) if ask is not None else np.nan,
        "current_spread": float(spread) if not pd.isna(spread) else np.nan,
        "current_ts": ts
    }

def merge_latest_snapshot(df: pd.DataFrame, snap: dict) -> pd.DataFrame:
    """Attach live snapshot on the last row; keep RangeIndex."""
    df = force_range_index(df)
    if df.empty:
        return df
    for c in ["current_price","current_bid","current_ask","current_spread","current_ts"]:
        if c not in df.columns:
            df[c] = np.nan
    for k, v in snap.items():
        df.loc[df.index[-1], k] = v
    return force_range_index(df)

def build_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add candle stats + RSI. Fully index-free."""
    df = df.reset_index(drop=True)
    out = df.copy()

    out["candle_color"] = np.where(out["close"] >= out["open"], "green", "red")
    out["body"] = out["close"] - out["open"]
    out["range"] = out["high"] - out["low"]
    out["upper_wick"] = out["high"] - out[["open", "close"]].max(axis=1)
    out["lower_wick"] = out[["open", "close"]].min(axis=1) - out["low"]

    out["pct_change_%"] = out["close"].pct_change() * 100
    out["rsi14"] = rsi(out["close"], RSI_PERIOD)

    cols = [
        "timestamp","open","high","low","close","volume",
        "candle_color","body","range","upper_wick","lower_wick",
        "pct_change_%","rsi14",
        "current_price","current_bid","current_ask","current_spread","current_ts"
    ]
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[cols].reset_index(drop=True)

def build_excel(dfs: Dict[Tuple[str, str], pd.DataFrame]) -> bytes:
    """One sheet per (symbol, timeframe)."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter", datetime_format="yyyy-mm-dd hh:mm:ss") as writer:
        for (sym, tf), df in dfs.items():
            sheet = f"{sym.replace('/','')}_{tf}"
            dfx = df.reset_index(drop=True)
            dfx.to_excel(writer, sheet_name=sheet, index=False)
            ws = writer.sheets[sheet]
            ws.freeze_panes(1, 1)
            ws.autofilter(0, 0, len(dfx), len(dfx.columns)-1)
            for col_idx, col in enumerate(dfx.columns):
                width = min(24, max(12, int(dfx[col].astype(str).str.len().clip(upper=24).mean())))
                ws.set_column(col_idx, col_idx, width)
    return output.getvalue()

# --------------------------
# UI
# --------------------------
ex = get_exchange()

left, right = st.columns([1, 2])

with left:
    st.subheader("Settings")

    all_symbols = list_spot_usdt_symbols()
    symbols = st.multiselect(
        "Select symbols (type to add more, e.g., XRP/USDT)",
        options=all_symbols,
        default=[s for s in DEFAULT_SYMBOLS if s in all_symbols],
        help="All Bitget spot /USDT pairs are listed. You can also add symbols manually."
    )
    manual_symbol = st.text_input("Or add custom symbol (exact CCXT format, e.g., XRP/USDT)", "")
    if manual_symbol.strip():
        ms = manual_symbol.strip().upper()
        if ms not in symbols:
            symbols.append(ms)

    timeframes = st.multiselect(
        "Select timeframes",
        options=DEFAULT_TFS,
        default=DEFAULT_TFS,
    )

    st.subheader("History window (days)")
    look15 = st.number_input("15m lookback (days)", min_value=1, max_value=30, value=3, step=1)
    look1h = st.number_input("1h lookback (days)", min_value=1, max_value=180, value=21, step=1)
    look4h = st.number_input("4h lookback (days)", min_value=1, max_value=365, value=90, step=1)

    tf_look_map = {"15m": look15, "1h": look1h, "4h": look4h}

    go = st.button("🚀 Fetch & Build Excel", type="primary", use_container_width=True)

with right:
    if go:
        if not symbols:
            st.warning("Add at least one symbol.")
        elif not timeframes:
            st.warning("Select at least one timeframe.")
        else:
            results: Dict[Tuple[str, str], pd.DataFrame] = {}
            total = len(symbols) * len(timeframes)
            done = 0
            prog = st.progress(0.0)

            for sym in symbols:
                st.markdown(f"### {sym}")
                cols = st.columns(max(1, len(timeframes)))
                for i, tf in enumerate(timeframes):
                    with cols[i]:
                        st.write(f"Fetching **{tf}** …")
                        try:
                            lb_days = tf_look_map.get(tf, 3)
                            df = fetch_ohlcv(ex, sym, tf, lb_days)
                            if df.empty:
                                st.error("No data returned.")
                                done += 1; prog.progress(done/total); continue

                            snap = fetch_latest_details(ex, sym)
                            df = merge_latest_snapshot(df, snap)
                            df = build_candle_features(df)  # index-free
                            df = df.reset_index(drop=True)   # final guard

                            results[(sym, tf)] = df
                            st.success(f"{len(df):,} rows")
                            st.dataframe(df.tail(6), use_container_width=True)

                        except Exception as e:
                            st.error(f"{type(e).__name__}: {e}")
                        finally:
                            done += 1
                            prog.progress(done / total)

            if results:
                excel_bytes = build_excel(results)
                fname = f"bitget_full_{pd.Timestamp.utcnow().strftime('%Y%m%d-%H%M%S')}.xlsx"
                st.download_button(
                    "⬇️ Download Excel",
                    data=excel_bytes,
                    file_name=fname,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
                st.caption("Sheets show closed candles; the **last row** also includes a live snapshot (`current_*` fields).")
