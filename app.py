import io
from datetime import datetime, timezone
from typing import List, Dict, Tuple
import pandas as pd
import ccxt
import streamlit as st

DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "LINK/USDT", "ADA/USDT", "INJ/USDT"]
DEFAULT_TIMEFRAMES = ["15m", "1h"]
CANDLE_LIMIT = 500
RSI_PERIOD = 14

st.set_page_config(page_title="Bitget → Excel", page_icon="📊", layout="wide")
st.title("📊 Bitget → Excel (RSI, Volume, % Change)")
st.caption("Tap Start to fetch Bitget candles and download a single Excel with one sheet per (symbol, timeframe).")

@st.cache_data(show_spinner=False)
def list_spot_markets() -> List[str]:
    ex = ccxt.bitget()
    markets = ex.load_markets()
    return sorted([m for m, info in markets.items() if info.get("spot") and m.endswith("/USDT")])

def rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = up / dn
    return 100 - (100 / (1 + rs))

def tf_minutes(timeframe: str) -> int:
    if timeframe.endswith("m"): return int(timeframe[:-1])
    if timeframe.endswith("h"): return int(timeframe[:-1]) * 60
    if timeframe.endswith("d"): return int(timeframe[:-1]) * 60 * 24
    raise ValueError(f"Unsupported timeframe: {timeframe}")

def fetch_df(ex: ccxt.bitget, symbol: str, timeframe: str, limit: int = CANDLE_LIMIT) -> pd.DataFrame:
    data = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=["ts","open","high","low","close","volume"])
    df["time_utc"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df[["time_utc","open","high","low","close","volume"]]
    df["time_utc"] = df["time_utc"].dt.tz_localize(None)  # make naive (Excel-friendly)
    df["rsi14"] = rsi(df["close"])
    df["pct_change"] = df["close"].pct_change() * 100.0
    try:
        df["close_24h_ago"] = df["close"].shift(int(round(24 * (60 / tf_minutes(timeframe)))))
        df["change_24h_pct"] = (df["close"] / df["close_24h_ago"] - 1.0) * 100.0
    except Exception:
        df["change_24h_pct"] = pd.NA
    return df

def build_excel(dfs: Dict[Tuple[str,str], pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter", datetime_format="yyyy-mm-dd HH:MM") as w:
        wb = w.book
        for (symbol, timeframe), df in dfs.items():
            sheet = f"{symbol.replace('/','_')}__{timeframe}"
            df.to_excel(w, sheet_name=sheet, index=False)
            ws = w.sheets[sheet]
            last_row = len(df) + 1
            close_chart = wb.add_chart({"type": "line"})
            close_chart.add_series({
                "name": "Close",
                "categories": [sheet, 1, 0, last_row-1, 0],
                "values": [sheet, 1, 4, last_row-1, 4],
            })
            close_chart.set_title({"name": f"{symbol} Close ({timeframe})"})
            close_chart.set_x_axis({"name": "Time (UTC)"})
            close_chart.set_y_axis({"name": "Price"})

            rsi_chart = wb.add_chart({"type": "line"})
            rsi_chart.add_series({
                "name": "RSI14",
                "categories": [sheet, 1, 0, last_row-1, 0],
                "values": [sheet, 1, 6, last_row-1, 6],
            })
            rsi_chart.set_title({"name": f"{symbol} RSI14 ({timeframe})"})
            rsi_chart.set_y_axis({"name": "RSI"})

            ws.insert_chart("J2", close_chart)
            ws.insert_chart("J20", rsi_chart)
    buf.seek(0)
    return buf.read()

with st.sidebar:
    st.subheader("Settings")
    try:
        all_syms = list_spot_markets()
    except Exception:
        all_syms = DEFAULT_SYMBOLS
    symbols = st.multiselect("Symbols", options=all_syms, default=DEFAULT_SYMBOLS)
    tfs = st.multiselect("Timeframes", options=["1m","3m","5m","15m","30m","1h","4h","1d"], default=DEFAULT_TIMEFRAMES)

col1, col2 = st.columns(2)
with col1:
    start = st.button("🚀 Start & Generate Excel", type="primary")
with col2:
    st.write("Add this page to your Home Screen for one-tap access.")

if start:
    st.info("Fetching candles from Bitget…")
    ex = ccxt.bitget()
    dfs, errors = {}, []
    for s in symbols:
        for tf in tfs:
            try:
                dfs[(s, tf)] = fetch_df(ex, s, tf)
            except Exception as e:
                errors.append(f"{s} {tf}: {e}")
    if errors:
        with st.expander("Warnings"):
            st.write("\n".join(errors))
    if not dfs:
        st.error("No data fetched. Try fewer symbols/timeframes and retry.")
    else:
        excel_bytes = build_excel(dfs)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
        st.success("Done! Tap below to download.")
        st.download_button(
            label=f"📥 Download Excel ({ts})",
            data=excel_bytes,
            file_name=f"bitget_data_{ts}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.caption("Share this Excel with me—I’ll analyze entries, stops, and targets.")
