# app.py
import io
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
import ccxt

# ---------- Config ----------
DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "LINK/USDT", "ADA/USDT"]
DEFAULT_TFS     = ["15m", "1h", "4h"]
RSI_PERIOD      = 14
PER_CALL_LIMIT  = 1000  # ccxt per-call candle limit

APP_VERSION = "v2.4 (no-index, no-tz, no-resample)"


# ---------- Indicators (index-free) ----------
def rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    d = series.diff()
    up = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = up / dn
    return 100 - (100 / (1 + rs))


# ---------- Data fetch (index-free) ----------
def fetch_ohlcv_full(
    ex: ccxt.Exchange, symbol: str, timeframe: str, days: int, limit_per_call: int = PER_CALL_LIMIT
) -> pd.DataFrame:
    """
    Fetch OHLCV and return a plain DataFrame with a normal 'time' column (tz-naive),
    no special index, sorted by time, duplicates removed.
    """
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

    df = pd.concat(chunks, ignore_index=True)
    # ensure numeric
    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"])

    # Convert ms → datetime, make tz-naive for Excel/consistency
    # We explicitly specify unit='ms' to avoid any inference surprises.
    t = pd.to_datetime(df["time"], unit="ms", utc=True).dt.tz_localize(None)
    df = df.assign(time=t).dropna(subset=["time"])

    # Remove duplicated bars (can happen at call boundaries), sort by time
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["rsi"] = rsi(out["close"])
    out["pct_change"] = out["close"].pct_change() * 100.0
    return out


def build_excel(sheets: Dict[Tuple[str, str], pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        for (symbol, tf), df in sheets.items():
            sheet = f"{symbol.replace('/','_')}_{tf}"
            dfx = df.copy()

            # Make absolutely sure 'time' is timezone-naive datetime
            if "time" in dfx.columns:
                t = pd.to_datetime(dfx["time"], errors="coerce")
                # Strip tz if any made it through
                try:
                    # t.dt.tz will exist on datetimetz; guard attribute access
                    if getattr(t.dt, "tz", None) is not None:
                        t = t.dt.tz_localize(None)
                except Exception:
                    pass
                dfx["time"] = t

            dfx.to_excel(
                w, sheet_name=sheet, index=False,
                freeze_panes=(1, 1)
            )
    return buf.getvalue()


# ---------- UI ----------
st.set_page_config(page_title="Bitget → Excel (Full History + Indicators)", layout="wide")
st.title("📊 Bitget → Excel (Full History + Indicators)")
st.caption(
    f"{APP_VERSION} · No DatetimeIndex anywhere. "
    "Fetch OHLCV from Bitget, compute RSI & % change, export one sheet per (symbol, timeframe)."
)

sidebar = st.sidebar
with sidebar:
    st.subheader("Settings")
    # Load available spot USDT pairs (cached)
    @st.cache_data(show_spinner=False)
    def list_spot_usdt_symbols() -> List[str]:
        ex = ccxt.bitget()
        markets = ex.load_markets()
        return sorted([m for m, info in markets.items() if info.get('spot') and m.endswith('/USDT')])

    all_syms = list_spot_usdt_symbols()
    symbols = st.multiselect(
        "Select symbols (type to add more, e.g., XRP/USDT)",
        options=all_syms,
        default=[s for s in DEFAULT_SYMBOLS if s in all_syms],
        placeholder="Start typing…",
    )
    manual = st.text_input("Or add custom symbol (exact CCXT format)", "")
    if manual and manual not in symbols:
        symbols.append(manual)

    tfs = st.multiselect("Select timeframes", options=DEFAULT_TFS, default=DEFAULT_TFS)

    st.markdown("### History window (days)")
    lb_15m = st.number_input("15m lookback (days)", 1, 30, 3, 1)
    lb_1h  = st.number_input("1h lookback (days)",  1, 180, 21, 1)
    lb_4h  = st.number_input("4h lookback (days)",  1, 365, 90, 1)

main = st.container()

with main:
    if st.button("🚀 Fetch & Build Excel", type="primary"):
        if not symbols or not tfs:
            st.warning("Pick at least one symbol and one timeframe.")
            st.stop()

        ex = ccxt.bitget()
        lookback = {"15m": lb_15m, "1h": lb_1h, "4h": lb_4h}
        results: Dict[Tuple[str, str], pd.DataFrame] = {}

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
                            df = add_indicators(raw)
                            results[(sym, tf)] = df
                            st.success(f"Fetched {len(df):,} rows")
                            st.dataframe(df.tail(5), use_container_width=True)
                    except Exception as e:
                        st.error(f"{type(e).__name__}: {e}")
                    finally:
                        done += 1
                        prog.progress(done / total)

        if results:
            excel = build_excel(results)
            ts = datetime.now().strftime("%Y%m%d-%H%M")
            name = f"bitget_data_{ts}.xlsx"
            st.download_button("⬇️ Download Excel", data=excel, file_name=name,
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.info("Nothing to export yet. Fix errors above or adjust selections.")
