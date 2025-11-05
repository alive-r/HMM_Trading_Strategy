from __future__ import annotations
import pandas as pd, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data_cache"
PRICES = CACHE / "prices_daily.parquet"
UNIVERSE = CACHE / "universe.parquet"
OUT = CACHE / "features_equity.parquet"

def rsi(close: pd.Series, n: int = 5) -> pd.Series:
    delta = close.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=close.index).rolling(n).mean()
    roll_down = pd.Series(down, index=close.index).rolling(n).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

def zscore(s: pd.Series, n: int=20) -> pd.Series:
    m = s.rolling(n).mean()
    sd = s.rolling(n).std(ddof=0)
    return (s - m) / (sd + 1e-12)

def main():
    prices = pd.read_parquet(PRICES)  #multiIndex columns: (ticker, OHLCV)
    close = prices.xs("Close", axis=1, level=1)
    vol = prices.xs("Volume", axis=1, level=1)

    rets1 = close.pct_change(fill_method=None)
    rets3 = close.pct_change(3, fill_method=None)
    rets5 = close.pct_change(5,fill_method=None)
    mom20 = close.pct_change(20,fill_method=None)
    rsi5 = close.apply(rsi, n=5)
    rsi14 = close.apply(rsi, n=14)

    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma_diff = (ma10 - ma20) / (ma20 + 1e-12)

    vol_z20 = vol.apply(lambda s: zscore(np.log1p(s), 20))

    def melt(name, df):
        m = df.stack().to_frame(name)
        m.index.set_names(["date","symbol"], inplace=True)
        return m

    feats = [
        melt("ret1", rets1),
        melt("ret3", rets3),
        melt("ret5", rets5),
        melt("mom20", mom20),
        melt("rsi5", rsi5),
        melt("rsi14", rsi14),
        melt("ma_diff", ma_diff),
        melt("volz20", vol_z20),
    ]
    out = pd.concat(feats, axis=1).dropna(thresh=4)
    print("features are below")
    print(out[10000:10010])
    out.to_parquet(OUT)
    print(f"[saved] {OUT} shape={out.shape}")

if __name__ == "__main__":
    main()