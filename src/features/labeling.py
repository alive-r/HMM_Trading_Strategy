from __future__ import annotations
import pandas as pd, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data_cache"
PRICES = CACHE / "prices_daily.parquet"
OUT = CACHE / "labels_5d.parquet"

def main(horizon: int = 5):
    prices = pd.read_parquet(PRICES)
    close = prices.xs("Close", axis=1, level=1)
  
    mkt = close["SPY"].pct_change().rename("mkt_ret")
    fwd = close.pct_change(periods=horizon, fill_method=None).shift(-horizon)  # get return for future horizon time frame

    mkt_fwd = mkt.to_frame().reindex(fwd.index).iloc[:,0] # reindex with date
    exret = fwd.sub(mkt_fwd, axis=0) # get the diff of fwd and mkt_fwd

    y = (exret > 0).astype(float) # mark 1 if exret>0, else 0
    y = y.stack().to_frame("y").rename_axis(["date","symbol"])
    y.to_parquet(OUT)
    print("labelings are below")
    print(y[10000:10010])
    print(f"[saved] {OUT} shape={y.shape}")

if __name__ == "__main__":
    main()