# CLI entry
# fetch and write into data_cache/*.parquet
from __future__ import annotations
import argparse, os, pandas as pd, numpy as np
from pathlib import Path
from .utils import to_parquet, read_parquet, parse_date
from .sources import yf_download_prices, fred_series, ken_french_daily_3f, sp500_constituents
from dotenv import load_dotenv
load_dotenv()

def main():
    parser = argparse.ArgumentParser("Data ingestion")
    parser.add_argument("--start", type = str, default="2005-01-01")
    parser.add_argument("--universe", type=str,default = "sp500")
    args = parser.parse_args()

    start = args.start

    # tickers, name, sector
    csv_path = Path(__file__).resolve().parent / "sp500.csv"
    uni = sp500_constituents(csv_path)
    to_parquet(uni, "universe.parquet")

    # prices based on given tickers and date
    tickers = uni["symbol"].tolist()
    prices = yf_download_prices(tickers, start)
    to_parquet(prices,"prices_daily.parquet")

    # fred data based on given date
    fred = fred_series(["VIXCLS","TB3MS"], start=start)
    fred = fred.asfreq("D").ffill()
    to_parquet(fred,"fred_macro.parquet")

    # ken french data based on given date
    ff=ken_french_daily_3f(start=start)
    to_parquet(ff,"ff_factors_daily.parquet")

    

    px_adj   = prices.xs("AdjClose", axis=1, level="Field"); px_adj.columns.name = None
    px_close = prices.xs("Close",    axis=1, level="Field"); px_close.columns.name = None
    to_parquet(px_adj,   "adjclose_wide.parquet")
    to_parquet(px_close, "close_wide.parquet")
    px = px_adj 
    # if "SPY" not in px.columns:
    #     spy = yf_download_prices(["SPY"], start=start)
    #     spy_adj = spy.xs("AdjClose", axis=1, level="Field")
    #     spy_adj.columns = [c for c in spy_adj.columns]
    #     px = px.join(spy_adj, how="outer")

    rets = px.pct_change(fill_method=None) # stickers daily returns

    mkt = rets["SPY"].rename("mkt_ret") # market daily return
    realized_vol_21 = rets["SPY"].rolling(21).std().rename("mkt_rvol_21") #rolling 21days SPmarket standdard deviation; will be used as a scale, high/low volatility
    breadth = (rets.iloc[:,:-1]
               .apply(lambda row:(row>0).sum()/row.count() if row.count()>0 else np.nan, axis=1)
               .rename("breadth_up_ratio")
               ) # participation info
    idx_feat = pd.concat([mkt, realized_vol_21, breadth], axis=1).join(fred["VIXCLS"]).ffill()
    to_parquet(idx_feat, "index_features.parquet")

if __name__ == "__main__":
    main()