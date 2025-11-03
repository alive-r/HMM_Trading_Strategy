# fetch functions for all data sources
from __future__ import annotations
import os, io, zipfile, datetime as dt
from typing import List, Dict
import pandas as pd
import pandas_datareader.data as pdr
from pandas_datareader import data,wb
import requests
from bs4 import BeautifulSoup
import datetime
import yfinance as yf

_YF_FIELD_ALIASES = {
    "Open": "Open",
    "High": "High",
    "Low": "Low",
    "Close": "Close",
    "Adj Close": "AdjClose",
    "Volume": "Volume",
}

# get stooq daily data
# return (sticker, Open / Low / High ...)
def yf_download_prices(tickers:list[str],start: str = "2025-01-01") -> pd.DataFrame:
    df = yf.download(
        tickers=tickers,
        start=start,
        auto_adjust=False,
        progress=False,
        group_by='ticker',
        threads=True
    )
    df = df.swaplevel(axis=1).sort_index(axis=1)
    new_cols = pd.MultiIndex.from_tuples(
            [(tk, _YF_FIELD_ALIASES.get(field, field)) for field, tk in df.columns],
            names=["Ticker", "Field"]
        )
    df.columns = new_cols
    return df.sort_index(axis=1)

# FRED Macro / risk
def fred_series(series_ids:List[str], start:str = "2005-01-01") -> pd.DataFrame:
    api_key = os.getenv("FRED_API_KEY")
    frames = []
    for sid in series_ids:
        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={sid}&api_key={api_key}&file_type=json&observation_start={start}"
        )
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        js = r.json()
        obs = js.get("observations",[])
        df = pd.DataFrame(obs)[["date", "value"]]
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.set_index("date").rename(columns={"value":sid})
        df.index = pd.to_datetime(df.index)
        frames.append(df)
    out = pd.concat(frames,axis=1).sort_index()
    return out

# Ken French Factor
def ken_french_daily_3f(start: str = "20050101") -> pd.DataFrame:
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
    r = requests.get(url,timeout=60)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    with z.open(z.namelist()[0]) as f:
        raw = f.read().decode("utf-8",errors="ignore")
    lines = raw.splitlines()

    header_idx = None
    for i, l in enumerate(lines):
        if l.strip().startswith(",Mkt-RF"):
            header_idx = i
            break

    data_text = "\n".join(lines[header_idx:])
    data = pd.read_csv(io.StringIO(data_text))
    first_col = data.columns[0]
    if first_col != "Date":
        data = data.rename(columns={first_col: "Date"})
    
    data = data[data["Date"].astype(str).str.match(r"^\d{8}$")]
    data["Date"] = pd.to_datetime(data["Date"], format="%Y%m%d")
    data = data.set_index("Date")
    data = data.loc[data.index >= pd.to_datetime(start)]
    data.columns = [c.strip().replace(" ","") for c in data.columns]
    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors="coerce")/100.0
    return data

# get SP500 tickers, security, sector
def sp500_constituents(csv_path:str) -> pd.DataFrame:
    df = pd.read_csv(csv_path,encoding="utf-8-sig")
    cols_map = {
        "Symbol":"symbol",
        "Security":"name",
        "GICS Sector":"sector",
    }
    df = df.rename(columns=cols_map)
    for c in ["symbol","name","sector"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    df["symbol"] = df["symbol"].str.replace(".","-", regex=False)
    df = df.dropna(subset=["symbol"]).drop_duplicates(subset=["symbol"])
    return df[["symbol", "name", "sector"]]



