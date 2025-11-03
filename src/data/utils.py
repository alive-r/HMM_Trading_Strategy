# common tools
# dates, IO

from __future__ import annotations
import os, pathlib, datetime as dt
from typing import Optional
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
CACHE = ROOT / "data_cache"
CACHE.mkdir(exist_ok=True, parents=True)

def to_parquet(df:pd.DataFrame,name:str):
    path = CACHE / name
    df.to_parquet(path, index=True)
    print(f"[saved]{path} shape = {df.shape}")

def read_parquet(name:str):
    path = CACHE / name
    if path.exists():
        return pd.read_parquet(path)
    return None

def parse_date(s:str | None, default:str) -> dt.date:
    return dt.datetime.strptime(s or default, "%Y-%m-%d").date()

def trading_day_floor(d:dt.date) -> dt.date:
    # simplified, need to be matched with trading date
    return d
