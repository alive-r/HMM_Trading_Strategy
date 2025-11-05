import pandas as pd
w = pd.read_parquet("data_cache/weights_soft.parquet")
print(w.groupby(level=0)["w"].apply(lambda s: s.abs().sum()).describe())  # weights analysis
print(w.tail())

rp = pd.read_parquet("data_cache/regime_proba.parquet")
w_abs = w.groupby(level=0)["w"].apply(lambda s: s.abs().sum())
chk = pd.concat([w_abs, rp["p_panic"]], axis=1).dropna()
print(chk.corr()) # correlation of abs weight and panic 