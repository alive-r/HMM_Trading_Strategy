from __future__ import annotations
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data_cache"
FEATS = CACHE / "features_equity.parquet"
LABELS = CACHE / "labels_5d.parquet"
OUT = CACHE / "alpha_probs.parquet"

# features
FEATS_CONTRARIAN = ["ret1","ret3","ret5","rsi5","volz20"]
FEATS_TREND = ["mom20","ma_diff","ret5","rsi14"]

def _fit_predict_block(X_tr, y_tr, X_te): # training data, labeling data, test data
    if len(np.unique(y_tr)) < 2:  # if there is only one labeling
        p = np.full(len(X_te), 0.5) 
        return p, p # return 0.5, it means doesn't contain the classification to learn
    clfA = LogisticRegression(max_iter=200) # model1, contrarian model
    clfB = LogisticRegression(max_iter=200) # model2, trend model

    XA = X_tr[FEATS_CONTRARIAN].values # get values of spefic features from training data as contrarian features value
    XB = X_tr[FEATS_TREND].values # get values of spefic features from training data as trending features value
    clfA.fit(XA, y_tr.values.ravel()) # fit features to label
    clfB.fit(XB, y_tr.values.ravel()) # fit features to label
    pA = clfA.predict_proba(X_te[FEATS_CONTRARIAN].values)[:,1] # use trained model to predict the probability of test data
    pB = clfB.predict_proba(X_te[FEATS_TREND].values)[:,1]
    return pA, pB

def main(train_win: int=252*2, step: int=21, horizon: int=21):
    feats = pd.read_parquet(FEATS) #(date, symbol)
    labels = pd.read_parquet(LABELS)
    df = feats.join(labels, how="inner")
    df = df.dropna()

    dates = sorted(df.index.get_level_values(0).unique())
    out_rows = []

    i = 0
    while i + train_win + 1 < len(dates):
        tr_start = dates[i]
        tr_end = dates[i + train_win - 1]
        te_start = dates[i + train_win]
        te_end_idx = min(i + train_win + horizon, len(dates) - 1)
        te_end = dates[te_end_idx]

        block_tr = df.loc[(slice(tr_start, tr_end), slice(None)), :]
        block_te = df.loc[(slice(te_start, te_end), slice(None)), :]

        mu = block_tr[feats.columns].mean()
        sd = block_tr[feats.columns].std(ddof=0).replace(0, np.nan)
        X_tr = (block_tr[feats.columns] - mu) / sd # standardlize block_tr, will be used to fit
        X_te = (block_te[feats.columns] - mu) / sd # use the same way to standardlize block_te, will be used to predict
        X_tr = X_tr.fillna(0); X_te = X_te.fillna(0)

        y_tr = block_tr[["y"]] # y is the labeling, 0 or 1, means return<mkt_return, or return>mkt_return
        pA, pB = _fit_predict_block(X_tr[FEATS_CONTRARIAN + FEATS_TREND], y_tr, X_te[FEATS_CONTRARIAN + FEATS_TREND])
        tmp = block_te[[]].copy()
        tmp["pA"] = pA
        tmp["pB"] = pB
        out_rows.append(tmp)

        i += step

    out = pd.concat(out_rows).sort_index()
    out.to_parquet(OUT)
    print(f"[saved] {OUT} shape={out.shape}")

if __name__ == "__main__":
    main()