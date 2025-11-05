from __future__ import annotations
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.utils import to_parquet
from src.models.hmm import RegimeHMM, HMMConfig
import math

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data_cache"
FEATURE_FILE = CACHE / "index_features.parquet"
OUT_FILE = CACHE / "regime_proba.parquet"

def robust_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        q05, q95 = df[c].quantile(0.05), df[c].quantile(0.95)
        df[c] = df[c].clip(lower=q05, upper=q95)
    mu, sigma = df.mean(), df.std(ddof=0).replace(0, np.nan)
    df = (df - mu) / sigma
    df = df.fillna(0.0)
    return df

def prepare_features() -> pd.DataFrame:
    idx_feat = pd.read_parquet(FEATURE_FILE)
    cols = ["mkt_ret", "mkt_rvol_21", "VIXCLS", "breadth_up_ratio"]
    features = idx_feat[cols].copy()
    features = features.shift(1)
    features = features.dropna()
    features = robust_transform(features)
    return features

def main():
    parser = argparse.ArgumentParser("Build HMM regime probabilities with soft assignment")
    parser.add_argument("--states", type=int, default=3, help="HMM states: 2 or 3")
    parser.add_argument("--train_win", type=int, default=750, help="rolling train window (days)")
    parser.add_argument("--step", type=int, default=21, help="refit frequency (days)")
    parser.add_argument("--horizon", type=int, default=21, help="predict horizon after each refit")
    args = parser.parse_args()

    features = prepare_features()

    cfg = HMMConfig(n_states=args.states)
    model = RegimeHMM(cfg)
    res = model.walkforward_predict(
        features, train_window=args.train_win, step=args.step, predict_horizon=args.horizon
    )

    probs = res[[f"p_{k}" for k in range(args.states)]].values
    # higher entropy, higher volatility
    probs = np.clip(probs, 1e-10, 1.0)
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1) 
    res["entropy"] = entropy
    # dominant_state indicates the column index that means the dominant_state
    res["dominant_state"] = probs.argmax(axis=1)
    # confidence [0,1], 1 means pretty confident
    res["confidence"] = probs.max(axis=1)

    features_aligned = features.reindex(res.index).fillna(0)
    features_values = features_aligned[["mkt_ret","mkt_rvol_21","VIXCLS","breadth_up_ratio"]].values
    state_means = []
    for i in range(args.states):
        w = probs[:,i:i+1]
        mu_k = (w*features_values).sum(axis=0) / (w.sum() + 1e-12)
        state_means.append(mu_k)
    
    #find which index should be tagged as panic, mr, or tr
    state_means = np.vstack(state_means) # change the list to 2d array, states are the rows(means), features above are the columns
    vix_argmax = int(np.argmax(state_means[:,2])) # get the row index of the maximum value for column index 2
    ret_argmax = int(np.argmax(state_means[:,0]))
    all_states = set(range(args.states))
    mr_candidate = list(all_states-{vix_argmax,ret_argmax})
    mr_idx = mr_candidate[0]
    if vix_argmax == ret_argmax:
        panic_idx = vix_argmax
        remaining = list(all_states - {panic_idx})
        ret_values = state_means[remaining, 0]
        trend_idx = remaining[np.argmax(ret_values)]
        mr_idx = list(all_states - {panic_idx, trend_idx})[0]
    else:
        panic_idx = vix_argmax
        trend_idx = ret_argmax
        mr_idx = list(all_states - {vix_argmax, ret_argmax})[0]
    panic_idx = vix_argmax
    trend_idx = ret_argmax
    
    # rename columns
    rename_map = {
        f"p_{mr_idx}": "p_mr",
        f"p_{trend_idx}": "p_tr",
        f"p_{panic_idx}": "p_panic",
    }
    out = res.rename(columns=rename_map)
    all_cols = ["p_mr","p_tr","p_panic","entropy", "dominant_state", "confidence"]
    out = out[all_cols].copy()
    out.index.name = "date"
    to_parquet(out,OUT_FILE)
    
    print("model is converged: ", model.model.monitor_.converged)
    print("number of iterations:", model.model.monitor_.iter)
if __name__ == "__main__":
    main()