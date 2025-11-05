from __future__ import annotations
import pandas as pd, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data_cache"
ALPHAS = CACHE / "alpha_probs.parquet"
REGIME = CACHE / "regime_proba.parquet"
OUT = CACHE / "weights_soft.parquet"

def map_to_weights(prob: pd.Series, long_q=0.8, short_q=0.2, gross_target=1.0) -> pd.Series:
    # long top 20%
    # short bottom 20%
    qs_long = prob.quantile(long_q)
    qs_short = prob.quantile(short_q)
    w = pd.Series(0.0, index=prob.index)
    w[prob >= qs_long] = 1.0
    w[prob <= qs_short] = -1.0
    g = w.abs().sum()
    if g > 0:
        w = w * (gross_target / g)
    return w

def main(panic_drag: float = 0.7):
    alphas = pd.read_parquet(ALPHAS)       # pA and pB, 
    regime = pd.read_parquet(REGIME)       # p_mr, p_tr, p_panic
    print(alphas[:10])
    print(regime[:10])
    start_date = max(alphas.index.get_level_values(0).min(),
                 regime.index.min())
    out_rows = []
    for d, df in alphas.groupby(level=0):
        if d<start_date:
            continue
        df = df.droplevel(0)
        wA = map_to_weights(df["pA"])
        wB = map_to_weights(df["pB"])
        if d in regime.index:
            p_mr = float(regime.loc[d, "p_mr"])
            p_tr = float(regime.loc[d, "p_tr"])
            p_pa = float(regime.loc[d, "p_panic"])
        else:
            p_mr = p_tr = 0.5; p_pa = 0.0

        w = p_mr * wA.add(0, fill_value=0) + p_tr * wB.add(0, fill_value=0)
        g = w.abs().sum()
        if g > 0:
            w = w / g
        w = w * (1 - panic_drag * p_pa)  # reduce weight if there is panic
        tmp = pd.DataFrame({"wA": wA, "wB": wB, "w": w})
        tmp["date"] = d
        out_rows.append(tmp.reset_index().set_index(["date","symbol"]))

    out = pd.concat(out_rows).sort_index()
    out.to_parquet(OUT)

    print("weights are below: ")
    print(out[10000:10010])
    print(f"[saved] {OUT} shape={out.shape}")

if __name__ == "__main__":
    main()