# Regime Switching Dual Style Daily Equity Strategy 

## Brief Intro

- This is a regime switching two style portfolio. 
- When an HMM indicates a mean reverting market, it activates Contrarian (long recent losers with decent fundamentals; short recent winners with weak quality). 
- When the regime is Trending, it activates Trend following (long strength, short weakness). 
- In Panic, it cut exposure or pause. Stock selection uses a supervised gradient boosted classifier for probability ranking
- Exits rely on ATR/time stops with an optional tiny RL policy (hold/halve/exit) for finer de risking. 

## Setup
```
git clone https://gitlab.oit.duke.edu/hy218/regime-switching-dual-style-daily-equity-strategy.git
```

## Command
#### get the most recent data by running command below
- start time is subjected to change, but the window should be enough for training window, step and horizon window mentioned below.
``` 
python -m src.data.ingest --start 2005-01-01 --universe sp500
```
#### run the hmm model
- States are set to 3. They are mkt_rvol_21, VIX, and breadth_up_ratio
- train_win means the time window for training data
- step means the time window for re-training
- horizon means the time window for forecasting 
- usually, horizon should be almost the same as step
```
 python -m src.pipelines.build_regime --states 3 --train_win 750 --step 10 --horizon 10
```
