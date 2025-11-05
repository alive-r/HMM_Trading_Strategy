# HMM Trading Strategy

## Brief Intro

- This is a regime switching portfolio. 
- When an Hiden Markov Model indicates a mean reverting market, it activates Contrarian (long recent losers with decent fundamentals; short recent winners with weak quality). 
- When the regime is Trending, it activates Trend following (long strength, short weakness). 
- In Panic, it cut exposure or pause. Stock selection uses a supervised gradient boosted classifier for probability ranking
- Exits rely on ATR/time stops with an optional tiny RL policy (hold/halve/exit) for finer de risking. 

## Setup
```
git clone https://gitlab.oit.duke.edu/hy218/regime-switching-dual-style-daily-equity-strategy.git
```

## Command
Run commands in root folder
#### get the most recent data by running command below
- start time is subjected to change, but the window should be enough for training window, step and horizon window mentioned below.
 (it may take time to get full data; it may fail due to distability of yfinance, try it again )
``` 
python -m src.data.ingest --start 2005-01-01 --universe sp500
```
#### run the hmm model
- states are set to 3, they are mkt_rvol_21, VIX, and breadth_up_ratio
- train_win means the time window for training data
- step means the time window for re-training
- horizon means the time window for forecasting 
- usually, horizon should be almost the same as step
```
 python -m src.pipelines.build_regime --states 3 --train_win 750 --step 10 --horizon 10
```

#### generate features
```
python -m src.features.make_features
```

#### generate labeling
```
python -m src.features.labeling
```

#### train contrarian and trend models to get probabilities for the two
```
python -m src.pipelines.train_alphas --train_win 504 --step 21 --horizon 21
```

#### generate weights using regime in hmm and probability from last step
```
python -m src.pipelines.build_weights
```