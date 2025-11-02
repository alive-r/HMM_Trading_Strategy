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
