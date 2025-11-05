from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

@dataclass
class HMMConfig:
    n_states: int = 3
    covariance_type: str = "full" # full or diag for high dimension
    n_iter: int = 200 #the maximum number of iterations; model.monitor_.converged to check the convergence
    tol: float = 1e-5 # tolerance, small-concise-slow or large-fast; when the difference of two iterations is less than tol, then it means we find convergence
    random_state: int = 400 # randomly initialize matrix and parameters; a random integer

class RegimeHMM:
    def __init__(self,hmmConfig:HMMConfig):
        self.hmmConfig = hmmConfig
        self.model: Optional[GaussianHMM] = None
        self.scaler: Optional[StandardScaler] = None
        self.features: List[str] = []

    def _fit_window(self, data:pd.DataFrame):
        self.scaler = StandardScaler()
        dataScalered = self.scaler.fit_transform(data.values)
        self.model = GaussianHMM(
            n_components=self.hmmConfig.n_states,
            covariance_type=self.hmmConfig.covariance_type,
            n_iter=self.hmmConfig.n_iter,
            tol=self.hmmConfig.tol,
            random_state=self.hmmConfig.random_state
        )
        self.model.fit(dataScalered)
    
    def _posteriors(self, data: pd.DataFrame):
        assert self.model is not None and self.scaler is not None
        dataScalered = self.scaler.transform(data.values)
        return self.model.predict_proba(dataScalered)

    # step: train again by step
    # predict the range with predict_horizon
    def walkforward_predict(self, data: pd.DataFrame, train_window: 700, step: 21, predict_horizon:21):
        dates = data.index
        n = len(data)
        out = []

        for i in range(0, n-train_window-1, step):
            trainedStart = i #trained range start index
            trainedEnd = i+train_window #trained range end index; not included in the train range
            predictedStart = trainedEnd #predicted range start index
            predictedEnd = min(trainedEnd+predict_horizon,n) #predicted range end index
            data_trained = data.iloc[trainedStart:trainedEnd]
            data_predicted = data.iloc[predictedStart:predictedEnd]
            if len(data_predicted) == 0:
                break
            self._fit_window(data_trained)
            probs_posteriors = self._posteriors(data_predicted)
            states_posteriors = probs_posteriors.argmax(axis=1)
            temp = pd.DataFrame(probs_posteriors, index=dates[predictedStart:predictedEnd], columns = [ f"p_{k}" for k in range(self.hmmConfig.n_states)])
            temp["state"] = states_posteriors
            out.append(temp)
        res = pd.concat(out).sort_index()
        
        return res
            

