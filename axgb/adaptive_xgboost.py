import numpy as np 
import pandas as pd

from river.drift import ADWIN 
from river.base import MiniBatchClassifier

import xgboost as xgb 

class AdaptiveXGBoostClassifier(MiniBatchClassifier):
    _PUSH_STRATEGY = 'push'
    _REPLACE_STRATEGY = 'replace'
    _UPDATE_STRATEGIES = [_PUSH_STRATEGY, _REPLACE_STRATEGY]

    def __init__(self, 
                 n_estimators = 30, 
                 learning_rate = 0.3,
                 max_depth = 6, 
                 max_window_size = 1000,
                 min_window_size = None,
                 detect_drift = False,
                 update_strategy = 'replace'):
        super().__init__()

        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_window_size = max_window_size
        self.min_window_size = min_window_size

        self._first_run = True
        self._ensemble = None
        self.detect_drift = detect_drift
        self._drift_detector = None
        self._X_buffer = np.array([])
        self._y_buffer = np.array([])
        self._samples_seen = 0
        self._model_idx = 0

        if update_strategy not in self._UPDATE_STRATEGIES:
            raise AttributeError("Invalid update_strategy: {}\n"
                                 "Valid options: {}".format(update_strategy,
                                                            self._UPDATE_STRATEGIES))
        self.update_strategy = update_strategy
        self._configure()
    
    def _adjust_window_size(self):
        if self._dynamic_window_size < self.max_window_size:
            self._dynamic_window_size *= 2
            if self._dynamic_window_size > self.max_window_size:
                self.window_size = self.max_window_size
            else:
                self.window_size = self._dynamic_window_size

    def _reset_window_size(self):
        if self.min_window_size:
            self._dynamic_window_size = self.min_window_size
        else:
            self._dynamic_window_size = self.max_window_size
        self.window_size = self._dynamic_window_size


    def _configure(self):
        if self.update_strategy == self._PUSH_STRATEGY:
            self._ensemble = []
        elif self.update_strategy == self._REPLACE_STRATEGY:
            self._ensemble = [None] * self.n_estimators
        self._reset_window_size()
        self._init_margin = 0.0
        self._boosting_params = {"silent": True,
                                 "objective": "binary:logistic",
                                 "eta": self.learning_rate,
                                 "max_depth": self.max_depth}
        if self.detect_drift:
            self._drift_detector = ADWIN()

    def _train_on_minibatch(self, X : np.ndarray, y: np.ndarray):
        #print(X)
        #print(y)

        if self.update_strategy == self._REPLACE_STRATEGY: 
            booster = self._train_booster(X, y, self._model_idx)

            self._ensemble[self._model_idx] = booster 
            self._samples_seen += X.shape[0]
            self._update_model_idx() 
        else:
            booster = self._train_booster(X, y, len(self._ensemble))

            if len(self._ensemble) == self.n_estimators:
                self._ensemble.pop()
            self._ensemble.append(booster)
            self._samples_seen += X.shape[0]
        pass

    def _train_booster(self, X: np.ndarray, y: np.ndarray, last_model_idx : int):
        d_mini_batch_train = xgb.DMatrix(X, y.astype(int))

        margins = np.asarray([self._init_margin] * d_mini_batch_train.num_row())
        for j in range(last_model_idx):
            margins = np.add(margins,
                             self._ensemble[j].predict(d_mini_batch_train, output_margin=True))
        d_mini_batch_train.set_base_margin(margin=margins)
        booster = xgb.train(params = self._boosting_params, 
                            dtrain=d_mini_batch_train, 
                            num_boost_round = 5,
                            verbose_eval = False)
        return booster

    def _update_model_idx(self):
        self._model_idx += 1
        if self._model_idx == self.n_estimators:
            self._model_idx = 0

    def _learn(self, X: np.ndarray, y: np.ndarray):
        if self._first_run: 
            self._X_buffer = np.empty((0, X.shape[0])) #np.array([]).reshape(0, X.shape[0])
            self._y_buffer = np.empty((0,1))
            self._first_run = False
        
        X = X.reshape(1, -1)
        y = y.reshape(1, -1)

        self._X_buffer = np.concatenate([self._X_buffer, X])
        self._y_buffer = np.concatenate([self._y_buffer, y])

        while self._X_buffer.shape[0] >= self.window_size: 
            self._train_on_minibatch(X = self._X_buffer[0: self.window_size, :],
                                     y = self._y_buffer[0:self.window_size])

            delete_idx = [i for i in range(self.window_size)]
            self._X_buffer = np.delete(self._X_buffer, delete_idx, axis=0)
            self._y_buffer = np.delete(self._y_buffer, delete_idx, axis=0)

            self._adjust_window_size()

        # Concept drifting
        
        if self.detect_drift:
            correctly_classifies = self.predict_many(X) == y 

            self._drift_detector.update(int(not correctly_classifies))

            if self._drift_detector.drift_detected:
                self._reset_window_size()
                if self.update_strategy == self._REPLACE_STRATEGY:
                    self._model_idx = 0

    def _predict(self, X: np.ndarray) -> np.ndarray:
        if self._ensemble: 
            if self.update_strategy == self._REPLACE_STRATEGY:
                trees_in_ensemble = sum(i is not None for i in self._ensemble)
            else: 
                trees_in_ensemble = len(self._ensemble)
            
            if trees_in_ensemble > 0: 
                d_test = xgb.DMatrix(X)

                for i in range(trees_in_ensemble - 1): 
                    margins = self._ensemble[i].predict(d_test, output_margin = True)
                    print(margins)
                    d_test.set_base_margin(margin = margins)
                predicted = self._ensemble[trees_in_ensemble - 1].predict(d_test)

                #print(predicted)
                return np.array(predicted > 0.5).astype(int)
        
        return np.zeros(X.shape[0])


    def reset(self): 
        self._first_run = True
        self._configure()

    def learn_one(self, x: dict, y):
        self._learn(np.array(list(x.values())), y)

    def learn_many(self, X: pd.DataFrame, y: pd.Series):
        self._learn(X.values(), y.values())

    def predict_one(self, x: dict):
        x = np.array(list(x.values()))
        return self._predict(x.reshape(1, -1))

    def predict_many(self, X : pd.DataFrame):
        return self._predict(X)
        