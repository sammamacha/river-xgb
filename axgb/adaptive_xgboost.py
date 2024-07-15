import numpy as np 
import pandas as pd

from river.drift import ADWIN 
from river.base import MiniBatchClassifier

import xgboost as xgb 

from axgb.util import row_convert, matrix_convert


"""
This is a modified port of https://github.com/jacobmontiel/AdaptiveXGBoostClassifier made to 
fit into the "MiniBatchClassifier" class instead of "ClassifierMixin" and some 
comments added for self-clarity. 

"""
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
        """
        Adaptive XGBoost classifier.

        Parameters
        ----------
        n_estimators: int (default=30)
            The number of estimators in the ensemble.

        learning_rate:
            Learning rate, a.k.a eta.

        max_depth: int (default = 6)
            Max tree depth.

        max_window_size: int (default=1000)
            Max window size.

        min_window_size: int (default=None)
            Min window size. If this parameters is not set, then a fixed size
            window of size ``max_window_size`` will be used.

        detect_drift: bool (default=False)
            If set will use a drift detector (ADWIN).

        update_strategy: str (default='replace')
            | The update strategy to use:
            | 'push' - the ensemble resembles a queue
            | 'replace' - oldest ensemble members are replaced by newer ones

        Notes
        -----
        The Adaptive XGBoost [1]_ (AXGB) classifier is an adaptation of the
        XGBoost algorithm for evolving data streams. AXGB creates new members
        of the ensemble from mini-batches of data as new data becomes
        available.  The maximum ensemble  size is fixed, but learning does not
        stop once this size is reached, the ensemble is updated on new data to
        ensure consistency with the current data distribution.

        References
        ----------
        .. [1] Montiel, Jacob, Mitchell, Rory, Frank, Eibe, Pfahringer,
           Bernhard, Abdessalem, Talel, and Bifet, Albert. “AdaptiveXGBoost for
           Evolving Data Streams”. In:IJCNN’20. International Joint Conference
           on Neural Networks. 2020. Forthcoming.
        """
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
        """
        Continually double window size until max is reached. 
        """
        if self._dynamic_window_size < self.max_window_size:
            self._dynamic_window_size *= 2
            if self._dynamic_window_size > self.max_window_size:
                self.window_size = self.max_window_size
            else:
                self.window_size = self._dynamic_window_size

    def _reset_window_size(self):
        """
        Set window size back to minimum, used when ran for first time, resetted or when ADWIN 
        detects a concept drift. 
        """
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
        self._boosting_params = {"verbosity": 0,
                                 "objective": "binary:logistic",
                                 "eta": self.learning_rate,
                                 "max_depth": self.max_depth}
        if self.detect_drift:
            self._drift_detector = ADWIN()

    def _train_on_minibatch(self, X : np.ndarray, y: np.ndarray):
        """
            'replace' - new boosters are added to the front
            'push' - new boosters are added to back FIFO
        """
        if self.update_strategy == self._REPLACE_STRATEGY: 
            booster = self._train_booster(X, y, self._model_idx)

            self._ensemble[self._model_idx] = booster 
            self._samples_seen += X.shape[0]
            self._update_model_idx() 
        else: # == self._PUSH_STRATEGY
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
                            num_boost_round = 5, # Original repo uses 1, modify at discretion
                            verbose_eval = False)
        return booster

    def _update_model_idx(self):
        self._model_idx += 1
        if self._model_idx == self.n_estimators:
            self._model_idx = 0

    def _learn(self, X: np.ndarray, y: np.ndarray):
        if self._first_run: 
            self._X_buffer = np.empty((0, X.shape[0])) 
            self._y_buffer = np.array([])
            self._first_run = False
        X = X.reshape(1, -1)

        self._X_buffer = np.concatenate([self._X_buffer, X])
        self._y_buffer = np.append(self._y_buffer, y)

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
        X = X.reshape(1, -1)
        if self._ensemble: 
            if self.update_strategy == self._REPLACE_STRATEGY:
                trees_in_ensemble = sum(i is not None for i in self._ensemble)
            else: 
                trees_in_ensemble = len(self._ensemble)
            
            if trees_in_ensemble > 0: 
                d_test = xgb.DMatrix(X)

                # Forward Additive modeling as seen in Equation (4) in "AdaptiveXGBoost for
                # Evolving Data Streams" is done this way via 'set_base_margin'
                for i in range(trees_in_ensemble - 1): 
                    margins = self._ensemble[i].predict(d_test, output_margin = True)
                    d_test.set_base_margin(margin = margins)
                predicted = self._ensemble[trees_in_ensemble - 1].predict(d_test)
                return np.array(predicted)
        return np.zeros(X.shape[0])


    def reset(self): 
        self._first_run = True
        self._configure()

    def learn_many(self, X, y):
        X = matrix_convert(X)
        for i in range(X.shape[0]):
            self._learn(X[i], y[i])

    def learn_one(self, x, y):
        x = row_convert(x)
        self._learn(x, y)

    def predict_many(self, X):
        p = self.predict_proba_many(X) 
        return np.array(p > 0.5).astype(int)

    def predict_one(self, x):
        x = row_convert(x)
        p = self.predict_proba_many(x)        
        return np.array(p > 0.5).astype(int)

    def predict_proba_many(self, X):
        X = matrix_convert(X)
        return self._predict(X) 

    def predict_proba_one(self, x):
        x = row_convert(x)
        return self._predict(x) 


        