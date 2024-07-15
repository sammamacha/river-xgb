import numpy as np 
import pandas as pd

from river.drift import ADWIN 
from river.base import MiniBatchClassifier

from sklearn.metrics import roc_auc_score

import xgboost as xgb 

from axgb.util import row_convert, matrix_convert

class BoaswinXGBoostClassifier(MiniBatchClassifier):
    _STATE_NORMAL = 0 
    _STATE_ALERT = 1 
    _STATE_DRIFT = 2 
    _STATES = [_STATE_NORMAL, _STATE_ALERT, _STATE_DRIFT]

    def __init__(self,
                 n_estimators = 30,
                 learning_rate = 0.3, 
                 max_depth = 6, 
                 gamma = 0, 
                 max_window_size = 500,
                 min_window_size = 100, 
                 alpha = 0.95, 
                 beta = 0.90, 
                 xgb_model = None | xgb.Booster,
                 window_metrics = roc_auc_score):
        """
        Bayesian Optimized Adaptive Sliding Window XGBoost classifier. 

        Parameters
        ----------
        n_estimators: int (default=30)
            The number of estimators in the ensemble.

        learning_rate:
            Learning rate, a.k.a eta.

        max_depth: int (default = 6)
            Max tree depth.

        gamma: float (default = 0)
            Minimum loss reduction to create a new split. 
    
        max_window_size: int (default = 500)
            Maximum size of the sliding window. 

        min_window_size: int (default = 100)
            Minimum size of the sliding window. 

        alpha: float (default = 0.95)
            The alert level. 

        beta: float (default = 0.90)
            The drift level. 

        xgb_model: Booster|None (default = None)
            Offline model trained on historical dataset 

        window_metrics: (default = roc_auc_score)   
            Accuracy evaluation when comparing windows 
        """
        super().__init__()

        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.gamma = gamma 
        self.max_window_size = max_window_size
        self.min_window_size = min_window_size
        self.alpha = alpha 
        self.beta = beta 
        self.window_metrics = window_metrics

        self._booster = xgb_model
        self.dynamic_window_size = min_window_size
        self._first_run = True 
        self._X_buffer = np.array([])
        self._y_buffer = np.array([])
        self._X_adaptive_buffer = np.array([])
        self._y_adaptive_buffer = np.array([])

        self._state = self._STATE_NORMAL
        self._boosting_params = {"verbosity": 0,
                                 "objective": "binary:logistic",
                                 "eta": self.learning_rate,
                                 "max_depth": self.max_depth,
                                 "gamma" : self.gamma}

    def _reset_window_size(self):
        if self.min_window_size:
            self._dynamic_window_size = self.min_window_size
        else:
            self._dynamic_window_size = self.max_window_size
        self.window_size = self._dynamic_window_size

    def _train_booster(self, X: np.ndarray, y: np.ndarray, override = False): 
        d_batch_train = xgb.DMatrix(X, y.astype(int))

        if self._booster and override:
            self._booster = xgb.train(self._boosting_params,
                                 dtrain = d_batch_train,
                                 num_boost_round = 5,
                                 verbose_eval = False,
                                 xgb_model = self._booster)
        elif self._booster == None: 
            self._booster = xgb.train(self._boosting_params,
                                 dtrain = d_batch_train,
                                 num_boost_round = 5,
                                 verbose_eval = False)

        
    def _drift_warning(self, X, y, acc_wp, acc_w):
        dynamic_size = self._X_adaptive_buffer.shape[0]

        if acc_w < self.beta * acc_wp: 
            self._state = self._STATE_DRIFT 

            d_train = xgb.DMatrix(self._X_adaptive_buffer,  
                                  label = self._y_adaptive_buffer)
            self._booster = xgb.train(self._boosting_params,
                                      dtrain = d_train,
                                      num_boost_round = 5,
                                      verbose_eval = False,
                                      xgb_model = self._booster)    
        elif (acc_w >= acc_wp * self.alpha) or (dynamic_size >= self.max_window_size): 
            self._state = self._STATE_NORMAL
            self._X_adaptive_buffer = np.empty((0, X.shape[0]))
            self._y_adaptive_buffer = np.array([])
        else: 
            self._X_adaptive_buffer = np.concatenate([self._X_buffer, X])
            self._y_adaptive_buffer = np.append(self._y_buffer, y) 



    def _drift_adaptation(self, X, y):
        X_w = self._X_buffer[-self.min_window_size]
        y_wp = self._y_buffer[-self.min_window_size * 2:-self.min_window_size]

        y_w = self._y_buffer[-self.min_window_size]
        X_wp = self._X_buffer[-self.min_window_size * 2:-self.min_window_size]

        # If there's no model we go ahead and make one 
        self._train_booster(X_wp, y_wp, override = False)

        d_batch_w = xgb.DMatrix(X_w)
        d_batch_wp = xgb.DMatrix(X_wp)

        acc_w = self._booster.predict(X_w)
        acc_w = np.array(acc_w > 0.5).astype(int)
        acc_w = self.window_metrics(y_w, acc_w)

        acc_wp = self._booster.predict(X_wp)
        acc_wp = np.array(acc_wp > 0.5).astype(int)
        acc_wp = self.window_metrics(y_wp, acc_wp)

        if acc_w < self.alpha * acc_wp: 
            self._state = self._STATE_ALERT

            self._X_adaptive_buffer = np.concatenate([self._X_buffer, X])
            self._y_adaptive_buffer = np.append(self._y_buffer, y) 
    
        if self._state == self._STATE_ALERT: 
            self._drift_warning(X, y, acc_wp, acc_w)
            while self._state == self._STATE_DRIFT: 
                self._drift_warning(X, y, acc_wp, acc_w) 

    def _learn(self, X: np.ndarray, y: np.ndarray):
        if self._first_run: 
            self._X_buffer = np.empty((0, X.shape[0]))
            self._y_buffer = np.array([])

            self._X_adaptive_buffer = np.empty((0, X.shape[0]))
            self._y_adaptive_buffer = np.array([])

            self._first_run = False 

        X = X.reshape(1, -1)


        self._X_buffer = np.concatenate([self._X_buffer, X])
        self._y_buffer = np.append(self._y_buffer, y) 
        
        # If we strayed too far we should trim the front 
        while self._X_buffer.shape[0] > self.min_window_size * 2: 
            delete_idx = [i for i in range(self.min_window_size * 2 - self._X_buffer.shape[0])]
            self._X_buffer = np.delete(self._X_buffer, delete_idx, axis = 0)
            self._y_buffer = np.delete(self._y_buffer, delete_idx, axis = 0)           

        # We needed to wait for it to accumulate before comparing 
        # current window and previous one 
        if self._X_buffer.shape[0] >= 2 * self.min_window_size:
            self._concept_drift_adaptation(X, y)


            
            


    def learn_many(self, X, y):
        pass

    def learn_one(self, x, y):
        pass

    def predict_many(self, X):
        pass

    def predict_one(self, x):
        pass

    def predict_proba_many(self, X):
        pass

    def predict_proba_one(self, x):
        pass