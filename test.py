from axgb.adaptive_xgboost import AdaptiveXGBoostClassifier
from sklearn import datasets
from river.datasets.synth import Agrawal
from sklearn.metrics import confusion_matrix, roc_auc_score
import pandas as pd

# Adaptive XGBoost classifier parameters
n_estimators = 30       # Number of members in the ensemble
learning_rate = 0.1     # Learning rate or eta
max_depth = 6           # Max depth for each tree in the ensemble
max_window_size = 1024  # Max window size
min_window_size = 16     # set to activate the dynamic window strategy
detect_drift = True    # Enable/disable drift detection

AXGBp = AdaptiveXGBoostClassifier(update_strategy='push',
                                  n_estimators=n_estimators,
                                  learning_rate=learning_rate,
                                  max_depth=max_depth,
                                  max_window_size=max_window_size,
                                  min_window_size=min_window_size,
                                  detect_drift=detect_drift)

AXGBr = AdaptiveXGBoostClassifier(update_strategy='replace',
                                  n_estimators=n_estimators,
                                  learning_rate=learning_rate,
                                  max_depth=max_depth,
                                  max_window_size=max_window_size,
                                  min_window_size=min_window_size,
                                  detect_drift=detect_drift)


true = [] 
pred_p = [] 
pred_r = []

agrawal = Agrawal(classification_function = 0, seed = 42)

i = 0 
for x, y in agrawal.take(10000):
    
    if i > min_window_size * 2:
        res = AXGBp.predict_one(x)
        pred_p.append(res)
       
        res = AXGBr.predict_one(x)
        pred_r.append(res)

        true.append(y)  

    AXGBr.learn_one(x, y)
    AXGBp.learn_one(x, y)

    i = i + 1



# Push strategy results 
mat = confusion_matrix(true, pred_p)
score = roc_auc_score(true, pred_p)

print("Push strategy confusion matrix:")
print(mat)
print("Push strategy ROC-AUC:")
print(score)

# Replace strategy results 
mat = confusion_matrix(true, pred_r)
score = roc_auc_score(true, pred_r)

print("\n")

print("Replace strategy confusion matrix:")
print(mat)
print("Replace strategy ROC-AUC:")
print(score)

