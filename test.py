from axgb.adaptive_xgboost import AdaptiveXGBoostClassifier
from sklearn import datasets
from sklearn.metrics import confusion_matrix, roc_auc_score
import pandas as pd

# Adaptive XGBoost classifier parameters
n_estimators = 30       # Number of members in the ensemble
learning_rate = 0.3     # Learning rate or eta
max_depth = 6           # Max depth for each tree in the ensemble
max_window_size = 100  # Max window size
min_window_size = 1     # set to activate the dynamic window strategy
detect_drift = False    # Enable/disable drift detection

AXGBp = AdaptiveXGBoostClassifier(update_strategy='push',
                                  n_estimators=n_estimators,
                                  learning_rate=learning_rate,
                                  max_depth=max_depth,
                                  max_window_size=max_window_size,
                                  min_window_size=min_window_size,
                                  detect_drift=detect_drift)

print("Test")

dataset = datasets.load_breast_cancer()
#print(dataset)
X, y = dataset.data, dataset.target

X = pd.DataFrame(X, columns = dataset['feature_names'])

pred = [] 
real = [] 

for i in range(X.shape[0]):
    x = X.iloc[i].to_dict()
    AXGBp.learn_one(x, y[i])
    res = AXGBp.predict_one(x)


    pred.append(res)
    real.append(y[i])

mat = confusion_matrix(real, pred)
score = roc_auc_score(real, pred)
print(mat)
print(score)