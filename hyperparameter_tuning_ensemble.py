import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, f1_score
import joblib

# ------------------- Load Extracted Features -------------------
X_train_comb = np.load("X_train_comb.npy")
y_train = np.load("y_train.npy")

# ------------------- Scoring Metric -------------------
scorer = make_scorer(f1_score, average='weighted')

# ------------------- XGBoost Grid Search -------------------
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_param_grid = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb_grid = GridSearchCV(estimator=xgb, param_grid=xgb_param_grid,
                        scoring=scorer, cv=3, verbose=2, n_jobs=-1)

print("Tuning XGBoost...")
xgb_grid.fit(X_train_comb, y_train)
print("Best XGBoost Parameters:", xgb_grid.best_params_)

# Save best XGB model
joblib.dump(xgb_grid.best_estimator_, "best_xgb.pkl")


# ------------------- Random Forest Grid Search -------------------
rf = RandomForestClassifier()
rf_param_grid = {
    'n_estimators': [200, 500, 800, 1000],
    'max_depth': [None, 10, 20],
    'max_features': ['sqrt', 'log2']
}

rf_grid = GridSearchCV(estimator=rf, param_grid=rf_param_grid,
                       scoring=scorer, cv=3, verbose=2, n_jobs=-1)

print("Tuning Random Forest...")
rf_grid.fit(X_train_comb, y_train)
print("Best Random Forest Parameters:", rf_grid.best_params_)

# Save best RF model
joblib.dump(rf_grid.best_estimator_, "best_rf.pkl")
