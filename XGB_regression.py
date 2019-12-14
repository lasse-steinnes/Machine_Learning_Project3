
"""
Code for XGB_regression
"""
from methods import Regression
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from helper_functions import DataWorkflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from pathlib import Path


# set file path
filep = Path("./Results/")
regtype = "XGB"
# Import data
X,y,ymax = DataWorkflow()
#creat const. feature
ones = np.ones((X.shape[0],1))
X = np.hstack([ones,X])

reg = xgb.XGBRegressor()

# Setting up parameter grid
parameters = {'nthread':[4],
              'booster': ['gblinear'],
              'objective':['reg:squarederror'],
              'learning_rate': [0.03, 0.05, 0.07,0.1,0.2,0.3],
              'gamma': [0.0, 0.1, 0.2, 0.3, 0.4],
              'reg_alpha':[0, 0.0001,0.001, 0.005, 0.01, 0.05,0,1],
              'reg_lambda': [0.0, 0.0001,0.001, 0.005, 0.01, 0.05,0,1],
              'max_depth': [4, 5, 6, 7],
              'min_child_weight': [1,2,4],
              'subsample': [0.7,0.8,0.9],
              'colsample_bytree': [0.7,0.8,0.9],
              'n_estimators': [100]} # number of trees to fit

grid = GridSearchCV(estimator=reg, param_grid=parameters, scoring='r2', n_jobs=4, cv= 5)
grid.fit(X, y)
print('\n All results:')
print(grid.cv_results_)
print('\n Best estimator:')
print(grid.best_estimator_)
print('\n Best score:')
print(grid.best_score_)
print('\n Best parameters:')
print(grid.best_params_)

results = pd.DataFrame(grid.cv_results_)
feature_importance = pd.DataFrame(grid.coef_)
results.to_csv(filep/regtype/'xgb_grid_search_results.csv', index=False)
feature_importance.to_csv(filep/regtype/'feature_importance.csv', index=False)
