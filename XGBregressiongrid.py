
"""
Code for XGB_regression inital grid search
"""
from methods import Regression
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from helper_functions import DataWorkflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from pathlib import Path


def gridsearch():
# Setting up parameter grid
    parameters = {'nthread':[4],
                'booster': ['gbtree'],
                'objective':['reg:squarederror'],
                'learning_rate': [0.1],
                'gamma': [0],
                'reg_alpha':[0.001],
                'reg_lambda': [0.0],
                'max_depth': [9],
                'min_child_weight': [2],
                'subsample': [0.8],
                'colsample_bytree': [0.8],
                'n_estimators': [200]} # number of trees to fit

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
    results.to_csv(filep/regtype/'xgb_grid_search_results.csv', index=False)

# set file path
filep = Path("./Results/")
regtype = "XGB"
# Import data
X,y,ymax = DataWorkflow()
#creat const. feature
ones = np.ones((X.shape[0],1))
X = np.hstack([ones,X])

reg = xgb.XGBRegressor()
gridsearch()
