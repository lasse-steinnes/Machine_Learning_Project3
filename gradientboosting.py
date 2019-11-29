# -*- coding: utf-8 -*-
"""
class for Gradient boosting
"""
from adaboosting import AdaBoost
from helper_functions import MSE, importData, scaler
import numpy as np
import pandas as pd
from sklearn import tree

import matplotlib.pyplot as plt

class GradientBoost:
    
    def __init__(self, iterations, depth, eta):
        '''
        accepts X_train and X_test which are (n_samples, n_features)
        accepts y_train and y_test in the shape (n_samples, 1)
        accepts depth as integer
        accepts iterations as integer
        
        '''
        X, y = importData()
        self.iterations = iterations
        self.depth = depth
        
        X_train, X_test, self.y_train, self.y_test = AdaBoost.shuffleAndsplit(self, X, y)
        self.X_train, self.X_test = scaler(X_train, X_test)
        self.n = X_train.shape[0]
    
    def initiateSuperBoost(self):   
        
        self.ensemble_train_pred = np.array([0.0 for i in range(0,len(self.y_train))])
        self.ensemble_test_pred = np.array([0.0 for i in range(0,len(self.y_test))])
        
        reg_weak = tree.DecisionTreeRegressor(max_depth = self.depth)
        reg_weak.fit(self.X_train, self.y_train)
            
        for i in range(0,self. iterations+1): 
            train_pred = reg_weak.predict(self.X_train)
            test_pred = reg_weak.predict(self.X_test)
            
            train_res = (train_pred - self.y_train)
            reg_weak.fit(self.X_train, train_res)
            
            self.ensemble_train_pred += train_pred
            self.ensemble_test_pred += test_pred
            
grad = GradientBoost(100, 3, 0.1)
grad.initiateSuperBoost()

plt.figure()
plt.plot(grad.y_train, grad.ensemble_train_pred, ".", label="gradientboost", linewidth=2)
plt.plot(grad.y_test, grad.ensemble_test_pred, '.', label='test gradientboost')
plt.title("Boosted Decision Tree Regression")
plt.xlabel("y training data")
plt.ylabel("predicted")
plt.legend()
plt.show()