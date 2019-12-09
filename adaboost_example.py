# -*- coding: utf-8 -*-
"""
Script for running Adaboost example
"""

from adaboosting import AdaBoost
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from helper_functions import DataWorkflow, CV
from tqdm import tqdm

X,y,ymax = DataWorkflow()
iter_sch = [100, 50, 20]
for i,loss in enumerate(["linear", "square", "exponential"]):
    
    toi = BoostCV(X, y, ymax, loss_func, iter_sch, folds = 10, depth = 2)
    
def plot ():
    plt.figure()
    plt.plot(ada.y_train, y_ensemble, ".", label="adaboost", linewidth=2)
    plt.plot(ada.y_test, ada.test_p, '.', label='test adaboost')
    plt.title("Boosted Decision Tree Regression")
    plt.xlabel("y training data")
    plt.ylabel("predicted")
    plt.legend()
    plt.show()

def BoostCV(X, y, ymax, loss, iter_sch, folds = 10, depth = 2):
    
    toi = pd.Dataframe(columns = ['MSE', 'R2', 'iter', 'tot_iter' ] + ["weight%i"%i for i in range(X.shape[0])])
    Xtrain, Xtest, ytrain, ytest = CV(X,y, folds =folds)
    ada = Adaboost()
    for i in tqdm(range(folds)):
        
        Adaboost.X_train = Xtrain[i]
        Adaboost.y_train = ytrain[i]
        ada = AdaBoost(tot_iter, depth)
        ada.initiateBoost(loss_func)
        y_ensemble = ada.ensemble_predict(False)
        
    
    return toi