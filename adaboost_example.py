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
from pathlib import Path

filep = Path("./Results/")

def plot ():
    plt.figure()
    plt.plot(ada.y_train, y_ensemble, ".", label="adaboost", linewidth=2)
    plt.plot(ada.y_test, ada.test_p, '.', label='test adaboost')
    plt.title("Boosted Decision Tree Regression")
    plt.xlabel("y training data")
    plt.ylabel("predicted")
    plt.legend()
    plt.show()

def boostCV(X, y, ymax, loss, iter_sch, folds = 10, depth = 2):
    
    toi = pd.DataFrame(columns = ['MSE', 'R2', 'iter', "depth" ]) #+ ["beta%i"%i for i in range(iter_sch[0])])
    Xtrain, Xtest, ytrain, ytest = CV(X,y, folds =folds)
    
    for i in tqdm(range(folds)):
        for i,loss_func in enumerate(["linear", "square", "exponential"]):
            for itera in iter_sch:
                for depth in depth_sch:
                    ada = AdaBoost(itera, depth)
                    ada.X_train = Xtrain[i]
                    ada.y_train = ytrain[i]
                    ada.initiateBoost(loss_func)
                    MSE, R2 = ada.ensemble_predict(False)
                    
                    d = {"MSE": MSE, "R2":R2, "iter": itera, "depth": depth }
                    #d.update({"beta%i"%k:ada.beta[k] for k in range(itera)})
                    toi = toi.append(d, ignore_index=True)
    
    return toi

X, y, ymax = DataWorkflow()
iter_sch = [100, 50, 20]
depth_sch = [3,2,1]
    
toi = boostCV(X, y, ymax, iter_sch, depth_sch, folds = 10)
toi.to_csv(filep/'adaboost'/'toi.csv')