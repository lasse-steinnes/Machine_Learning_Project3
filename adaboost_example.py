# -*- coding: utf-8 -*-
"""
Script for running Adaboost example
"""

from adaboosting import AdaBoost
import matplotlib.pyplot as plt
import pandas as pd
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

def boostCV(X, y, ymax, iter_sch, depth, folds = 10):
    
    toi = pd.DataFrame(columns = ['MSE', 'R2', "data set", 'iter', "depth", "loss function" ]) #+ ["beta%i"%i for i in range(iter_sch[0])])
    Xtrain, Xtest, ytrain, ytest = CV(X,y, folds =folds)
   
    for i in tqdm(0,range(1)):
        for i,loss_func in enumerate(["linear", "square", "exponential"]):
            for itera in iter_sch:
                for depth in depth_sch:
                    print ('iteratoin', itera)
                    ada = AdaBoost(itera, depth,Xtrain[i],ytrain[i],Xtest[i],ytest[i])
                    
                    ada.initiateBoost(loss_func)
                    MSEtrain, R2train, MSEtest, R2test, MSEeval, R2eval = ada.ensemble_eval(False)
                    
                    d = {"MSE": MSEtrain, "data set": "train", "R2":R2train, "iter": itera, "depth": depth, "loss function": loss_func }
                    #d.update({"beta%i"%k:ada.beta[k] for k in range(itera)})
                    toi = toi.append(d, ignore_index=True)
                    
                    d = {"MSE": MSEtest, "R2":R2test, "data set": "test", "iter": itera, "depth": depth, "loss function": loss_func }
                    toi = toi.append(d, ignore_index=True)
                    
                    d = {"MSE": MSEeval, "R2":R2eval, "data set": "evaluation", "iter": itera, "depth": depth, "loss function": loss_func }
                    toi = toi.append(d, ignore_index=True)
                    
                    
    return toi

X, y, ymax = DataWorkflow()
iter_sch = [100, 50, 20]
depth_sch = [3,2,1]
    
toi = boostCV(X, y, ymax, iter_sch, depth_sch, folds = 10)
toi.to_csv(filep/'adaboost'/'toi.csv')