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

def plot ():
    plt.figure()
    plt.plot(y_train, y_ensemble, ".", label="adaboost", linewidth=2)
    plt.plot(y_test, test_p, '.', label='test adaboost')
    plt.title("Boosted Decision Tree Regression")
    plt.xlabel("y training data")
    plt.ylabel("predicted")
    plt.legend()
    plt.show()

def boostCV(X, y, ymax, iter_sch, depth, folds = 10):
    
    toi = pd.DataFrame(columns = ['MSE', 'R2', "data set", 'iter', "depth", "loss function" ])
    
    Xtemp, Xeval, ytemp, yeval = CV(X,y, folds =folds)

    for i in tqdm(range(folds)):
        for j,loss_func in enumerate(["linear", "square", "exponential"]):
            for itera in iter_sch:
                for depth in depth_sch:
                    Xtrain, Xtest, ytrain, ytest = CV(Xtemp[i],ytemp[i], folds =folds//2)
                    for m in tqdm(range(folds//2)):
                        
                        ada = AdaBoost(itera, depth, Xtrain[m], ytrain[m], Xtest[m], ytest[m], Xeval[i], yeval[i])
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

def boost(X, y , ymax, iter_sch, depth, folds):
    
    toi = pd.DataFrame(columns = ['MSE', 'R2', "data set", 'iter', "depth", "loss function"])
    
    Xtrain, Xtest, ytrain, ytest = CV(X,y, folds =folds)

    for i in tqdm(range(1)):
        for j,loss_func in enumerate(["square"]):
            for itera in iter_sch:
                for depth in depth_sch:
                    
                    ada = AdaBoost(itera, depth, Xtrain[i], ytrain[i], Xtest[i], ytest[i])
                    ada.initiateBoost(loss_func)
                    MSEtrain, R2train, MSEtest, R2test = ada.ensemble_eval(False)
                    
                    d = {"MSE": MSEtrain, "data set": "train", "R2":R2train, "iter": itera, "depth": depth, "loss function": loss_func }
                    #d.update({"beta%i"%k:ada.beta[k] for k in range(itera)})
                    toi = toi.append(d, ignore_index=True)
                    
                    d = {"MSE": MSEtest, "R2":R2test, "data set": "test", "iter": itera, "depth": depth, "loss function": loss_func }
                    toi = toi.append(d, ignore_index=True)
                    
    return toi, ada.y_train, ada.train_p

def pred_vs_actual(y,ymax, best_eval, filepath):
    
    p = best_eval
    plt.figure(figsize=(10,10))
    plt.scatter(y*ymax, p*ymax**2)
    plt.plot([0,ymax],[0,ymax], linestyle ='--')
    plt.xlabel('act. T$_c$ in K', fontsize =32)
    plt.ylabel("pred. T$_c$ in K ", fontsize= 32)
    #plt.xlim(0, ymax)
    #plt.ylim(0, ymax)
    plt.tick_params(size =24, labelsize=26)
    plt.tight_layout()
    plt.savefig(filepath/'pred_vs_act.pdf')
  
filep = Path("./Results/adaboost/")
X, y, ymax = DataWorkflow()
iter_sch = [100]
depth_sch = [1]
toi, y_test, test_p = boost(X, y, ymax, iter_sch, depth_sch, folds = 5)
toi.to_csv(filep/'toi.csv')
pred_vs_actual(y_test, ymax, test_p, filep)

'''  
X, y, ymax = DataWorkflow()
iter_sch = [100, 50, 20]
depth_sch = [3,2,1]
    
toi = boostCV(X, y, ymax, iter_sch, depth_sch, folds = 10)
toi.to_csv(filep/'adaboost'/'toi.csv')
'''