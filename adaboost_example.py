# -*- coding: utf-8 -*-
"""
Script for running Adaboost example. Implementation is from following Drucker 
1997 paper. 
"""

from adaboosting import AdaBoost
import matplotlib.pyplot as plt
import pandas as pd
from helper_functions import DataWorkflow, CV
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.model_selection import train_test_split

def boost(X, y , ymax, iter_sch, depth_sch, functions, folds, evaluationSet = False):
    '''
    A function to run adaboost with different hyper-parameters and
    to perform kfold cross-validation. 
    
    Inputs:
        X : (n_samples, n_features)
        The design matrix
        y : (n_samples, 1)
        The y data
        ymax : int
        The maximum value of y, this is found during scaling of data.
        inter_sch : list(various iterations, 1)
        The iterations schedule for training, the iterations are int
        depth_sch : list(number of depths,1)
        The depth schedule for training, the depths are int
        functions : (no. functions,1)
        the functions are a string : "linear", "square" or "exponential".
        folds: int
        the number of folds for K-fold CV
        evaluationSet: Boolean
        If True there is an evaluation set as well as test and train.
        
    Returns:
        toi : pandas.DataFrame (columns = 'MSE', 'R2', "data set", 'iter', "depth", "loss function")
        Table of information containing all relevant data
        y_eval : (n_samples,1)
        The y evaluation data
        eval_predict : (n_samples,1)
        The prediction of evaluation data
        eval_MSE*ymax**2 : int
        The mse of the evaluation prediction multiplied by ymax squared
        to scale it back to as mse should be prior to scaling. 
        eval_R2 : int
        The R2_score of the evaluation data.
    
    '''
    toi = pd.DataFrame(columns = ['MSE', 'R2', "data set", 'iter', "depth", "loss function"])
    
    if evaluationSet == True:
        np.random.seed(2019)
        np.random.shuffle(X)
        np.random.seed(2019)
        np.random.shuffle(y)
        X, X_eval, y, y_eval = train_test_split(X, y, test_size = 0.1)
        
    Xtrain, Xtest, ytrain, ytest = CV(X,y, folds =folds)

    for i in tqdm(range(1)):
        for iteration in tqdm(iter_sch):
            for depth in tqdm(depth_sch):
                for loss_func in functions: 
                    ada = AdaBoost(iteration, depth, Xtrain[i], ytrain[i], Xtest[i], ytest[i])
                    ada.training(loss_func)
                    train_predict, train_MSE, train_R2 = ada.evaluate(Xtrain[i],ytrain[i])
                    test_predict, test_MSE, test_R2 = ada.evaluate(Xtest[i], ytest[i])
                    
                    d = {"MSE": train_MSE*ymax**2, "R2": train_R2, "iter": iteration, "depth": depth, "loss function": loss_func, "data set": "train"}
                    #d.update({"beta%i"%k:ada.beta[k] for k in range(itera)})
                    toi = toi.append(d, ignore_index=True)
                    
                    d = {"MSE": test_MSE*ymax**2, "R2": test_R2, "iter": iteration, "depth": depth, "loss function": loss_func, "data set": "test"}
                    toi = toi.append(d, ignore_index=True)
    
    eval_predict, eval_MSE, eval_R2 = ada.evaluate(X_eval, y_eval)
    d = {"MSE": eval_MSE*ymax**2, "R2": eval_R2, "iter": iteration, "depth": depth, "loss function": loss_func, "data set": "evaluation"}
    toi = toi.append(d, ignore_index=True)
                
    return toi, y_eval, eval_predict, eval_MSE*ymax**2, eval_R2

def Stats2(toi, filepath, ymax, skip_eval = True):
    """
    find optimal model by looking at toi, evaluate CV performance
    returns optimal parameters
    """
    f = open(filepath/"stats2.txt",'w')
    #find best row based on test
    toi = toi.groupby(['iter', "depth", "loss function", "data set"], as_index = False).mean()
    
    toi.to_csv(filepath/'grouped.csv')
    
    idx = toi[toi["data set"]=="test"]["MSE"].idxmin()
    tabel = {"test": toi.iloc[idx]}  
    tabel.update({ "train": toi.iloc[idx+1]}) 
    if not skip_eval: 
        tabel.update({"eval": toi.iloc[idx +1]})
    f.write('Best model:\n')
    for data_set in ["train","test", "eval"]: 
        if skip_eval & (data_set =='eval'):
            break
        f.write(data_set +': \n')
        
        for name in ["MSE", "R2"]:
            
            f.write(name + ': %.9f\n'%tabel[data_set][name])
            
        for name in ["iter", "depth", "loss function"]:
            
            f.write(name + ': %s' %tabel[data_set][name])
            
            f.write("\n")
            
        f.write("ymax is: %.5f" %ymax)
    f.close()
    
def pred_vs_actual(y,ymax, p, MSE, R2, filepath):
    '''
    A function to plot the actual temperature vs predicted temperature.
    '''
    
    plt.figure(figsize=(10,10))
    plt.scatter(y*ymax, p*ymax , label="adaboost \n MSE: %.3f, R2: %.2f"%(MSE, R2))
    plt.plot([0,ymax],[0,ymax], linestyle ='--')
    plt.xlabel('act. T$_c$ in K', fontsize =32)
    plt.ylabel("pred. T$_c$ in K ", fontsize= 32)
    plt.legend()
    #plt.xlim(0, ymax)
    #plt.ylim(0, ymax)
    plt.tick_params(size =24, labelsize=26)
    plt.tight_layout()
    plt.savefig(filepath/'pred_vs_act.pdf')
  

filep = Path("./Results/adaboost/")
X, y, ymax = DataWorkflow()
iter_sch = [20,50,100]
depth_sch = [1,2,3,5,10,20]
functions = ["square", "linear", "exponential"]
toi, y_eval, eval_predict, MSE, R2  = boost(X, y, ymax, iter_sch, depth_sch, functions, folds = 10, evaluationSet = True)
toi.to_csv(filep/'toi.csv')
pred_vs_actual(y_eval, ymax, eval_predict, MSE, R2, filep/'eval')

#store best parameters in txt file    
Stats2(toi, filep, ymax, skip_eval = True)
