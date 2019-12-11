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
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.model_selection import train_test_split

def plot ():
    plt.figure()
    plt.plot(y_train, y_ensemble, ".", label="adaboost", linewidth=2)
    plt.plot(y_test, test_p, '.', label='test adaboost')
    plt.title("Boosted Decision Tree Regression")
    plt.xlabel("y training data")
    plt.ylabel("predicted")
    plt.legend()
    plt.show()

def boostCV(X, y, ymax, iter_sch, depth_sch, folds = 10):
    
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

def boost(X, y , ymax, iter_sch, depth_sch, functions, folds, evaluationSet = False):
    
    toi = pd.DataFrame(columns = ['MSE', 'R2', "data set", 'iter', "depth", "loss function"])
    
    if evaluationSet == True:
        np.random.seed(2019)
        np.random.shuffle(X)
        np.random.seed(2019)
        np.random.shuffle(y)
        X, X_eval, y, y_eval = train_test_split(X, y, test_size =0.1)
        
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

def final_predict(X,y, best_trees, best_betas):
    p = np.zeros(len(y))
    for k in range(0,len(best_betas)):
        p += best_betas[k] * best_trees[k].predict(X)
    MSE = mean_squared_error(y, p)
    R2 = r2_score(y, p)
     
    return p, MSE, R2

def Stats(toi, filepath, plot_par = False, features =81, skip_eval=False, bayse =False):
    """
    find optimal model by looking at toi, evaluate CV performance
    returns optimal parameters
    """
    f = open(filepath/"stats.txt",'w')
    #find best row based on test
    idx = toi[toi["data set"]=="test"]["MSE"].idxmin()
    tabel = {"test": toi.iloc[idx]}  
    tabel.update({ "train": toi.iloc[idx-1]}) 
    if not skip_eval: 
        tabel.update({"eval": toi.iloc[idx +1]})
    f.write('Best model:\n')
    for data_set in ["train","test", "eval"]: 
        if skip_eval & (data_set =='eval'):
            break
        f.write(data_set +': \n')
        
        for name in ["MSE", "R2"]:
            
            f.write(name + ': %.9f\n'%tabel[data_set][name])
            
        #for name in ["iter", "depth", "lossfunction"]:
            
            #f.write(name + ': %s\n'%tabel[data_set][name])
            
        f.write("\n")
    #model variablity at best lam    
    if not bayse:
        toi = toi[toi["depth"]==tabel["test"]["depth"]]

    av = toi.groupby("data set").agg(["mean","std"])

    f.write('Average model:\n')
    for data_set in ["train","test", "eval"]: 
        if skip_eval & (data_set =='eval'):
            break
        f.write(data_set +' CV reults: \n')
        for name in ["MSE", "R2"]:
            
            f.write(name + ': %.9f +- %.9f\n'%(av.loc[data_set][name, "mean"],av.loc[data_set][name, "std"]))
            
        f.write("\n")
    f.close()

    inds3 = ["par%i"%i for i in range(features)]
    if plot_par:
        plt.figure(figsize =(10,10))
        inds = [("par%i"%i, 'mean') for i in range(features)]
        inds2 = [("par%i"%i, 'std') for i in range(features)]

        #95% confidence interval
        plt.errorbar(np.arange(features),av.loc["test"][inds], yerr=1.96*av.loc["test"][inds2], linestyle ='', marker='o', color ='tab:orange', label = r'95% conf. interv.')
        plt.plot(np.arange(features), tabel["test"][inds3], linestyle ='', marker ='x', color = 'tab:green', label ='Best param.')
        plt.legend(loc='best', fontsize = 28)
        plt.ylabel('par. value', fontsize =32)
        plt.xlabel("par. number", fontsize= 32)
        plt.xlim(-0.5, features+0.5)
        plt.tick_params(size =24, labelsize=26)
        plt.tight_layout()
        plt.savefig(filepath/'params.pdf')

    return tabel["test"][inds3]

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
  
def main():
    filep = Path("./Results/adaboost/")
    X, y, ymax = DataWorkflow()
    iter_sch = [15]
    depth_sch = [10]
    functions = ["square"]#, "linear", "exponential"]
    toi, y_eval, eval_predict, MSE, R2  = boost(X, y, ymax, iter_sch, depth_sch, functions, folds = 5, evaluationSet = True)
    toi.to_csv(filep/'toi.csv')
    pred_vs_actual(y_eval, ymax, eval_predict, MSE, R2, filep/'eval')
    
main()
filep = Path("./Results/adaboost/")
toi = pd.read_csv(filep/'toi.csv')
#ind3 = Stats(toi, filep, plot_par = False, skip_eval = True)
Stats2(toi, filep, ymax, skip_eval = True)
