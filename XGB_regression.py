
"""
Code for XGB_regression
"""
from methods import Regression
from helper_functions import DataWorkflow, CV
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from  tqdm import tqdm
from sklearn.metrics import mean_squared_error
from pathlib import Path

# Function that performs manual grid search
def XGB_GridCV(X,y,Xtest, ytest, paths,  folds, toi, fnum, reg):
    """
    Function that performs grid search for XGB regression
    Paths is all the best hyperparameters from the gridsearch (XGBregressiongrid.py)
    in an array object.
    paths: [booster,learning_rate,gamma,alpha,lam,depth,child_weight, subs,cols, n_estimators]
    """
    booster,learning_rate,gamma,alpha,lam,depth,child_weight, subs,cols, n_estimators = paths
    features = X.shape[1]
    Xinner_train, Xinner_test, yinner_train, yinner_test = CV(X,y, folds =folds)
    for j in range(folds//2):
        reg.X = Xinner_train[j]
        reg.y = yinner_train[j]

        MSE, R2 = reg.weak_regressor(booster, depth, child_weight,subs,cols, n_estimators,learning_rate,gamma, alpha, lam)
        d = {"fit num":fnum,"MSE": MSE, "R2":R2, "unc. Tc": np.sqrt(MSE)*ymax,"eta": learning_rate,"gamma":gamma,"alpha": alpha,"lambda":lam,"max_depth":depth,"min_child_weight":child_weight,"subsample":subs,"colsample_bytree":cols,"n_trees":n_estimators, "data set": "train"}
        d.update({"par%i"%k:reg.weights[k] for k in range(features)})
        toi = toi.append(d, ignore_index=True)

        MSE, R2 = reg.evaluation(Xinner_test[j], yinner_test[j])
        d = {"fit num":fnum,"MSE": MSE, "R2":R2, "unc. Tc": np.sqrt(MSE)*ymax,"eta": learning_rate,"gamma":gamma,"alpha": alpha,"lambda":lam,"max_depth":depth,"min_child_weight":child_weight,"subsample":subs,"colsample_bytree":cols,"ntrees":n_estimators, "data set": "test"}
        d.update({"par%i"%k:reg.weights[k] for k in range(features)})
        toi =toi.append(d, ignore_index=True)

        MSE, R2 = reg.evaluation(Xtest, ytest)
        d = {"fit num":fnum,"MSE": MSE, "R2":R2, "unc. Tc": np.sqrt(MSE)*ymax,"eta": learning_rate,"gamma":gamma,"alpha": alpha,"lambda":lam,"max_depth":depth,"min_child_weight":child_weight,"subsample":subs,"colsample_bytree":cols,"n_trees":n_estimators, "data set": "eval"}
        d.update({"par%i"%k:reg.weights[k] for k in range(features)})
        toi = toi.append(d, ignore_index=True)
        fnum +=1
    return ytest,reg.y_pred,toi, fnum

# Function that performs cross validation
def XGB_CV(X, y, ymax,paths, folds = 10):
    """
    Performs cross validation for XGB regression with hyperparameter tuning (grid search)
    A nested CV of folds*folds/2 is performed for each hyperparameter
    kargs:
        folds:            number of cross validations
        Parameter paths:  list of hyper parameters to test
    returns tabel of information (toi) with all scores, parameters and hyperparameter
    """
    features = X.shape[1]
    reg = Regression()
    toi = pd.DataFrame(columns=["fit num","MSE","R2","unc. Tc", "eta","gamma","alpha","lambda","max_depth","min_child_weight","subsample","colsample_bytree","n_trees", "data set",] + ["par%i"%i for i in range(features)])

    Xtrain, Xtest, ytrain, ytest = CV(X,y, folds = folds)
    fnum = 0
    for i in tqdm(range(folds)):
        y_test, pred,toi, fnum = XGB_GridCV(Xtrain[i],ytrain[i], Xtest[i], ytest[i], paths, folds//2, toi, fnum, reg)
    return y_test,pred,toi

# storing statistics
def XGB_stats(toi, filepath, plot_par = False, features =81, skip_eval=False):
    """
    find optimal model by looking at toi, evaluate CV performance
    returns optimal parameters
    """
    f = open(filepath/"xgb_stats.txt",'w')

    #find best row based on test
    idx = toi[toi["data set"]=="test"]["MSE"].idxmin() # finding mininum MSE
    tabel = {"test": toi.iloc[idx]}
    tabel.update({ "train": toi.iloc[idx-1]})
    if not skip_eval:
        tabel.update({"eval": toi.iloc[idx +1]})
    f.write('Best model:\n')
    for data_set in ["train","test", "eval"]:
        if skip_eval & (data_set =='eval'):
            break
        f.write(data_set +': \n')

        for name in ["MSE", "R2", "unc. Tc","eta","gamma","alpha","lambda","max_depth","min_child_weight","subsample","colsample_bytree","n_trees"]:
            f.write(name + ': %.9f\n'%tabel[data_set][name])

        f.write("\n")

    # model variablity at best hyper parameter
    av = toi.groupby("data set").agg(["mean","std"])

    f.write('Average model:\n')
    for data_set in ["train","test", "eval"]:
        if skip_eval & (data_set =='eval'):
            break
        f.write(data_set +' CV results: \n')
        for name in ["MSE", "R2", "unc. Tc"]:

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
        plt.savefig(filepath/'XGB_params.pdf')

    return tabel["test"][inds3]

#  Plot the predicted versus the actual
def pred_vs_actual(y,y_pred,ymax, filepath):

    p = y_pred
    plt.figure(figsize=(10,10))
    plt.scatter(y*ymax, p*ymax)
    plt.plot([0,ymax],[0,ymax], linestyle ='--')
    plt.ylabel('act. T$_c$ in K', fontsize =32)
    plt.xlabel("pred. T$_c$ in K ", fontsize= 32)
    plt.xlim(0, ymax)
    plt.ylim(0, ymax)
    plt.tick_params(size =24, labelsize=26)
    plt.tight_layout()
    plt.savefig(filepath/'XGB_pred_vs_act.pdf')


# set file path
filep = Path("./Results/")
regtype = "XGB"
# Import data
X,y,ymax = DataWorkflow()
#creat const. feature
ones = np.ones((X.shape[0],1))
X = np.hstack([ones,X])

booster = 'gbtree';learning_rate = 0.1; gamma = 0.0;
alpha = 0.001; lam = 0.0; depth = 9; child_weight = 2;
subs = 0.8;cols = 0.8; n_estimators = 200

paths = [booster,learning_rate,gamma,alpha,lam,depth,child_weight, subs,cols, n_estimators]

# Storing statistics for best model

filep = Path("./Results/")

skip = False
X,y,ymax = DataWorkflow()

#creat const. feature
ones = np.ones((X.shape[0],1))
X = np.hstack([ones,X])

features = X.shape[1]
reg = "XGB"
print('Regression: XGB')
print("Grid")
y_test,y_pred,toi = XGB_CV(X,y,ymax,paths, folds =5)
toi.to_csv(filep/reg/'toi.csv')
best_par = XGB_stats(toi, filep/reg, plot_par=True, features= features, skip_eval=skip)
pred_vs_actual(y_test,y_pred, ymax, filep/reg)
