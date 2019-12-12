from methods import Regression
from helper_functions import DataWorkflow, CV
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from  tqdm import tqdm
from sklearn.linear_model import BayesianRidge 
from BaysianOptimizer import BaysianMaximization
from sklearn.metrics import mean_squared_error
from pathlib import Path

def GridCV(X,y,Xtest, ytest, lam_path, folds, toi, fnum, reg, reg_type):
    """
    grid search 
    using in outer CV
    X, Xtest shaped (samples, features)
    y, ytest shaped (samples,)
    lam_path lambdas to test Ridge and LASSO (list)
    folds number of CV folds on X, y -> Xtest ytest becomes evaluation data
    toi tabe of information to append results to
    fnum fit number for toi
    reg instance of Regression class
    reg_type regression type, str either 'LASSO' or 'Ridge'
    """
    features = X.shape[1]
    for lam in lam_path:
                Xinner_train, Xinner_test, yinner_train, yinner_test = CV(X,y, folds =folds)
                for j in range(folds//2):
                    reg.X = Xinner_train[j]
                    reg.y = yinner_train[j]

                    
                    MSE, R2 = reg.linear(reg_type,lam = lam)
                    d = {"fit num":fnum,"MSE": MSE, "R2":R2, "unc. Tc": np.sqrt(MSE)*ymax, "lambda":lam, "data set": "train"}
                    d.update({"par%i"%k:reg.weights[k] for k in range(features)})
                    toi = toi.append(d, ignore_index=True)

                    MSE, R2 = reg.evaluation(Xinner_test[j], yinner_test[j])
                    d = {"fit num":fnum,"MSE": MSE, "R2":R2, "unc. Tc": np.sqrt(MSE)*ymax, "lambda":lam, "data set": "test"}
                    d.update({"par%i"%k:reg.weights[k] for k in range(features)})
                    toi =toi.append(d, ignore_index=True)

                    MSE, R2 = reg.evaluation(Xtest, ytest)
                    d = {"fit num":fnum,"MSE": MSE, "R2":R2, "unc. Tc": np.sqrt(MSE)*ymax, "lambda":lam, "data set": "eval"}
                    d.update({"par%i"%k:reg.weights[k] for k in range(features)})
                    toi = toi.append(d, ignore_index=True)
                    fnum +=1 
    return toi, fnum

def BayseCV(X,y, Xtest,ytest,lam_range, folds, toi, fnum, reg, reg_type, bootstraps=10):
    """
    Bayesian optimization search 
    using in outer CV
    X, Xtest shaped (samples, features)
    y, ytest shaped (samples,)
    lam_range touple of lower and upper lambda
    folds number of CV folds on X, y -> Xtest ytest becomes evaluation data
    toi tabe of information to append results to
    fnum fit number for toi
    reg instance of Regression class
    reg_type regression type, str either 'LASSO' or 'Ridge'
    bootstraps number of bootstraps to perform for each inner fold
    """
    features = X.shape[1]
    Xinner_train, Xinner_test, yinner_train, yinner_test = CV(X,y, folds =folds)

    def lin(Xtrain, Xtest, ytrain, ytest, lam = 0.1, reg=reg, reg_type = reg_type):
            reg.X = Xtrain
            reg.y = ytrain
            MSE,_ = reg.linear(reg_type,lam =lam)
            return -MSE

    for j in range(folds):     
        for boot in range(bootstraps):
            opt = BaysianMaximization(lin, {}, {'lam':lam_range})
            opt.train=[Xinner_train[j],Xinner_test[j], yinner_train[j], yinner_test[j]]
            opt.eval =[Xinner_train[j],Xtest, yinner_train[j], ytest]
            opt.InitialGuesses(50)
            opt.OptimizeHyperPar(cycles=50)
            lam = opt.best_model_kargs["lam"]

            MSE, R2 = reg.linear(reg_type,lam = lam)
            d = {"fit num":fnum,"MSE": MSE, "R2":R2, "unc. Tc": np.sqrt(MSE)*ymax, "lambda":lam, "data set": "train"}
            d.update({"par%i"%k:reg.weights[k] for k in range(features)})
            toi = toi.append(d, ignore_index=True)

            MSE, R2 = reg.evaluation(Xinner_test[j], yinner_test[j])
            d = {"fit num":fnum,"MSE": MSE, "R2":R2, "unc. Tc": np.sqrt(MSE)*ymax, "lambda":lam, "data set": "test"}
            d.update({"par%i"%k:reg.weights[k] for k in range(features)})
            toi =toi.append(d, ignore_index=True)

            MSE, R2 = reg.evaluation(Xtest, ytest)
            d = {"fit num":fnum,"MSE": MSE, "R2":R2, "unc. Tc": np.sqrt(MSE)*ymax, "lambda":lam, "data set": "eval"}
            d.update({"par%i"%k:reg.weights[k] for k in range(features)})
            toi = toi.append(d, ignore_index=True)
            fnum +=1  
    return toi, fnum          
                   



def LinerRegressionCV(X, y, ymax,reg_type='LinearRegression', folds =10, lambda_path=None, bayse=False):
    """
    Perfomrs corss validation for linear regression, Ridge, and LASSO regression with hyperparameter tuning (grid search)
    If lambda_path is provided a nested CV of folds*folds/2 is performed for each element in lambda_path
    kargs:
        reg_type:   either LineraRegression, Ridge or LASSO
        folds:      number of cross validations
        lambda_path list of lambdas to test
    returns tabel of information (toi) with all scores, parameters and hyperparameter lambda
    """
    features = X.shape[1]
    reg = Regression()
    toi = pd.DataFrame(columns=["fit num","MSE","R2","unc. Tc", "lambda", "data set",] + ["par%i"%i for i in range(features)])

    Xtrain, Xtest, ytrain, ytest = CV(X,y, folds =folds)
    fnum = 0
    for i in tqdm(range(folds)):

        if type(lambda_path) == list:
            if bayse:
                toi, fnum = BayseCV(Xtrain[i],ytrain[i], Xtest[i], ytest[i], lambda_path, folds//2, toi, fnum, reg, reg_type)
            else:
                toi, fnum = GridCV(Xtrain[i],ytrain[i], Xtest[i], ytest[i], lambda_path, folds//2, toi, fnum, reg, reg_type)
            
        else:

            reg.X = Xtrain[i]
            reg.y = ytrain[i]

            MSE, R2 = reg.linear(reg_type)
            d = {"fit num":fnum,"MSE": MSE, "R2":R2, "unc. Tc": np.sqrt(MSE)*ymax, "lambda":-1, "data set": "train"}
            d.update({"par%i"%k:reg.weights[k] for k in range(features)})
            toi = toi.append(d, ignore_index=True)

            MSE, R2 = reg.evaluation(Xtest[i], ytest[i])
            d = {"fit num":fnum,"MSE": MSE, "R2":R2, "unc. Tc": np.sqrt(MSE)*ymax, "lambda":-1, "data set": "test"}
            d.update({"par%i"%k:reg.weights[k] for k in range(features)})
            toi =toi.append(d, ignore_index=True)
            fnum +=1

    return toi

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
        
        for name in ["MSE", "R2", "unc. Tc", "lambda"]:
            
            f.write(name + ': %.9f\n'%tabel[data_set][name])
            
        f.write("\n")
    #model variablity at best lam    
    if not bayse:
        toi = toi[toi["lambda"]==tabel["test"]["lambda"]]

    av = toi.groupby("data set").agg(["mean","std"])

    f.write('Average model:\n')
    for data_set in ["train","test", "eval"]: 
        if skip_eval & (data_set =='eval'):
            break
        f.write(data_set +' CV reults: \n')
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
        plt.savefig(filepath/'params.pdf')

    return tabel["test"][inds3]

def perd_vs_actual(X,y,ymax, weights, filepath):
    """
    plot of predicted vs actual Tc
    """
    p = np.dot(X, weights)
    plt.figure(figsize=(10,10))
    plt.scatter(y*ymax, p*ymax)
    plt.plot([0,ymax],[0,ymax], linestyle ='--')
    plt.ylabel('act. T$_c$ in K', fontsize =32)
    plt.xlabel("pred. T$_c$ in K ", fontsize= 32)
    plt.xlim(0, ymax)
    plt.ylim(0, ymax)
    plt.tick_params(size =24, labelsize=26)
    plt.tight_layout()
    plt.savefig(filepath/'pred_vs_act.pdf')


filep = Path("./Results/")
lam_path = [10**(-i) for i in range(7)] + [2*10**(-i) for i in range(1,7)] + [4*10**(-i) for i in range(1,7)] + [6*10**(-i) for i in range(1,7)] +[8*10**(-i) for i in range(1,7)] 
print(lam_path)
lam = [None, lam_path, lam_path]
skip = [True,False, False]
X,y,ymax = DataWorkflow()

#creat const. feature
ones = np.ones((X.shape[0],1))
X = np.hstack([ones,X])

features = X.shape[1]

for i,reg in enumerate(["LinearRegression", "Ridge", "LASSO"]):
    print(reg)
    print("Grid")
    toi = LinerRegressionCV(X,y,ymax, reg_type=reg, folds =10, lambda_path=lam[i])
    toi.to_csv(filep/reg/'toi.csv')
    best_par = Stats(toi, filep/reg, plot_par=True, features= features, skip_eval=skip[i])
    perd_vs_actual(X[::50], y[::50], ymax , best_par, filep/reg)
    if(reg=='LinearRegression'):
        continue
    print("Bayse")
    toi = LinerRegressionCV(X,y,ymax, reg_type=reg, folds =10, lambda_path=[0,1], bayse=True)
    toi.to_csv(filep/'BaysianOpt'/reg/'toi.csv')
    best_par = Stats(toi, filep/'BaysianOpt'/reg, plot_par=True, features= features, skip_eval=skip[i], bayse=True)
    perd_vs_actual(X[::50], y[::50], ymax , best_par, filep/'BaysianOpt'/reg)
