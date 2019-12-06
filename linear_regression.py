from methods import Regression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from  tqdm import tqdm
from sklearn.linear_model import BayesianRidge 
from BaysianOptimizer import BaysianMaximization
from sklearn.metrics import mean_squared_error

def grid_searc_lam(lambdas, bootstraps = 10, reg_type = 'LinearRegression'):
    df = df = pd.DataFrame(columns=["lam","MSE", "R2","set"])
    for i in tqdm(range(len(lambdas))):
        for j in range(bootstraps):
            regression = Regression()
            #data workflow
            regression.importData('./Data')
            regression.scale()

            #creat Designmatrix
            regression.generate_polynomic_features(order =1)
            #train test split, evaluation set only necessary when hyperpar tuning
            regression.train_test_eval_split(test_size =0.2, eval_size=0)

            #fit OLS
            MSE,R2 = regression.linear(reg_type)
            df = df.append({'lam': lambdas[i],'MSE':MSE*regression.ymax**2, "R2": R2, "set":"train"}, ignore_index=True)
            #evaluate the fit
            MSE, R2 = regression.evaluation()
            df = df.append({'lam': lambdas[i],'MSE':MSE*regression.ymax**2, "R2": R2, "set":"test"}, ignore_index=True)
            del regression

    df["MSE"] = np.sqrt(df["MSE"])
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    sns.lineplot(x ='lam',y='MSE',hue='set', data =df)
    plt.xlabel("$\lambda$", fontsize = 28)
    plt.ylabel(r"$\sigma(T_{c})$ in K", fontsize =28)
    plt.legend(loc='best', fontsize =24,title =None)
    plt.tick_params(labelsize=22, size = 20)
    plt.subplot(1,2,2)
    sns.lineplot(x= 'lam',y='R2',hue='set', data=df)
    plt.xlabel("$\lambda$", fontsize = 28)
    plt.ylabel("R2", fontsize =28)
    plt.legend(loc='best', fontsize =24, title = None)
    plt.tick_params(labelsize=22, size = 20)
    plt.tight_layout()
    plt.savefig('./Results/'+ reg_type +'/score_no_features.pdf')

    df = df[df['set']=='test']
    grouped = df.groupby("lam").mean()
    grouped.reset_index(inplace = True)
    return df.iloc[grouped["MSE"].idxmin()]
"""
#grid search
lam = np.linspace(1e-9, 1e-1,20)
for reg in ['Ridge', 'LASSO']:
    print(reg)
    print(grid_searc_lam(lam, reg_type=reg))



lams = np.zeros(20)
av_lam = 0
av_MSE = 0
for i in range(20):
    #compar own BaysOpt vs sklearn Baysopt for Ridge
    regression = Regression()
    #data workflow
    regression.importData('./Data')
    regression.scale()
    regression.generate_polynomic_features(order =1)
    regression.train_test_eval_split()

    def ridge_opt(X_train, X_test, y_train, y_test, lam = 0.01, regressor = regression):

        regression.linear('LASSO', lam = lam)
        MSE, _ = regression.evaluation()
        return -1* MSE

    optimizer = BaysianMaximization(ridge_opt, {},{'lam':(0,1)})
    optimizer.train = [regression.X, regression.X_test, regression.y, regression.y_test]
    optimizer.eval = [regression.X, regression.X_eval, regression.y, regression.y_eval]
    optimizer.InitialGuesses(100)
    optimizer.OptimizeHyperPar(cycles=50, exploration= 0.01)

    MSE_opt = ridge_opt(*optimizer.eval, **optimizer.best_model_kargs)
    print("unc. T_c: +-", np.sqrt(-MSE_opt)*regression.ymax)
    print("Optimal lam:", optimizer.best_model_kargs['lam'])

    lams[i] = optimizer.best_model_kargs['lam']
    av_lam += lams[i] /20
    av_MSE += MSE_opt/20
    lam = np.linspace(0,1,400).reshape((-1,1))
    p, sd = optimizer.Predict(lam)
    lam = lam.flatten()
    f = plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.plot(lam, np.sqrt(-p)*regression.ymax, color='tab:orange', label ='GP' )
    #plt.fill_between(lam, np.sqrt(-p - sd)*regression.ymax, np.sqrt(-p +sd)*regression.ymax, color = 'tab:orange', alpha = 0.4)
    plt.scatter(optimizer.hyperpar.flatten(), np.sqrt(-optimizer.model_score)*regression.ymax, label='Samples')
    plt.xlabel("$\lambda$", fontsize = 28)
    plt.ylabel(r"$\sigma(T_{c})$ in K", fontsize =28)
    plt.legend(loc='best', fontsize =24,title =None)
    plt.tick_params(labelsize=22, size = 20)

    plt.subplot(1,2,2)
    plt.hist(optimizer.hyperpar.flatten())
    plt.xlabel("$\lambda$", fontsize = 28)
    plt.ylabel(r"Count", fontsize =28)
    plt.tick_params(labelsize=22, size = 20)

    plt.tight_layout()
    plt.savefig('Results/LASSO/bays_opt.pdf')
    del f

plt.figure(figsize=(10,10))
plt.hist(lams)
plt.xlabel("$\lambda$", fontsize = 28)
plt.ylabel(r"Count", fontsize =28)
plt.tick_params(labelsize=22, size = 20)

plt.tight_layout()
plt.savefig('Results/LASSO/bays_opt_lam.pdf')
print("averages")
print("unc. T_c: +-", np.sqrt(-av_MSE)*regression.ymax)
print("Optimal lam:", av_lam)

sk_opt = BayesianRidge()
sk_opt.fit(regression.X, regression.y)
pred = sk_opt.predict(regression.X_eval)
MSE_sk = mean_squared_error(regression.y_eval, pred)
print("unc. T_c: +-", np.sqrt(MSE_sk)*regression.ymax)
print("Optimal pars:", sk_opt.get_params())
"""

av_MSE = 0
av_R2=0
for  i in range(20):
    regression = Regression()
    #data workflow
    regression.importData('./Data')
    regression.scale()
    regression.generate_polynomic_features(order =1)
    regression.train_test_eval_split(eval_size=0)
    regression.linear('LinearRegression')
    MSE, R2 = regression.evaluation()
    av_MSE += MSE/20
    av_R2 += R2/20

print("unc. T_c: +-", np.sqrt(av_MSE)*regression.ymax)
print("R2:", av_R2)

