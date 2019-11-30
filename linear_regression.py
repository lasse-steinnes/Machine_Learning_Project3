from methods import Regression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from  tqdm import tqdm

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
    plt.savefig('./Results/'+ reg_type +'/score.pdf')

    df = df[df['set']=='test']
    grouped = df.groupby("lam").mean()
    grouped.reset_index(inplace = True)
    return df.iloc[grouped["MSE"].idxmin()]

lam = np.linspace(1e-9, 1e-1,20)
for reg in ['Ridge', 'LASSO']:
    print(reg)
    print(grid_searc_lam(lam, reg_type=reg))
