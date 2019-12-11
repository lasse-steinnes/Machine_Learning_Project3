import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import csv 

path = Path('./')

def extract(path, av_lam=False):
    #get headers from train
    path_to_columns = Path('/home/lukas/Documents/Machine_Learning_Project3/Data/train.csv')
    unique_headers = set()
    with open(path_to_columns, 'r') as fin:
        csvin = csv.reader(fin)
        unique_headers.update(next(csvin, []))
    unique_headers = list(unique_headers)
    for fold in ['LinearRegression', 'Ridge', 'LASSO']:
        if fold == 'LinearRegression':
            set_filter ='test'
        else:
            set_filter ='eval'

        f = open(path/fold/'stats.txt', 'a')
        try:
            df = pd.read_csv(path/fold/'toi.csv')
        except:
            continue
        if av_lam:
            lam = df["lambda"].mean()
            std_lam= df["lambda"].std()
            f.write("average lambda: %.9f +- %.9f\n\n"%(lam, std_lam))
            
        idx = df[df["data set"]==set_filter]["MSE"].idxmin()
        best = df.iloc[idx]
        np_par = best.filter(regex="par.*").to_numpy()
        np_par_args = np.abs(np_par).argsort()
        f.write("Most importand parameters based on best model:\n")
        for i in [-1,-2,-3]:
            f.write("%i. importand parameter: %s\n"%(-i, unique_headers[np_par_args[i] - 1])) #parameter shifted by -1 because const. parameter not in name!
        f.close()
def comp_search(path, toi =None):
    if toi==None:
        toi = pd.DataFrame(columns=["MSE", "lambda", "Regression"])
        for fold in [ 'Ridge', 'LASSO']:
            df = pd.read_csv(path/fold/'toi.csv')
            df = df[df["data set"]!="train"][['MSE', 'lambda']]
            df['Regression']=fold
            toi = toi.append(df, ignore_index = True)
    return toi
#extract(path)
#extract(path/'BaysianOpt', av_lam=True)

toi = comp_search(path)
toi["Search"] ='Grid'
toi1 = comp_search(path/'BaysianOpt')
toi1["Search"] ='Bayes'
toi = toi.append(toi1)

sns.set(font_scale=3, style='white', context='paper')
g = sns.FacetGrid(toi, col='Regression', hue='Search',margin_titles=True,height=7, aspect=1, sharey=True, legend_out=True)
g.map(sns.lineplot, 'lambda', 'MSE',ci='sd', legend='brief' ).set_axis_labels("$\lambda$")
g.add_legend()
plt.xscale('log')
plt.yscale('log')
plt.savefig("search.pdf")
