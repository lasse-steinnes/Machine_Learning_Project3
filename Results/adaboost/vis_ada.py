# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 17:42:28 2019

@author: anjat
"""
#df = pd.read_csv('toi.csv')

import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import sys 
from pathlib import Path
#font size controles
SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20
sns.set_context("paper", rc={"font.size":MEDIUM_SIZE,"axes.titlesize":MEDIUM_SIZE,"axes.labelsize":MEDIUM_SIZE, 'legend':MEDIUM_SIZE})

Y = 'depth'

def draw_heatmap(*args, **kwargs):
    """
    heatmap function for FacetGrid
    """
    data = kwargs.pop('data')
    agg_func = kwargs.pop('aggregate')
    if agg_func =='max':
        data = data.groupby([args[0], args[1]], sort=False)[args[2]].max().reset_index()
    elif agg_func =='min':
         data = data.groupby([args[0], args[1]], sort=False)[args[2]].min().reset_index()
    cmap = sns.light_palette("navy", reverse=True)
    d = data.pivot(index=args[1], columns=args[0], values=args[2])
    f =sns.heatmap(d, cmap = cmap, annot = True,cbar = False, fmt ='.2f', annot_kws={'size':SMALL_SIZE},  **kwargs)
    return f

def Grid():
    df.rename(columns={"iter":"number of iterations"} , inplace=True)
    #mse plot
    g = sns.FacetGrid(df[ df["data set"] == 'test'], col ='loss function',  margin_titles=True, sharey = False)
    g.map_dataframe(draw_heatmap, 'number of iterations',Y,'MSE', vmin = df["MSE"].min(), vmax = df["MSE"].max(),  aggregate = 'min')
    plt.show()


def Grid_test_train(filepath):
    
    df_f = df['data set'] != 'evaluation'
    g = sns.FacetGrid(df[df_f], col = 'iter', row="loss function",hue = 'data set', sharey = False, sharex = False, legend_out = True)
    #g.figure(figuresize = (10,10))
    g = g.map(plt.plot, "depth", "MSE")
    g.add_legend(fontsize = 15)
    '''
    train_filter = df['data set'] == 'train'
    test_filter = df['data set'] == 'test'
    sns.lineplot(data = df[train_filter], x = "depth", y = "MSE")
    sns.lineplot(data = df[test_filter], x = "depth", y = "MSE")
    '''
    plt.figure(figsize=(10, 10))
    plt.savefig(filepath/'test_train.pdf')
    plt.show()

filepath = Path("./")
df = pd.read_csv('ave_toi.csv')
Grid_test_train(filepath)


