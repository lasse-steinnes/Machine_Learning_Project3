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

df = pd.read_csv('ave_toi.csv')
df.rename(columns={"iter":"number of iterations"} , inplace=True)
#mse plot
g = sns.FacetGrid(df[ df["data set"] == 'test'], col ='loss function', margin_titles=True, sharey = False)
g.map_dataframe(draw_heatmap, 'number of iterations',Y,'MSE', vmin = df["MSE"].min(), vmax = df["MSE"].max(),  aggregate = 'min')
plt.show()
