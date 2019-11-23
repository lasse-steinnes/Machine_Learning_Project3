# -*- coding: utf-8 -*-
"""
Script for running boosting
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import graphviz 

data_path = Path("./Data/")
file = []
for files in data_path.glob('*.csv'):
    temp = pd.read_csv(files)
    file.append(temp)
    
X_temp = file[0]
chemical_comp_temp = file[1]

X = X_temp.drop(columns =['critical_temp'])
y = X_temp['critical_temp']

X = X[0:1000] #algorithm testing with smaller samples than full data
y = y[0:1000]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.1)

def tree_normal():
    tree_reg= tree.DecisionTreeRegressor(max_depth = 3)
    tree_reg.fit(X_train, y_train)
    y_predict = tree_reg.predict(X_test)
    y_train_predict = tree_reg.predict(X_train)
    
    #print("Train set R2 score is: {:.2f}".format(tree_reg.score(X_train,y_train)))
    print("Test set R2 score is: {:.2f}".format(tree_reg.score(X_test,y_test)))
    
    mse_predict = MSE(y_test, y_predict)
    print('Test set mse is: {:.2f}'.format(mse_predict))
    
    print('The number of leaves in the decision tree is:',tree_reg.get_n_leaves())
    
def tree_scaled():
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    tree_reg= tree.DecisionTreeRegressor(max_depth = 3)
    tree_reg.fit(X_train_scaled, y_train)
    y_train_predict = tree_reg.predict(X_train_scaled)
    
    y_predict = tree_reg.predict(X_test_scaled)
    print("Test set R2 score is: {:.2f}".format(tree_reg.score(X_test_scaled,y_test)))
    
    mse_predict = MSE(y_test, y_predict)
    print('Test set mse is: {:.2f}'.format(mse_predict))
    
    print('The number of leaves in the decision tree is:',tree_reg.get_n_leaves())
    
def boosting():
    reg_weak = tree.DecisionTreeRegressor(max_depth = 2)
    k = 10 # the number of Adaboost rounds
    n = X_train.shape[0]
    W = [1/n for i in range(0,n)]
    W = np.array(weights)
    
    for r in range(1,k):
        print ('r is:', r)
        W_norm = W / np.sum(W)
        
        reg_weak.fit(X_train, y_train, sample_weight = W_norm)
        y_predict = reg_weak.predict(X_train)
        
        err = np.absolute(y_predict - y_train)
        err_sum = np.sum(err)
        loss = err/ err_sum
        loss_ave = np.sum(loss*W)
        
        if loss_ave > 0.5:
            break
        
        beta = loss_ave / (1-loss_ave)
        W = W * (beta **(1-loss_ave))
        
def MSE(data, model):
    """
    Calculates the Mean Squared Error if both data and model are vectos
    Calculates Variance if data is vector and model is the mean value of the data
    """
    n = np.shape(data)[0]
    res = np.array(data - model)
    return (1.0/n) *(res.T.dot(res))