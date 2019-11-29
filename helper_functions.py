# -*- coding: utf-8 -*-
"""
script containing helper functions
"""

from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from pathlib import Path

def importData():
    data_path = Path("./Data/") # data should be stored in folder Data
    file = []
    for files in data_path.glob('*.csv'):
        temp = pd.read_csv(files)
        file.append(temp)
        
    X_temp = file[3]
    chemical_comp_temp = file[1]
        
    X = X_temp.drop(columns =['critical_temp'])
    y = X_temp['critical_temp']

    X = X.to_numpy()
    y = y.to_numpy()
    
    return X, y
        
def scaler(X_train, X_test):
    '''
    Function for scaling test and training data. First the function fits
    to the training data and applies the same transformation to 
    training and test data.
    
    Returns scaled data between 0 and 1.
    
    X_train: - (n_samples, n_features)
    X_test:  - (n_samples, n_features)
    '''
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled

def MSE(data, model):
    """
    Calculates the Mean Squared Error if both data and model are vectos
    Calculates Variance if data is vector and model is the mean value of the data
    """
    n = np.shape(data)[0]
    res = np.array(data - model)
    return (1.0/n) *(res.T.dot(res))