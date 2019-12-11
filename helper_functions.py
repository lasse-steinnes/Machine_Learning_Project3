# -*- coding: utf-8 -*-
"""
script containing helper functions
"""

from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from pathlib import Path
from methods import Regression

def DataWorkflow(file_path =Path("./Data")):
    data = Regression()
    data.importData(file_path)
    data.scale()
    X, y = data.X, data.y
    np.random.seed(2019)
    np.random.shuffle(X)
    np.random.seed(2019)
    np.random.shuffle(y)
    return X, y, data.ymax

def CV(X, y, folds = 10):
    """
    X is shape (samples,features)
    y is shapes (samples,)
    """
    Xtest = np.array_split(X, folds, axis = 0)
    ytest = np.array_split(y, folds, axis = 0)

    Xtrain =[ np.vstack(Xtest[:i]+Xtest[i+1:]) for i in range(folds)]
    ytrain =[ np.hstack(ytest[:i]+ytest[i+1:]) for i in range(folds)]
    return Xtrain, Xtest, ytrain, ytest





def DataWorkflow(file_path =Path("./Data")):
    data = Data()
    data.importData(file_path)
    data.scale()
    X, y = data.X, data.y
    np.random.seed(2019)
    np.random.shuffle(X)
    np.random.seed(2019)
    np.random.shuffle(y)
    return X, y, data.ymax

def CV(X, y, folds = 10):
    """
    X is shape (samples,features)
    y is shapes (samples,)
    """
    Xtest = np.array_split(X, folds, axis = 0)
    ytest = np.array_split(y, folds, axis = 0)

    Xtrain =[ np.vstack(Xtest[:i]+Xtest[i+1:]) for i in range(folds)]
    ytrain =[ np.hstack(ytest[:i]+ytest[i+1:]) for i in range(folds)]
    return Xtrain, Xtest, ytrain, ytest

class Data:
    def importData(self, filepath):
        """
        Imports training data train.csv from filepath
        sets X, y numpy arrays
        """
        data_path = Path(filepath) # data should be stored in folder Data
        df = pd.read_csv(data_path/'train.csv')
            
        self.y = df["critical_temp"].to_numpy()
        self.X = df.drop(columns = ["critical_temp"]).to_numpy()
    
    def scale(self):
        """
        scales X according to standard scaler
        scales y to [0,1] and keeps ymax for reversed scaling of prediction
        """
        self.scaled = True
        X_scale = StandardScaler()
        self.X = X_scale.fit_transform(self.X)
        self.ymax = self.y.max()
        self.y /= self.ymax

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