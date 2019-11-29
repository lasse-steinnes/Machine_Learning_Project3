# -*- coding: utf-8 -*-
"""
script containing helper functions
"""

from sklearn.preprocessing import StandardScaler


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