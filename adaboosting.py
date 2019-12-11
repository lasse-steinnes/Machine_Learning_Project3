# -*- coding: utf-8 -*-
"""
Script for Adaboost class
"""
from helper_functions import scaler, MSE, importData
from sklearn.model_selection import train_test_split
from sklearn import tree
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

class AdaBoost:
    
    def __init__(self, iterations, depth, X_train, y_train, X_test, y_test, X_eval = None , y_eval = None):
        '''
        Initialise the parameters of the AdaBoost.
        
        Inputs:
            iterations : int
            The number of iterations of the boost
            depth : int
            The maximum depth of the tree one would like to grow.
            X_train, X_eval, X_test : (n_samples, n_features)
            y_train, y_eval, y_test : (n_samples, 1)
        Returns:
            Nothing
            
        '''
        self.iterations = iterations
        self.depth = depth
        self.X_train, self.y_train, self.X_test, self.y_test = X_train, y_train, X_test, y_test
        self.X_eval, self.y_eval = X_eval, y_eval
        self.n = self.X_train.shape[0]
        
    #define the loss functions for adaboost
    def linear (self, y_predict, y):
        loss = np.absolute(y_predict - y)
        return loss / np.amax(loss)
    
    def exponential(self, y_predict, y):
        loss = np.absolute(y_predict - y)
        return 1 - np.exp(- loss  / np.amax(loss))
        
    def square(self, y_predict, y):
        loss = (y_predict - y)**2
        return loss / np.amax(loss)
    
    def prediction_loss(self, y, p):
        '''
        A function for finding the loss using on of the loss functions below.
        
        Inputs:
            y : (n_samples, 1)
            The y data, this could be train or test.
            p : (n_samples, 1)
            The prediction data
        Returns:
            loss: (n_samples,1)
            The calculated loss
        '''
        
        if self.loss_func == 'linear':
            loss = AdaBoost.linear(self, p, y)
        elif self.loss_func == 'square':
            loss = AdaBoost.square(self, p, y)
        elif self.loss_func == 'exponential':
            loss = AdaBoost.exponential(self, p, y)
            
        return loss
    
    def training(self, loss_func):
        '''
        Function to train or begin the adaboost iterative process. 
        
        Inputs:
        loss_func: - string 
        loss function is 'square', 'linear' or 'exponential'.
        
        Returns:
        loss (and 'test_loss'): (n_samples,1)
        Returns the loss depending on which loss function is calculated from the training set and y.
        '''
        self.loss_func = loss_func
        W = np.ones(self.n) # initialise sample weights as 1.0
        self.test_predict_iter = np.zeros(len(self.y_test))
        self.train_predict_iter = np.zeros(len(self.y_train))
        
        if type(self.X_eval) != type(None):
            self.eval_predict = np.zeros(len(self.y_eval))
        
        self.iteration_weight = np.zeros(self.iterations)
        self.trees = []
        self.beta = np.zeros(self.iterations)
        self.loss = np.zeros((self.iterations, len(self.y_train)))
        self.test_loss = np.zeros((self.iterations, len(self.y_test)))
        train_mask = np.ones(self.iterations, dtype = bool)
        test_mask = np.ones(self.iterations, dtype = bool)
        for i in range(0, self.iterations): 
            #normalise the weights
            W_norm = W / np.sum(W)
            
            # fit a weak decision tree
            reg_weak = tree.DecisionTreeRegressor(max_depth = self.depth)
            reg_weak.fit(self.X_train, self.y_train, sample_weight = W_norm)
            
            # predict on train and test
            train_predict = reg_weak.predict(self.X_train)
            test_predict = reg_weak.predict(self.X_test)
            
            loss = AdaBoost.prediction_loss(self, train_predict, self.y_train)
            test_loss = AdaBoost.prediction_loss(self, test_predict, self.y_test)
            self.loss[i] = loss
            self.test_loss[i] = test_loss
            #find the average loss and update sample weights
            loss_ave = np.sum(loss * W_norm)
            
            #stop learning 
            if loss_ave >= 0.5:
                print ('breaking Adaboost')
                print(len(test_mask))
                test_mask[[i]] = False
                train_mask[[i]] = False
                self.iterations = i - 1
                self.loss = self.loss[train_mask]
                self.test_loss = self.test_loss[test_mask]
                
                return  self.loss, self.test_loss
            
            beta = loss_ave / (1.0 - loss_ave)
            self.iteration_weight[i] = np.log(1.0 / beta)
            self.beta[i] = beta
            self.test_predict_iter += beta* test_predict
            self.train_predict_iter += beta* train_predict
            #update weights
            W = W_norm * (beta**(1-loss))
            self.trees.append(reg_weak)

    def evaluate(self, X, y):
        '''
        This function finds the ensemble prediction and returns the mse
        and r2 score of this prediction.
        
        Inputs:
            X: (n_samples, n_features)
            The X matrix to be used in the ensemble prediction. Usually
            the training X would be passed here, or evaluation if used.
            y: (n_samples,1)
        Returns:
            median_predict: (n_samples,1)
            The ensemble prediction
            MSE : float
            r2 : float
        '''
        prediction = []
        
        for i in range(0, self.iterations):
            prediction.append(self.trees[i].predict(X))
        prediction = np.array(prediction).T
        ordered_matrix_idx = np.argsort(prediction, axis = 1)
        
        iteration_weight_cumu = np.cumsum(self.iteration_weight[ordered_matrix_idx], axis = 1)
        max_cumu = iteration_weight_cumu[:, -1][:, np.newaxis]
        median_true = iteration_weight_cumu >= 0.5 * max_cumu
        median_idx = median_true.argmax(axis=1)
        
        median_iteration = ordered_matrix_idx[np.arange(X.shape[0]), median_idx]
        median_predict = prediction[np.arange(X.shape[0]), median_iteration]
        '''
        for i in range(0, X.shape[0]):
            median_iteration = ordered_matrix_idx[i, median_idx[i]]
            median_predict = prediction[i, median_iteration]
        '''
        
        MSE, R2 = AdaBoost.calcMSE_R2(self, median_predict, y)
        
        return median_predict, MSE, R2
    
    def calcMSE_R2(self, p, y):
        '''
        A function for calculate the mean squared error and r2 score
        
        Inputs:
            p : (n_samples,1)
            the prediction
            y: (n_samples,1)
            the y values
        Returns:
            MSE : float
            R2: float
        '''
        MSE = mean_squared_error(y, p)
        R2 = r2_score(y, p)
        
        return MSE, R2
      
            
    def shuffleAndsplit(self, X, y):
        '''
        A function for shuffling and splitting data. This 
        function is no longer used. Now the data is split 
        in the Regression class in methods.py
        '''
        curr_seed= 0
        np.random.seed(curr_seed)
        np.random.shuffle(X)
        np.random.seed(curr_seed)
        np.random.shuffle(y)
        
        X = X[0:1000] #algorithm testing with smaller samples than full data
        y = y[0:1000]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
        
        return X_train, X_test, y_train, y_test



    

            
