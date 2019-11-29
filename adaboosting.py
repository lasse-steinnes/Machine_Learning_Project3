# -*- coding: utf-8 -*-
"""
Script for Adaboost class
"""
from helper_functions import scaler, MSE, importData
from sklearn.model_selection import train_test_split
from sklearn import tree
import numpy as np
import pandas as pd

class Adaboost:
    
    def __init__(self, iterations, depth):
        
        X, y = importData()
        self.iterations = iterations
        self.depth = depth
        
        X_train, X_test, self.y_train, self.y_test = Adaboost.shuffleAndsplit(self, X, y)
        self.X_train, self.X_test = scaler(X_train, X_test)
        self.n = X_train.shape[0]
    
    def initiateBoost(self, loss_func):
        '''
        initiates the adaboost
        
        loss func:  - loss function is a string, 'square', 'linear' or 'exponential'.
        '''
        W = np.array([1 for i in range(0, self.n)]) # initialise the weights as 1/n
        self.beta = [] #list of all betas
        self.y_p = [] #list of all training prediction arrays 
        self.test_p = np.array([0.0 for i in range(0,len(self.y_test))])
        self.training_p = np.array([0.0 for i in range(0,len(self.y_train))])
        
        for i in range(0, self.iterations+1): 
            # renormalise the weights
            W_norm = W / np.sum(W)
            
            # fit a weak decision tree
            reg_weak = tree.DecisionTreeRegressor(max_depth = self.depth)
            reg_weak.fit(self.X_train, self.y_train, sample_weight = W_norm)
            
            # predict on train and test
            y_predict_train = reg_weak.predict(self.X_train)
            self.y_p.append(y_predict_train)
            y_predict_test = reg_weak.predict(self.X_test)
            
            # decide on loss function
            if loss_func == 'linear':
                loss = Adaboost.linear(self, y_predict_train, self.y_train)
            elif loss_func == 'square':
                loss = Adaboost.square(self, y_predict_train, self.y_train)
            elif loss_func == 'exponential':
                loss = Adaboost.exponential(self, y_predict_train, self.y_train)
            
            #find weighted loss and update prediction
            loss_ave = np.sum(loss * W_norm)
            beta = loss_ave / (1-loss_ave)
            self.beta.append(beta)
            self.test_p += beta* y_predict_test
            self.training_p += beta* y_predict_train
            
            #breaking function - not well understood
            if loss_ave > 0.5:
                print ('breaking Adaboost')
                break
            
            #update weights
            W = W_norm * (beta**(1-loss))
    
    def ensemble_predict(self, beta_stopping = False):
        if beta_stopping == True:
            max_iteration = Adaboost.beta_eval(self)
        else:
            max_iteration = self.iterations
            
        ensemble_y = 0
        for i in range(0, max_iteration):
            weight = self.y_p[i] * self.beta[i]
            ensemble_y += weight
        
        return ensemble_y
            
    def shuffleAndsplit(self, X, y):
        curr_seed= 0
        np.random.seed(curr_seed)
        np.random.shuffle(X)
        np.random.seed(curr_seed)
        np.random.shuffle(y)
        
        X = X[0:1000] #algorithm testing with smaller samples than full data
        y = y[0:1000]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
        
        return X_train, X_test, y_train, y_test

    def tree_normal(self):
        tree_reg= tree.DecisionTreeRegressor(max_depth = 3)
        tree_reg.fit(self.X_train, self.y_train)
        y_predict = tree_reg.predict(self.X_test)
        y_train_predict = tree_reg.predict(self.X_train)
        
        #print("Train set R2 score is: {:.2f}".format(tree_reg.score(X_train,y_train)))
        print("Test set R2 score is: {:.2f}".format(tree_reg.score(self.X_test, self.y_test)))
        
        mse_predict = MSE(self.y_test, y_predict)
        print('Test set mse is: {:.2f}'.format(mse_predict))
        print('The number of leaves in the decision tree is:',tree_reg.get_n_leaves())
    
    #loss functions for boost
    def linear (self, y_predict, y):
        return np.absolute(y_predict - y) / np.sum(np.absolute(y_predict - y))
    
    def exponential(self, y_predict, y):
        return 1 - np.exp(- np.absolute(y_predict - y) / np.sum(np.absolute(y_predict - y)))
        
    def square(self, y_predict, y):
        return (y_predict - y)**2 / np.sum((y_predict - y)**2)
    
    # step number 8 in Drucker1997
    def beta_eval(self):
        beta_l_tot = 0
        beta_log = []
    
        for beta in self.beta:
            beta_l = np.log(1/beta)
            beta_log.append(beta_l)
            beta_l_tot += beta_l

        beta_cumu = 0
        #print ('the total beta is: {:.2e}'.format(beta_tot*0.5))
        max_iter = 0 
        for beta in beta_log:
            beta_cumu += beta
            #print('beta_cumu {:.2e}'.format(beta_cumu))
            if beta_cumu >= 0.5* beta_l_tot:
                #print (True, counter)
                break
            max_iter += 1
        return max_iter
    

            
