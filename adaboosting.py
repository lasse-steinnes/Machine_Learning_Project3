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
        
        #X, y = importData()
        self.iterations = iterations
        self.depth = depth
        
        self.X_train, self.y_train, self.X_test, self.y_test = X_train, y_train, X_test, y_test #AdaBoost.shuffleAndsplit(self, X, y)
        
        self.X_eval, self.y_eval = X_eval, y_eval
            
        self.n = self.X_train.shape[0]
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
    
    def training(self, loss_func):
        '''
        trains/begins the iterative process of the adaboost
        
        loss func:  - loss function is a string, 'square', 'linear' or 'exponential'.
        '''
        W = np.array([1 for i in range(0, self.n)]) # initialise the weights as 1/n
        self.beta = [] #list of all betas
        self.train_predict_list = [] #list of all training prediction arrays 
        self.test_predict = np.array([0.0 for i in range(0,len(self.y_test))])
        self.train_predict = np.array([0.0 for i in range(0,len(self.y_train))])
        if type(self.X_eval) != type(None):
            self.eval_predict = np.array([0.0 for i in range(0,len(self.y_eval))])
        self.trees = []
        
        for i in range(0, self.iterations+1): 
            # renormalise the weights
            W_norm = W / np.sum(W)
            
            # fit a weak decision tree
            reg_weak = tree.DecisionTreeRegressor(max_depth = self.depth)
            reg_weak.fit(self.X_train, self.y_train, sample_weight = W_norm)
            
            # predict on train and test
            train_predict_single = reg_weak.predict(self.X_train)
            test_predict_single = reg_weak.predict(self.X_test)
            self.train_predict_list.append(train_predict_single)
            
            # decide on loss function
            if loss_func == 'linear':
                loss = AdaBoost.linear(self, train_predict_single, self.y_train)
            elif loss_func == 'square':
                loss = AdaBoost.square(self, train_predict_single, self.y_train)
            elif loss_func == 'exponential':
                loss = AdaBoost.exponential(self, train_predict_single, self.y_train)
            
            #find weighted loss and update prediction
            loss_ave = np.sum(loss * W_norm)
            beta = loss_ave / (1-loss_ave)
            self.beta.append(beta)
            self.test_predict += beta* test_predict_single
            self.train_predict += beta* train_predict_single
            
            #breaking function - not well understood
            if loss_ave > 0.5:
                print ('breaking Adaboost')
                break
            
            #update weights
            W = W_norm * (beta**(1-loss))
            self.trees.append(reg_weak)

    def evaluate(self, best_mse, best_trees, best_betas, betaStopping = False):
        if betaStopping == True:
            max_iteration = AdaBoost.beta_eval(self)
        else:
            max_iteration = self.iterations
            
        train_predict_all = np.zeros(len(self.beta))
        for i in range(0, max_iteration):
            predict = self.train_predict_list[i] * self.beta[i]
            train_predict_all += predict
        
        MSE_train = mean_squared_error(self.y_train, self.train_predict)
        MSE_test = mean_squared_error(self.y_test, self.test_predict)
        R2_train = r2_score(self.y_train, self.train_predict)
        R2_test = r2_score(self.y_test, self.test_predict)
        
        #store the best betas and trees for the final evaluation
        if MSE_test < best_mse:
            best_trees = self.trees
            best_betas = self.beta
        
        #predict on evaluation data
        if type(self.X_eval) != type(None):
            for i in range(0,self.iterations+1):
                self.eval_p += self.beta[i] * self.trees[i].predict(self.X_eval)
            MSE_eval = mean_squared_error(self.y_eval, self.eval_p)
            R2_eval = r2_score(self.y_eval, self.eval_p)
            return MSE_train, R2_train, MSE_test , R2_test, MSE_eval, R2_eval, best_mse, best_trees, best_betas
        
        else:
            return MSE_train, R2_train, MSE_test , R2_test, best_mse, best_trees, best_betas
            
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
    

    

            
