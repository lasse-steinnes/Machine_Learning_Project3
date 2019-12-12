# Machine_Learning_Project3
Github-link for project 3: Superconductivity

This is a github repository with various models (SVM, linear regression, adaboost and XGBoost) in order to predict the critical temperature of known superconductors. 

Folders:
  Data: contains the files from the Kaggle page as well as the data which scraped from the same sources as the paper.
  
  Results: contains the csv files and figures from the different methods: SVM, Adaboost, Linear Regression, XGBoost
  
Files:
  BaysianOptimizer.py contains a class for maximising a models score function with respect to the models hyper-parameters.
  SVM_regression.py contains functions for the SVM model.
  adaboost_example.py contains script for an example run using the Adaboost.
  adaboosting.py contains a class 'AdaBoost' for running the adaboost.
  helper_functions.py contains functions for supporting the different classes, e.g importing data.
  linear_regression.py contains script for running linear regression functions, OLS, Ridge and LASSO as well as using Gridsearch and Bayesiansearch for hyperparameters. 
  methods.py contains a class Regression which has various regression model functions within it.
  scrape.py is a script for scaping the japanese website for the oxide superconductor data.
  test_adaboost.py is a unit test for comparing own implementation with sklearns adaboost
  test_bays.py is a script for testing the BayesianOptimizer
  
  
  
