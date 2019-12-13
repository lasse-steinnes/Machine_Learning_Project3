# -*- coding: utf-8 -*-
"""
test adaboost
"""
from helper_functions import scaler, MSE, importData
from adaboosting import AdaBoost
from sklearn.ensemble import AdaBoostRegressor
import matplotlib.pyplot as plt
from sklearn import tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path

def shuffleAndsplit( X, y):
    curr_seed= 0
    np.random.seed(curr_seed)
    np.random.shuffle(X)
    np.random.seed(curr_seed)
    np.random.shuffle(y)
    
    #X = X[0:1000] #algorithm testing with smaller samples than full data
    #y = y[0:1000]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
    
    return X_train, X_test, y_train, y_test
    
# Create the dataset
X, y = importData()
X_train, X_test, y_train, y_test = shuffleAndsplit(X, y)

rng = np.random.RandomState(1)

# Fit regression model
depth = 15
regr_1 = tree.DecisionTreeRegressor(max_depth=depth)

regr_2 = AdaBoostRegressor(tree.DecisionTreeRegressor(max_depth=depth),
                          n_estimators=50, random_state=rng, loss = 'linear')

regr_1.fit(X_train, y_train)
regr_2.fit(X_train, y_train)

# Predict
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

mse1 = mean_squared_error(y_test, y_1)
r21 = r2_score(y_test, y_1)
mse2 = mean_squared_error(y_test, y_2)
r22 = r2_score(y_test, y_2)

print( X_test.shape)
print( y_test.shape)

  
plt.fontsize = 32
filep = Path("./Results/adaboost/")
# Plot the results
plt.figure(figsize=(10,10))
ymax = y_test[np.argmax(y_test)]
plt.plot([0,ymax],[0,ymax], linestyle ='--')
#plt.scatter(X_train, y_train, c="k", label="training samples")
#plt.plot(y_test, y_1,'r.' , label="decision tree \n MSE: %.3f, R2: %.2f"% (mse1,r21), linewidth=2)
plt.scatter(y_test, y_2,  label="adaboost \n MSE: %.3f, R2: %.2f"%(mse2, r22))
plt.xlabel('act. T$_c$ in K', fontsize =32)
plt.ylabel("pred. T$_c$ in K ", fontsize= 32)
#plt.title("Boosted Decision Tree Regression")
plt.tick_params(size =24, labelsize=26)
plt.tight_layout()
plt.legend(fontsize = 20)
plt.savefig(filep/'sklearn_pred_vs_act.pdf')
plt.show()