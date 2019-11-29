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

# Create the dataset
X, y = importData()
X_train, X_test, y_train, y_test = AdaBoost.shuffleAndsplit(X, y)

rng = np.random.RandomState(1)

# Fit regression model
depth = 3
regr_1 = tree.DecisionTreeRegressor(max_depth=depth)

regr_2 = AdaBoostRegressor(tree.DecisionTreeRegressor(max_depth=depth),
                          n_estimators=81, random_state=rng)

regr_1.fit(X_train, y_train)
regr_2.fit(X_train, y_train)

# Predict
y_1 = regr_1.predict(X_train)
y_2 = regr_2.predict(X_train)

print( X_train.shape)
print( y_train.shape)
# Plot the results
plt.figure()
#plt.scatter(X_train, y_train, c="k", label="training samples")
plt.plot(y_train, y_1, ".", label="decision tree", linewidth=2)
plt.plot(y_train, y_2, "r.", label="adaboost", linewidth=2)
plt.xlabel("y training data")
plt.ylabel("predicted")
plt.title("Boosted Decision Tree Regression")
plt.legend()
plt.show()