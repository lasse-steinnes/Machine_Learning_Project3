# -*- coding: utf-8 -*-
"""
adaboost example
"""
from adaboosting import AdaBoost
import matplotlib.pyplot as plt

ada = AdaBoost(100, 3)
ada.initiateBoost('linear')
y_ensemble = ada.ensemble_predict(False)

plt.figure()
plt.plot(ada.y_train, y_ensemble, ".", label="adaboost", linewidth=2)
plt.plot(ada.y_test, ada.test_p, '.', label='test adaboost')
plt.title("Boosted Decision Tree Regression")
plt.xlabel("y training data")
plt.ylabel("predicted")
plt.legend()
plt.show()