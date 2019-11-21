### Importing packages for SVM, XGB, and regression ###
from sklearn import svm
import xgboost as xgb
from sklearn. metrics import r2_score, mean_squared_error
from sklearn import linear_model

"""
Documentation:
- Linear models:https://scikit-learn.org/stable/modules/linear_model.html
- Linear support vector regression: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html#sklearn.svm.LinearSVR
- Gradient boosting regression: https://xgboost.readthedocs.io/en/latest/python/python_api.html
"""

# Might include own class for supervised/unsupervised learning

class Regression():
    """
    ----------------------------------------
    A class for different regression methods
    ----------------------------------------
    Parameters:
    Desig nmatrix: X (m elements,n features)
    Target data = y (m elements)

    """
    def __init__(self,X,y):
        self.data = X
        self.target = y

    def svm(self):
        """
        Regression using Support Vector Machines (SVM)
        """
        self.clf = svm.LinearSVR()
        fit = self.clf.fit(self.X, self.y)
        self.weights = self.clf.coeff_

    def linear(self,regtype):
        """
        -----------------------
        Linear regression method
        ------------------------
        Parameters:
        regtype: Either Ridge, LinearRegression,Lasso
        """
        self.clf = linear_model.regtype()
        fit = self.clf.fit(self.X,self.y)
        self.weights = self.clf.coeff_
        self.inter

    def weak_regressor(self):
        """
        Weak regressor method, gradient boosting
        """
        self.clf = xgb.XGBRegressor()
        fit = self.clf.fit(self.X,self.y)
        self.weights = self.clf.coef_  # only for booster = gblinear (linear learners)

    def evalute(self,x_test):
        """
        Perform prediction.
        Use test or validation
        """
        pred = self.clf.predict(x_test)
