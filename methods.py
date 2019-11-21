### Importing packages for SVM, XGB, and regression ###
from sklearn import svm
import xgboost as xgb
from sklearn. metrics import r2_score, mean_squared_error
from sklearn import linear_model

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

    def svm(self,Kfold = False):
        """
        Regression using Support Vector Machines (SVM)
        """
        self.clf = svm.SVR()
        fit = self.clf.fit(X, y)

    def linear(self,regtype,Kfold = False):
        """
        -----------------------
        Linear regression method
        ------------------------
        Parameters:
        Type: Either Ridge, LinearRegression,Lasso
        """
        self.clf = linear_model.regtype()
        fit = self.clf.fit(X,y)

    def weak_regressor(self, Kfold = False):
        """
        Weak regressor method
        """
        self.clf = xgb.XGBRegressor()
        fit = clf.self.fit()

    def prediction(self):
        """
        Perform prediction.
        Use test or validation
        """
        pred = self.clf.prediction()
