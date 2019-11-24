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

    Must have preprocessed input, since fit_intercept = False
    """
    def __init__(self,X,y):
        self.data = X
        self.target = y

    def svm(self, epsilon = 0.0, penalty = 1.0, tol = 0.0001):
        """
        ---------------------------------------------
        Regression using Support Vector Machines (SVM)
        ---------------------------------------------
        Parameters:
        epsilon: parameter in loss function
        penalty: L2-penalty for error term. The larger, the less regularisation is used.
        tol: Tolerance for stopping criteria
        loss: Set to epsilon_insensitive, standard SVR
        """
        self.penalty = penalty
        self.eps = epsilon
        self.tol = tol

        self.clf = svm.LinearSVR(epsilon=self.eps, tol=self.tol, C=self.penalty, loss='epsilon_insensitive', fit_intercept=False, max_iter=10e5)
        fit = self.clf.fit(self.X, self.y)
        self.weights = self.clf.coeff_

    def linear(self,regtype, lam = 0.01, tol = 0.001):
        """
        -----------------------
        Linear regression method
        ------------------------
        Parameters:
        regtype: Either Ridge, LinearRegression,Lasso
        lam = Regularisation
        tol = tolerance for stopping criteria

        Set default to LinearRegression
        """
        self.lam = lam
        self.tol = tol

        # Choosing linear regression method
        #l1 regularisation
        if regtype == 'Lasso':
            self.clf = linear_model.Lasso(alpha=self.lam, max_iter=10e5,tol = self.tol, precompute = True, fit_intercept = False)

        #l2 regularisation
        elif regtype =='Ridge':
            self.clf = linear_model.Ridge(alpha = self.lam,fit_intercept = False, solver ='svd')

        # Ordinary least squares
        else:
            self.clf = linear_model.LinearRegression(fit_intercept = False)

        fit = self.clf.fit(self.X,self.y)
        self.weights = self.clf.coeff_

    def weak_regressor(self,booster, max_dp, n,eta = 0.1,gamma = 0, alpha = 0, lam = 1):
        """
        -------------------------------------
        Weak regressor method, gradient boosting
        -------------------------------------

        booster: Should be a string, either gblinear,gbtree or dart
        max_dp: max depth of tree
        n: Number of estimators
        eta: learning rate
        gamma: "Minimum loss reduction required to make a further partition on a leaf node of the tree"
        alpha: l1,regularisation parameter
        lambda: l2,regularisation paramter
        """
        self.n = n; self.gamma = gamma
        self.booster = booster; self.max_dp = max_dp
        self.eta = eta; self.alpha = alpha
        self.lam = lam

        self.clf = xgb.XGBRegressor(max_depth = self.max_dp, learning_rate = self.eta, n_estimators = self.n, verbosity = 1, gamma = self.gamma, reg_alpha = self.alpha, \
                    reg_lambda = self.lam, booster= self.booster, min_child_weight = 1, subsample = 1,colsample_bytree= 1, num_parallel_tree = 1)

        fit = self.clf.fit(self.X,self.y)
        if booster == 'gblinear':
            self.weights = self.clf.coef_  # only for linear learners

    def evalute(self,x_test):
        """
        Perform prediction.
        Use test or validation
        """
        pred = self.clf.predict(x_test)
