### Importing packages for SVM, XGB, and regression ###
from sklearn import svm
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn. metrics import r2_score, mean_squared_error
from sklearn import linear_model
from pathlib import Path
import pandas as pd

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
    def __init__(self,X=None,y=None):
        self.data = X
        self.target = y
        self.scaled = False
        self.splited = False
        self.has_evaluation = False

    def svm(self,type_, epsilon = 0.0, penalty = 1.0, tol = 0.0001):
        """
        ---------------------------------------------
        Regression using Support Vector Machines (SVM)
        ---------------------------------------------
        Parameters:
        epsilon: Parameter in loss function. Defines margin where no penalty is given to errors.
        penalty: L2-penalty for error term. The larger, the less regularisation is used.
        tol: Tolerance for stopping criteria
        loss: Set to epsilon_insensitive, standard SVR
        """
        self.penalty = penalty
        self.eps = epsilon
        self.tol = tol
        self.clf = svm.LinearSVR(epsilon=self.eps, tol=self.tol, C=self.penalty, loss='epsilon_insensitive', fit_intercept=False, max_iter=10e5)
        fit = self.clf.fit(self.X,self.y)
        self.weights = self.clf.coef_
        pred = Regression.predict(self, self.X)
        MSE = mean_squared_error(self.y, pred)
        return MSE, self.clf.score(self.X, self.y)


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
        if regtype == 'LASSO':
            self.clf = linear_model.Lasso(alpha=self.lam, max_iter=10e5,tol = self.tol, precompute = True, fit_intercept = False)

        #l2 regularisation
        elif regtype =='Ridge':
            self.clf = linear_model.Ridge(alpha = self.lam,fit_intercept = False, solver ='svd')

        # Ordinary least squares
        else:
            self.clf = linear_model.LinearRegression(fit_intercept = False)
        fit = self.clf.fit(self.X,self.y)
        self.weights = self.clf.coef_
        pred = Regression.predict(self, self.X)
        MSE = mean_squared_error(self.y, pred)
        return MSE, self.clf.score(self.X, self.y)

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
        else:
            self.weights = self.clf.feature_importances_
        pred = Regression.predict(self, self.X)
        MSE = mean_squared_error(self.y, pred)
        return MSE, r2_score(self.y,pred)

    def predict(self,x_test, rescaled=False):
        """
        Perform prediction. If rescaled =True, the model output is rescaled from [0,1] to to [0, ymax] (only applicable if scale was used)
        """
        pred = self.clf.predict(x_test)
        if rescaled:
            return self.ymax *pred
        else:
            return pred

    def evaluation(self, X=None, y=None, eval = False):
        """
        perform evaluation of the model on given X,y
        if both None use test or evaluation
        returns the MSE and R2
        """
        if X.all() == None and y.all() == None:
            if eval:
                X = self.X_eval
                y = self.y_eval
            else:
                X = self.X_test
                y = self.y_test
        pred = Regression.predict(self,X)
        MSE = mean_squared_error(pred, y)
        R2 = self.clf.score(X,y) #score might return different things for different models!
        return MSE, R2

    def importData(self, filepath):
        """
        Imports training data train.csv from filepath
        sets X, y numpy arrays
        """
        data_path = Path(filepath) # data should be stored in folder Data
        df = pd.read_csv(data_path/'train.csv')

        self.y = df["critical_temp"].to_numpy()
        self.X = df.drop(columns = ["critical_temp"]).to_numpy()

    def scale(self):
        """
        scales X according to standard scaler
        scales y to [0,1] and keeps ymax for reversed scaling of prediction
        """
        self.scaled = True
        X_scale = StandardScaler()
        self.X = X_scale.fit_transform(self.X)
        self.ymax = self.y.max()
        self.y /= self.ymax

    def generate_polynomic_features(self, order=1):
        """
        create the design matrix with polynomial features
        """
        feature = PolynomialFeatures(degree=order,)
        self.X =feature.fit_transform(self.X)


    def train_test_eval_split(self, test_size = 0.2, eval_size = 0.1):
        """
        splits X,y in training, testing and evaluation targets
        if eval size 0 this is ignored
        sizes are relative fraction to the total sample size
        """
        self.splited = True
        self.has_evaluation = eval_size != 0

        self.X, X_temp, self.y, y_temp = train_test_split(self.X, self.y, test_size=test_size +eval_size)

        if self.has_evaluation:
            self.X_test, self.X_eval, self.y_test, self.y_eval = train_test_split(X_temp, y_temp, test_size = eval_size/(eval_size + test_size))
        else:
            self.X_test = X_temp
            self.y_test = y_temp
