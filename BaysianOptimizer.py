import tensorflow as tf 


import numpy as np

from warnings import catch_warnings
from warnings import simplefilter

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split

from scipy.stats import norm

from tqdm import tqdm

class BaysianMaximization:
    def __init__(self, model, search_space_categorical, search_space_interval, search_space_complex = None, gp_kargs=None, verbose_level=0 ):
        """
        This class maximises a model score function with respect to the hyperparameters of a given model and training data.
        Initialize all necessary search space related operations and the Gaussian Process model GP
        Args:
            model:                      Any model function to optimize which must have
                                            X_train, X_test, y_train, y_test as *args
                                            the search_dim_names as **kargs and 
                                        and return the scalar model score to maximize (i.e. -MSE)

            search_space_categorical:   dict with search_dim_name : categorical_search_item
                                        categorical_search_item is a list of itmes which will be randomly drawn (uniform probability)
                                        or a tuple of items and their weights both as list

            search_space_interval:      dict with search_dim_name : (interval_start, interval_stop)
                                        draw random uniform samples from [interval_start, interval_stop)
            

        Kargs:
            search_space_complex:       dict with more complex sampling search_dim_name : (func , **kargs)
                                            func must take arg num_sample and return model kargs_value, scalar_map_value 

            gp_kargs:                   dict with kargs to sklearn GaussianProcessRegressor
                                        default None

            verbose_level:              0 -> no output except bars
                                        1 -> final stats
                                        2 -> all
        """
        self.model = model
        self.categorical = search_space_categorical
        #creat simple lookup table to map to numerical values by index in interval [0,1]
        self.categorical_encoding = { name:{ opt: (i + 1)/len(self.categorical[name]) for i, opt in enumerate(values)} 
                                                                                      for name, values in self.categorical.items()}
        self.interval    = search_space_interval
        self.complex = search_space_complex
        self.has_complex = False
        if gp_kargs != None:
            self.gp_estimator = GaussianProcessRegressor(**gp_kargs)
        else:
            self.gp_estimator = GaussianProcessRegressor()
        #store for hyperparameters and odel score
        self.model_score = []
        self.verbose = verbose_level

    def SetData(self, data, target, test_fraction = 0.2, eval_fraction = 0.1):
        """
        set the data and target data for the model to optimize
        perform train, test and evaluation split
        Args:
            data: numpy array with shape (samples,features)
            target numpy array with shape(samples, predictors)
        Kargs:
            fraction of data goin into test, train, and eval set
        """
        tot_split = test_fraction + eval_fraction
        X_train, X_temp, y_train, y_temp = train_test_split(data, target, test_size = tot_split)
        X_test, X_eval, y_test, y_eval = train_test_split(X_temp, y_temp, test_size =eval_fraction/tot_split)
        self.train = [X_train, X_test, y_train, y_test]
        self.eval = [X_train, X_eval, y_train, y_eval]

    def InitialGuesses(self, num_guess):
        """
        Initial num_guesses to the GP regressor
        """
        temp_par = []
        for i in tqdm(range(num_guess)):
            temp = BaysianMaximization.DrawSample(self, num_samples = 1)
            temp_par.append(temp)
            if i == 0:
                self.hyperpar = BaysianMaximization.ToNumbers(self, temp)
            else:
                self.hyperpar = np.vstack([self.hyperpar, 
                                         BaysianMaximization.ToNumbers(self, temp)])
            temp["sequential"].update(temp["interval"])
            temp = {name: value[0] for name, value in temp["sequential"].items()}
            self.model_score = np.append(self.model_score, self.model(*self.train, **temp))

        self.gp_estimator.fit(self.hyperpar, self.model_score)
        pred, _ = BaysianMaximization.Predict(self, self.hyperpar)

        self.maximal_predicted_score= np.max(pred)
        max_ind = np.argmax(self.model_score)
        self.maximal_score = self.model_score[max_ind] 
        self.best_model_kargs = BaysianMaximization.JoinDicts(self, temp_par[max_ind], 0) 
        if self.verbose >0:
            print("GP R2: %.4f"%self.gp_estimator.score(self.hyperpar, self.model_score))

    def Predict(self, params):
        """
        Predict the score value and uncertanty of params with the current GP model
        """
        # catch any warning generated when making a prediction
        with catch_warnings():
            # ignore generated warnings
            simplefilter("ignore")
            return self.gp_estimator.predict(params, return_std=True)

    def DrawSample(self, num_samples = 1):
        """
        Draws random sample of the search space
        returns kargs for the model function
        """
        to_encode = {"sequential":dict(), "interval":dict()}
        if self.complex != None:
            self.has_complex = True
            to_encode.update({ "complex":dict()})
        for name, entry in self.categorical.items():

            if type(entry) != tuple:
                #draw uniformly
                temp = np.random.choice(entry, size=num_samples)
            else:
                temp = np.random.choice(entry[0],size=num_samples, p = entry[1])
            to_encode["sequential"].update({name : temp})

        for name, entry in self.interval.items():
            low, up = entry
            temp = low + (up-low) * np.random.rand(num_samples)
            to_encode["interval"].update({name : temp})
        if self.has_complex :
            for name, entry in self.complex.items():
                to_encode["complex"].update({name : entry[0](num_samples, **entry[1])})

        return to_encode

    def ToNumbers(self, to_encode):
        """
        convert a dictonary of search_dim_names : values to a numeric vector with each element [-1,1]
        """
        ret = []
        for name, values in to_encode["sequential"].items():
            ret.append( [self.categorical_encoding[name][sample] for sample in values])
        for name, values in to_encode["interval"].items():
            ret.append(values/max(np.abs(self.interval[name])) )
        if self.has_complex:
            for _, values in to_encode["complex"].items():
                ret.append(values[1])
        return np.transpose(ret)

    def NextTry(self, num_random, exploration = 0.01):
        """
        find the next set of hyperparameters on which the model is evaluated
        suggested num_random variables
        returns most promissing kargs for model and append numeric representation to hyperpars
        """
        try_keys = BaysianMaximization.DrawSample(self, num_samples=num_random)
        try_map = BaysianMaximization.ToNumbers(self, try_keys)
        pred, std = BaysianMaximization.Predict(self, try_map)

        Z = pred - self.maximal_predicted_score - exploration
        expected_improvment = Z*norm.cdf(Z/(std + 1E-9),) + std*norm.pdf(Z/(std+1E-9))

        max_ind = np.argmax(expected_improvment)
        if self.verbose>1:
            print("Expected Improvment: %.4f"% expected_improvment[max_ind])

        self.hyperpar = np.vstack([self.hyperpar, try_map[max_ind]])
        
        #creat kargs dict
        return BaysianMaximization.JoinDicts(self, try_keys, max_ind)

    def OptimizeHyperPar(self, cycles =1000, samples_per_cycle = 1000, exploration = 0.01):
        """
        Optimizing the model's hyperparameters
        """
        for i in tqdm(range(cycles)):
            model_kargs = BaysianMaximization.NextTry(self, samples_per_cycle, exploration) 
            self.model_score = np.append(self.model_score, self.model(*self.train, **model_kargs))
            improv = self.model_score[-1] - self.maximal_score
            if improv > 0:
                self.maximal_score = self.model_score[-1]
                self.best_model_kargs = model_kargs
            if self.verbose >1:
                print("Acutual Improvment: %.4f"% improv )
            pred, _ = BaysianMaximization.Predict(self, self.hyperpar)
            self.maximal_predicted_score= np.max(pred)
        if self.verbose >0:
            print("GP R2: %.4f"%self.gp_estimator.score(self.hyperpar, self.model_score))
            print("Best Model Score on test: %.4f" %self.maximal_score)
            print("Best Model Score on evaluation: %.4f" %self.model(*self.eval, **self.best_model_kargs))

    def JoinDicts(self, dict_to_join, max_ind):
        ret= { name: value[max_ind] for name, value in dict_to_join["sequential"].items()}
        ret.update( { name: value[max_ind] for name, value in dict_to_join["interval"].items()})
        if self.has_complex:
            ret.update( { name: value[0][max_ind] for name, value in dict_to_join["complex"].items()})
        return ret