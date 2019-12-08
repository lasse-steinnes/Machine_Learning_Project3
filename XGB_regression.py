
"""
Code for XGB_regression
"""
from methods import Regression
from helper_functions import DataWorkflow, CV
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from  tqdm import tqdm
from sklearn.linear_model import BayesianRidge
from BaysianOptimizer import BaysianMaximization
from sklearn.metrics import mean_squared_error
from pathlib import Path
