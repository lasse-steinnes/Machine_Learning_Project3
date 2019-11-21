"""
Main script to fit model, storing table of information
(toi) (and possibly some figures).
"""

# Importing packages
from sklearn.model_selection import train_test_split


# Train and test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)
