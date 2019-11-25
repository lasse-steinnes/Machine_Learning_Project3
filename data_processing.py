#%%
from opt_model import BaysianMaximization
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])


# %%
def get_Xy( data_path = Path("./Data/")):   
    temp1 = pd.read_csv(data_path/"train.csv")
    temp2 = pd.read_csv(data_path/"unique_m.csv")
    temp2.drop( columns=["critical_temp",	"material"], inplace=True)

    df = pd.concat([temp1,temp2], axis =1 )

    y = df["critical_temp"]
    X = df.drop(columns="critical_temp")

    return X, y

def scale_y(y):
    """
    scale critical temperature to [0,1]
    return ymax for proper rescaling
    """
    ymax = y.max()
    return y/ymax, ymax
# %%
X,y = get_Xy()
y, ymax = scale_y(y)
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
#%%
sns.heatmap(X.corr())
plt.show()
# %%
print((len(X.index),len(X.columns)))
pca = PCA(n_components=0.95)#95% of variance should be explainable by reduced data
X = pca.fit_transform(X)
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y.to_numpy(), test_size=0.2)

# %%
#hyper par search space
num_layers_range = (1, 15)
num_nodes_range = (2, 1000)
activation =['tanh', 'relu', 'sigmoid']
layers =['Dense', 'Dropout']
dropout_range = (0.01, 0.8)

optimizer =['Adagrad', 'Adam', 'SGD']
num_epochs_range = (5, 40)
batch_size_range =(32, 256)
learning_rate_range= (0.01, 0.001)


def neural_net(X_train, X_test, y_train, y_test, optimizer ='Adam', epochs =10, learning_rate = 0.01, batch_size =64 ):
    model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(X.shape[1], activation ='tanh'),
                tf.keras.layers.Dense(500, activation ='relu'),
                tf.keras.layers.Dense(500, activation ='tanh'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(100, activation ='relu'),
                tf.keras.layers.Dense(20, activation ='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(4, activation ='relu'),
                tf.keras.layers.Dense(1, activation ='relu')]
            )
    model.compile(optimizer=optimizer,
                loss='mse', learning_rate=learning_rate, batch_size=batch_size)

    model.fit(X_train, y_train, epochs=epochs, verbose=0)

    return -model.evaluate(X_test,  y_test, verbose=2)

optim = BaysianMaximization(neural_net,
                        {"optimizer": optimizer, "epochs":np.arange(5,20), "batch_size" :np.arange(32,256,8)},
                        {"learning_rate":(0.1, 0.0001)})
optim.SetData(X, y.to_numpy())
optim.InitialGuesses(2)
optim.OptimizeHyperPar(5, samples_per_cycle= 1000)
print(optim.best_model_kargs)
# %%
