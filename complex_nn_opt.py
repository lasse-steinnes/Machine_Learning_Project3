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
from tensorflow.keras.layers import Dense, Dropout 
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
X_train, X_test, y_train, y_test = train_test_split(X, y.to_numpy(), test_size =0.2)
# %%
#hyper par search space

optimizer =['Adagrad', 'Adam', 'SGD']
num_epochs_range = (5, 40)
batch_size_range =(32, 256)
learning_rate_range= (0.01, 0.001)


def neural_net(X_train, X_test, y_train, y_test, optimizer ='Adam', epochs =10, learning_rate = 0.01, batch_size =64
            , layers = [Dense(500, input_dim=X.shape[1], activation ='tanh'), Dense(500, activation ='relu'),
                        Dense(500, activation ='tanh'), Dropout(0.2), Dense(100, activation ='relu'),
                        Dense(20, activation ='relu'),  Dropout(0.2), Dense(4, activation ='relu')],
                        graphics = False):
    tf.keras.backend.clear_session()
    model = tf.keras.models.Sequential()
    for layer in layers:
        model.add(layer)
    model.add(Dense(1, activation ='relu', name='Out'))

    model.compile(optimizer=optimizer,
                loss='MSE', metric=['MSE'], learning_rate=learning_rate, batch_size=batch_size)
    if graphics:
        logdir="logs/BaysOpt/" 
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
        cb = [tensorboard_callback]
    else:
         cb = None

    hist = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=epochs, verbose=0, callbacks = cb)
    neg_mse = -model.evaluate(X_test,  y_test, verbose=2, callbacks = cb, batch_size=batch_size)
    
    if graphics:
        tf.keras.utils.plot_model(model, to_file='./model.png', show_shapes=True, show_layer_names=True,
                                         rankdir='LR', expand_nested=False, dpi=96)
        plt.figure(figsize=(10,10))
        plt.plot(hist.history["loss"])
        plt.xlabel("Epoch" , fontsize = 32)
        plt.ylabel(r"MSE / T$_{max}$", fontsize =32)
        plt.tick_params(size =24, labelsize=26 )
        plt.legend(["train", "test"], loc='best', fontsize = 28)
        plt.tight_layout()
        plt.savefig("./train_hist.pdf")

    del model
    return neg_mse

def layer_sample(num_samples, X_features = 10, 
                 activ_fun= ['tanh', 'relu', 'sigmoid'], layer_range=(1,15),
                 num_node= np.arange(2,700,25), layer_func = [[Dense, Dropout],[0.8,0.2]],
                 dropout_range =(0.05, 0.08)):
    layer = []
    map_value =[]
    
    actv_dict = {name : i+1 for i, name in enumerate(activ_fun)}
    max_map = layer_range[1]*num_node[-1]*len(activ_fun)
    for i in range(num_samples):
        num_layer = np.random.randint(*layer_range)
        num_nodes = np.random.choice(num_node, size =num_layer)
        activation =np.random.choice(activ_fun, size = num_layer)
        layers = np.random.choice(layer_func[0], size=num_layer, p = layer_func[1] )
        temp_layer =[]
        temp_map = 0
        for j, lay in enumerate(layers):
            if j ==0:
                temp_layer.append(Dense(num_nodes[j], activation=activation[j], input_dim = X_features, name=activation[j] + str(j) ))
            elif lay.__name__ =='Dropout':
                low, up = dropout_range
                dropout = low +  (up -low)*np.random.rand()
                temp_layer.append(lay(dropout, name = 'Dropout' + str(j)))
                temp_map += dropout
            else:
                temp_layer.append(lay(num_nodes[j], activation = activation[j], name=activation[j] + str(j) ))
                temp_map += num_nodes[j]*actv_dict[activation[j]]
        layer.append(temp_layer)
        map_value.append(temp_map/max_map)

    return layer, map_value

optim = BaysianMaximization(neural_net,
                        {"optimizer": optimizer, "epochs":np.arange(10,20), "batch_size" :np.arange(32,256,8)},
                        {"learning_rate":(0.05, 0.0005)},
                        search_space_complex={"layers": (layer_sample, {"X_features" : X.shape[1], "activ_fun": ['tanh', 'relu', 'sigmoid'],
                                                                        "layer_range":(5,15), "num_node": np.arange(2,700,25), 
                                                                        "layer_func" : [[Dense, Dropout],[0.8,0.2]], "dropout_range" :(0.05, 0.45)}) })
optim.SetData(X, y.to_numpy())
optim.InitialGuesses(50)
optim.OptimizeHyperPar(100, samples_per_cycle= 1000, exploration = 0.01)
mse=neural_net(X_train, X_test, y_train, y_test, graphics=True, **optim.best_model_kargs)
print("Temp. uncertainty from MSE: +-", ymax*np.sqrt(-mse)," K")
# %%
