### Third party imports
import warnings
import numpy as np
from numba import cuda
import time

from numba.core.errors import NumbaPerformanceWarning
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)


### First party imports
from Pre_Processing.Pre_Process import Pre_Process_Inputs
from Layers.layers import *
from Activations.activations import Activations
from Costs.costs import Costs
from Models.Neural_Nets.mlp import *

from Models.Neural_Nets.cuda_functions import *





# DATA
# x_train = np.reshape(np.arange(0,0.9, 0.01), (90,1))
# y_train = np.sin(x_train)

# x_test = np.reshape(np.arange(0.91,1.0, 0.01), (9,1))
# y_test = np.sin(x_test)

test_prices = np.reshape(np.arange(0, 90, 0.1), (900,1))
test_prices = np.sin(test_prices)

further_prices = np.reshape(np.arange(90,180, 0.1), (900,1))
further_prices = np.sin(further_prices)


# MODEL
mlp = MLP()
# mlp.add( Fully_Connected_Layer(x_train.shape[1], 30) )
mlp.add( Fully_Connected_Layer(test_prices.shape[1], 30) )
mlp.add( Activation_Layer(getattr(Activations, "tanh"), getattr(Activations, "tanh_prime")) )

mlp.add( Fully_Connected_Layer(30, 50) )
mlp.add( Activation_Layer(getattr(Activations, "tanh"), getattr(Activations, "tanh_prime")) )

mlp.add( Fully_Connected_Layer(50, 50) )
mlp.add( Activation_Layer(getattr(Activations, "tanh"), getattr(Activations, "tanh_prime")) )

mlp.add( Fully_Connected_Layer(50, 50) )
mlp.add( Activation_Layer(getattr(Activations, "tanh"), getattr(Activations, "tanh_prime")) )

mlp.add( Fully_Connected_Layer(50, 25) )
mlp.add( Activation_Layer(getattr(Activations, "tanh"), getattr(Activations, "tanh_prime")) )

mlp.add( Fully_Connected_Layer(25, 15) )
mlp.add( Activation_Layer(getattr(Activations, "tanh"), getattr(Activations, "tanh_prime")) )

mlp.add( Fully_Connected_Layer(15, 1) )
mlp.add( Activation_Layer(getattr(Activations, "tanh"), getattr(Activations, "tanh_prime")) )


# TRAIN
# mlp.use( getattr(Costs, "mse"), getattr(Costs, "mse_prime") )
# mlp.fit( x_train, y_train, batch_size=64, epochs=100, learning_rate=0.01 )
mlp.use( getattr(Costs, "port_delta"), getattr(Costs, "port_delta_prime") )
mlp.fit( test_prices, test_prices, batch_size=64, epochs=1000, learning_rate=0.01 )
# batches = mlp.fit( test_prices, test_prices, batch_size=64, epochs=1000, learning_rate=0.01 )


# TEST
# out_train = np.reshape( mlp.predict(x_train), (x_train.shape[0], y_train.shape[1]) )
# out_test = np.reshape( mlp.predict(x_test), (x_test.shape[0], y_test.shape[1]) )
out_train = np.reshape( mlp.predict(test_prices), (test_prices.shape[0], test_prices.shape[1]) )
train_port_val = 0
train_port_deltas = [0]
for p in range(1, test_prices.shape[0]):
    delta = out_train[p]*(test_prices[p]-test_prices[p-1])
    train_port_deltas.append(delta)
    train_port_val+=delta
    
    
out_test = np.reshape( mlp.predict(further_prices), (further_prices.shape[0], further_prices.shape[1]) )
test_port_val = 0
test_port_deltas = [0]
for p in range(1, further_prices.shape[0]):
    delta = out_test[p]*(further_prices[p]-further_prices[p-1])
    test_port_deltas.append(delta)
    test_port_val+=delta
















