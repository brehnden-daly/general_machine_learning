### Third party imports
import warnings
import numpy as np
from numba import cuda
import time

from numba.core.errors import NumbaPerformanceWarning
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)


### First party imports
from Pre_Processing.Pre_Process import Pre_Process_Inputs
from Activations.activations import Activations
from Costs.costs import Costs
from Models.Neural_Nets.Custom import *

from Models.Neural_Nets.cuda_functions import *





# training data
x_train = np.reshape(np.arange(0,0.9, 0.01), (90,1))
y_train = np.square(x_train)

x_test = np.reshape(np.arange(0.91,1.0, 0.01), (9,1))
y_test = np.square(x_test)




# mlpwork
mlp = MLP()
mlp.add( Fully_Connected_Layer(x_train.shape[1], 30) )
mlp.add( Activation_Layer(getattr(Activations, "tanh"), getattr(Activations, "tanh_prime")) )

mlp.add( Fully_Connected_Layer(30, 100) )
mlp.add( Activation_Layer(getattr(Activations, "tanh"), getattr(Activations, "tanh_prime")) )

mlp.add( Fully_Connected_Layer(100, 30) )
mlp.add( Activation_Layer(getattr(Activations, "tanh"), getattr(Activations, "tanh_prime")) )

mlp.add( Fully_Connected_Layer(30, 10) )
mlp.add( Activation_Layer(getattr(Activations, "tanh"), getattr(Activations, "tanh_prime")) )

mlp.add( Fully_Connected_Layer(10, 1) )
mlp.add( Activation_Layer(getattr(Activations, "tanh"), getattr(Activations, "tanh_prime")) )

# train
mlp.use( getattr(Costs, "mse"), getattr(Costs, "mse_prime") )
mlp.fit( x_train, y_train, epochs=3000, learning_rate=0.01 )

# test
out_train = np.reshape( mlp.predict(x_train), (x_train.shape[0], y_train.shape[1]) )
out_test = np.reshape( mlp.predict(x_test), (x_test.shape[0], y_test.shape[1]) )
















