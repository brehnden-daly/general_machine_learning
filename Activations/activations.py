from enum import Enum
import numpy as np


class Activations(Enum):
    # activation function and its derivative
    def tanh(self, x):
        return np.tanh(x);
    
    def tanh_prime(self, x):
        return 1-np.tanh(x)**2;